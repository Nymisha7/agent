"""Optional voice input and output for the terminal UI.

Voice is deliberately independent from model setup.  It uses an already
configured OpenAI-compatible key when one is available, or local commands when
the host provides them.  Nothing in this module prompts for credentials,
installs software, or contacts a provider while determining availability.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

from openai import OpenAI


_DEFAULT_RECORD_SECONDS = 20
_MAX_RECORD_SECONDS = 120
_REALTIME_CHUNK_BYTES = 1280
_HF_REALTIME_VOICE_SPACE_URL = "https://huggingface.co/spaces/smolagents/hf-realtime-voice"
_HF_REALTIME_VOICE_PROVIDERS = {
    "hf",
    "hf-realtime",
    "hf-space",
    "huggingface",
    "huggingface-realtime",
    "huggingface-space",
}


@dataclass(frozen=True)
class VoiceStatus:
    input_ready: bool
    input_reason: str | None
    tts_ready: bool
    tts_reason: str | None
    auto_speak: bool
    stt_provider: str | None
    tts_provider: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "input_ready": self.input_ready,
            "input_reason": self.input_reason,
            "tts_ready": self.tts_ready,
            "tts_reason": self.tts_reason,
            "auto_speak": self.auto_speak,
            "stt_provider": self.stt_provider,
            "tts_provider": self.tts_provider,
        }


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool
    record_seconds: int
    recorder_command: tuple[str, ...] | None
    stt_command: tuple[str, ...] | None
    tts_command: tuple[str, ...] | None
    playback_command: tuple[str, ...] | None
    mode: str
    realtime_url: str | None
    realtime_session_update: dict[str, object] | None
    realtime_timeout_seconds: int
    api_key: str | None
    base_url: str | None
    stt_model: str
    tts_model: str
    tts_voice: str
    auto_speak: bool

    @classmethod
    def from_environment(cls) -> "VoiceConfig":
        return cls(
            enabled=_environment_flag("AGENT_VOICE_ENABLED", default=True),
            record_seconds=_environment_int(
                "AGENT_VOICE_RECORD_SECONDS",
                default=_DEFAULT_RECORD_SECONDS,
                minimum=1,
                maximum=_MAX_RECORD_SECONDS,
            ),
            recorder_command=_environment_command("AGENT_VOICE_RECORDER"),
            stt_command=_environment_command("AGENT_VOICE_STT_COMMAND"),
            tts_command=_environment_command("AGENT_VOICE_TTS_COMMAND"),
            playback_command=_environment_command("AGENT_VOICE_PLAYBACK"),
            mode=_voice_mode_from_environment(),
            realtime_url=_realtime_voice_url_from_environment(),
            realtime_session_update=_environment_json_object(
                "AGENT_REALTIME_VOICE_SESSION_UPDATE_JSON"
            ),
            realtime_timeout_seconds=_environment_int(
                "AGENT_REALTIME_VOICE_TIMEOUT_SECONDS",
                default=30,
                minimum=5,
                maximum=180,
            ),
            api_key=(
                os.environ.get("AGENT_VOICE_API_KEY", "").strip()
                or os.environ.get("OPENAI_API_KEY", "").strip()
                or None
            ),
            base_url=os.environ.get("AGENT_VOICE_BASE_URL", "").strip() or None,
            stt_model=os.environ.get(
                "AGENT_VOICE_STT_MODEL", "gpt-4o-mini-transcribe"
            ).strip(),
            tts_model=os.environ.get(
                "AGENT_VOICE_TTS_MODEL", "gpt-4o-mini-tts"
            ).strip(),
            tts_voice=os.environ.get("AGENT_VOICE_TTS_VOICE", "alloy").strip(),
            auto_speak=_environment_flag("AGENT_TTS_ENABLED", default=False),
        )


def voice_status() -> VoiceStatus:
    """Describe usable voice paths without touching the microphone or network."""
    try:
        config = VoiceConfig.from_environment()
    except RuntimeError as exc:
        return VoiceStatus(False, str(exc), False, str(exc), False, None, None)
    if not config.enabled:
        return VoiceStatus(
            input_ready=False,
            input_reason="Voice is disabled.",
            tts_ready=False,
            tts_reason="Voice is disabled.",
            auto_speak=False,
            stt_provider=None,
            tts_provider=None,
        )

    try:
        recorder = _recorder_command(config)
        stt_provider = _stt_provider(config)
        tts_provider = _tts_provider(config)
    except RuntimeError as exc:
        return VoiceStatus(False, str(exc), False, str(exc), False, None, None)
    input_reason = None
    if recorder is None:
        input_reason = "No supported microphone recorder was found."
    elif stt_provider is None:
        input_reason = "No speech-to-text provider is available."

    tts_reason = None
    if tts_provider is None:
        tts_reason = "No text-to-speech provider is available."

    return VoiceStatus(
        input_ready=recorder is not None and stt_provider is not None,
        input_reason=input_reason,
        tts_ready=tts_provider is not None,
        tts_reason=tts_reason,
        auto_speak=config.auto_speak and tts_provider is not None,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
    )


def transcribe_microphone() -> str:
    """Capture one bounded microphone turn and return its transcription."""
    config = VoiceConfig.from_environment()
    status = voice_status()
    if not status.input_ready:
        raise RuntimeError(status.input_reason or "Voice input is unavailable.")

    with tempfile.TemporaryDirectory(prefix="nym-voice-") as directory:
        audio_path = Path(directory) / "recording.wav"
        _record_audio(config, audio_path)
        return _transcribe_audio(config, audio_path)


def speak(text: str) -> None:
    """Speak text through the selected optional provider.

    This deliberately has no effect for blank text.  The caller may run it in a
    background process so response generation never waits for audio playback.
    """
    text = text.strip()
    if not text:
        return
    config = VoiceConfig.from_environment()
    status = voice_status()
    if not status.tts_ready:
        raise RuntimeError(status.tts_reason or "Text-to-speech is unavailable.")

    if config.tts_command:
        if "{input}" in config.tts_command:
            with tempfile.TemporaryDirectory(prefix="nym-voice-") as directory:
                input_path = Path(directory) / "speech.txt"
                input_path.write_text(text, encoding="utf-8")
                _run_template_command(config.tts_command, input_path=input_path, text=text)
        else:
            _run_template_command(config.tts_command, text=text)
        return

    local_tts = _local_tts_command()
    if local_tts:
        subprocess.run(local_tts, input=text, text=True, check=True, timeout=60)
        return

    with tempfile.TemporaryDirectory(prefix="nym-voice-") as directory:
        audio_path = Path(directory) / "speech.wav"
        client = _openai_client(config)
        response = client.audio.speech.create(
            model=config.tts_model,
            voice=config.tts_voice,
            input=text,
            response_format="wav",
        )
        response.write_to_file(audio_path)
        _play_audio(config, audio_path)


def bridge_voice_record() -> dict[str, object]:
    try:
        return {"ok": True, "answer": transcribe_microphone()}
    except Exception as exc:
        return {"ok": False, "error": _voice_error("Voice input failed", exc)}


def bridge_voice_speak(text: str) -> dict[str, object]:
    try:
        speak(text)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": _voice_error("Voice output failed", exc)}


def _record_audio(config: VoiceConfig, output: Path) -> None:
    command = _recorder_command(config)
    if command is None:
        raise RuntimeError("No supported microphone recorder was found.")
    _run_template_command(command, output=output, timeout=config.record_seconds + 5)
    if not output.is_file() or output.stat().st_size < 64:
        raise RuntimeError("The recorder did not produce usable audio.")


def _transcribe_audio(config: VoiceConfig, audio_path: Path) -> str:
    if config.mode == "realtime":
        return transcribe_realtime_audio(config, audio_path)

    if config.stt_command:
        completed = _run_template_command(config.stt_command, input_path=audio_path)
        return completed.stdout.strip()

    client = _openai_client(config)
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=config.stt_model,
            file=audio_file,
            response_format="text",
        )
    return str(response).strip()


def _play_audio(config: VoiceConfig, audio_path: Path) -> None:
    if config.playback_command:
        _run_template_command(config.playback_command, input_path=audio_path)
        return
    for command in _default_playback_commands(audio_path):
        if shutil.which(command[0]):
            subprocess.run(command, check=True, timeout=60)
            return
    raise RuntimeError("No supported audio playback command was found.")


def _recorder_command(config: VoiceConfig) -> tuple[str, ...] | None:
    if config.recorder_command:
        _require_placeholder(config.recorder_command, "{output}", "AGENT_VOICE_RECORDER")
        return config.recorder_command if _command_available(config.recorder_command) else None

    if shutil.which("ffmpeg"):
        return tuple(_ffmpeg_record_command(config.record_seconds))
    if shutil.which("arecord"):
        return (
            "arecord",
            "--quiet",
            "--format=S16_LE",
            "--rate=16000",
            "--channels=1",
            "--duration",
            str(config.record_seconds),
            "{output}",
        )
    return None


def _ffmpeg_record_command(seconds: int) -> list[str]:
    output = "{output}"
    if sys.platform == "darwin":
        return [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "avfoundation", "-i", ":0", "-ac", "1", "-ar", "16000",
            "-t", str(seconds), output,
        ]
    source_format = "pulse" if os.environ.get("PULSE_SERVER") or _is_wsl() else "alsa"
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", source_format, "-i", "default", "-ac", "1", "-ar", "16000",
        "-t", str(seconds), output,
    ]


def _stt_provider(config: VoiceConfig) -> str | None:
    if config.mode == "realtime":
        if not config.realtime_url:
            raise RuntimeError(
                "AGENT_REALTIME_VOICE_URL is required when AGENT_VOICE_MODE=realtime."
            )
        if importlib.util.find_spec("websockets") is None:
            raise RuntimeError(
                "Realtime voice requires the Python package 'websockets'."
            )
        if _is_huggingface_realtime_url(config.realtime_url):
            return "huggingface-realtime"
        return "realtime-websocket"
    if config.stt_command:
        _require_placeholder(config.stt_command, "{input}", "AGENT_VOICE_STT_COMMAND")
        return "command" if _command_available(config.stt_command) else None
    if config.api_key and config.stt_model:
        return "openai-compatible"
    return None


def _tts_provider(config: VoiceConfig) -> str | None:
    if config.tts_command:
        _require_any_placeholder(config.tts_command, ("{input}", "{text}"), "AGENT_VOICE_TTS_COMMAND")
        return "command" if _command_available(config.tts_command) else None
    if _local_tts_command():
        return "system"
    if config.api_key and config.tts_model and config.tts_voice and _has_playback_command(config):
        return "openai-compatible"
    return None


def _local_tts_command() -> list[str] | None:
    for name in ("espeak-ng", "espeak"):
        path = shutil.which(name)
        if path:
            return [path, "--stdin"]
    return None


def _has_playback_command(config: VoiceConfig) -> bool:
    if config.playback_command:
        _require_placeholder(config.playback_command, "{input}", "AGENT_VOICE_PLAYBACK")
        return _command_available(config.playback_command)
    return any(shutil.which(command) for command in ("ffplay", "aplay", "paplay"))


def _default_playback_commands(audio_path: Path) -> Iterable[list[str]]:
    path = str(audio_path)
    yield ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path]
    yield ["aplay", "--quiet", path]
    yield ["paplay", path]


def transcribe_realtime_audio(config: VoiceConfig, audio_path: Path) -> str:
    """Send one recorded turn through an OpenAI-Realtime-style speech server.

    The Hugging Face realtime voice Space documents a WebSocket route that
    accepts PCM16 16 kHz mono audio chunks and emits transcript deltas. Nym uses
    that protocol only as a voice transport: the returned transcript is still
    submitted to the normal Agent runtime, so desktop actions and approvals keep
    the same behavior as typed text.
    """
    if not config.realtime_url:
        raise RuntimeError("AGENT_REALTIME_VOICE_URL is not configured.")
    chunks = list(_pcm16_chunks(audio_path))
    if not chunks:
        raise RuntimeError("Recorded audio did not contain usable PCM data.")
    return asyncio.run(_transcribe_realtime_chunks(config, chunks))


async def _transcribe_realtime_chunks(
    config: VoiceConfig,
    chunks: list[bytes],
) -> str:
    connect_url = _realtime_connect_url(config.realtime_url or "")
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Realtime voice requires the Python package 'websockets'.") from exc

    transcript_parts: list[str] = []
    async with websockets.connect(connect_url, open_timeout=config.realtime_timeout_seconds) as ws:
        await _realtime_handshake(ws, config)
        for chunk in chunks:
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))
        while True:
            try:
                raw = await asyncio.wait_for(
                    ws.recv(),
                    timeout=config.realtime_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Realtime voice server did not finish transcription.") from exc
            event = _json_event(raw)
            event_type = str(event.get("type") or "")
            if event_type == "error":
                raise RuntimeError(_realtime_error_message(event))
            fragment = _transcript_fragment(event)
            if fragment:
                transcript_parts.append(fragment)
            if _realtime_done(event_type):
                break
    transcript = " ".join("".join(transcript_parts).split())
    if not transcript:
        raise RuntimeError("Realtime voice server returned no transcript.")
    return transcript


async def _realtime_handshake(ws: object, config: VoiceConfig) -> None:
    update = config.realtime_session_update or {
        "type": "session.update",
        "session": {
            "audio": {
                "input": {
                    "format": "pcm16",
                    "sample_rate": 16000,
                },
                "output": {
                    "format": "pcm16",
                    "sample_rate": 24000,
                },
            },
            "output_modalities": ["text"],
        },
    }
    await ws.send(json.dumps(update))


def _realtime_connect_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url:
        raise RuntimeError("AGENT_REALTIME_VOICE_URL is not configured.")
    if url.startswith(("ws://", "wss://")):
        return url

    errors: list[str] = []
    for session_url in _realtime_session_url_candidates(url):
        request = urllib.request.Request(
            session_url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            errors.append(f"{session_url}: {exc}")
            continue
        connect_url = payload.get("connect_url") or payload.get("websocket_url")
        if isinstance(connect_url, str) and connect_url.strip():
            return connect_url.strip()
        errors.append(f"{session_url}: response did not include connect_url")

    detail = "; ".join(errors) if errors else "no session endpoints were tried"
    raise RuntimeError(f"Could not create realtime voice session: {detail}")


def _realtime_session_url(raw_url: str) -> str:
    return _realtime_session_url_candidates(raw_url)[0]


def _realtime_session_url_candidates(raw_url: str) -> list[str]:
    url = raw_url.strip().rstrip("/")
    if not url:
        raise RuntimeError("AGENT_REALTIME_VOICE_URL is not configured.")
    url = _huggingface_space_app_url(url)
    if "://" not in url:
        url = f"https://{url}"
    if url.endswith("/session") or url.endswith("/api/session"):
        return [url]
    return [f"{url}/session", f"{url}/api/session"]


def _huggingface_space_app_url(raw_url: str) -> str:
    """Convert a Hugging Face Space page URL to the callable hf.space app host."""
    value = raw_url.strip().rstrip("/")
    if not value:
        return value
    parse_target = value if "://" in value else f"https://{value}"
    parsed = urllib.parse.urlparse(parse_target)
    host = parsed.netloc.casefold()
    parts = [part for part in parsed.path.split("/") if part]

    if host in {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}:
        if len(parts) >= 3 and parts[0] == "spaces":
            return f"https://{_hf_space_host(parts[1], parts[2])}"

    if not parsed.netloc and len(parts) == 2:
        return f"https://{_hf_space_host(parts[0], parts[1])}"

    return value


def _hf_space_host(owner: str, space: str) -> str:
    return f"{_hf_domain_label(owner)}-{_hf_domain_label(space)}.hf.space"


def _hf_domain_label(value: str) -> str:
    label = "".join(
        character.casefold() if character.isalnum() or character == "-" else "-"
        for character in value
    ).strip("-")
    return label or "space"


def _is_huggingface_realtime_url(raw_url: str | None) -> bool:
    if not raw_url:
        return False
    normalized = _huggingface_space_app_url(raw_url)
    parsed = urllib.parse.urlparse(normalized if "://" in normalized else f"https://{normalized}")
    return parsed.netloc.casefold().endswith(".hf.space") or (
        parsed.netloc.casefold() in {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}
        and "/spaces/" in parsed.path
    )


def _pcm16_chunks(audio_path: Path) -> Iterable[bytes]:
    if shutil.which("ffmpeg"):
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ]
        completed = subprocess.run(command, check=True, capture_output=True)
        yield from _chunk_bytes(completed.stdout, _REALTIME_CHUNK_BYTES)
        return
    yield from _wav_pcm16_chunks(audio_path)


def _wav_pcm16_chunks(audio_path: Path) -> Iterable[bytes]:
    import wave

    with wave.open(str(audio_path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != 16000 or audio.getsampwidth() != 2:
            raise RuntimeError(
                "Realtime voice requires ffmpeg or a 16 kHz mono PCM16 WAV recorder."
            )
        while data := audio.readframes(_REALTIME_CHUNK_BYTES // 2):
            yield data


def _chunk_bytes(data: bytes, size: int) -> Iterable[bytes]:
    for index in range(0, len(data), size):
        chunk = data[index:index + size]
        if chunk:
            yield chunk


def _json_event(raw: object) -> dict[str, object]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _transcript_fragment(event: dict[str, object]) -> str:
    event_type = str(event.get("type") or "")
    if event_type.endswith("input_audio_transcription.completed"):
        return str(event.get("transcript") or "")
    if event_type.endswith("input_audio_transcription.delta"):
        return str(event.get("delta") or "")
    return ""


def _realtime_done(event_type: str) -> bool:
    return event_type.endswith("input_audio_transcription.completed") or event_type in {
        "response.done",
        "response.completed",
    }


def _realtime_error_message(event: dict[str, object]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or "Realtime voice server returned an error.")


def _run_template_command(
    command: tuple[str, ...],
    *,
    output: Path | None = None,
    input_path: Path | None = None,
    text: str | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    replacements = {
        "{output}": str(output) if output is not None else "{output}",
        "{input}": str(input_path) if input_path is not None else "{input}",
        "{text}": text if text is not None else "{text}",
    }
    expanded = [replacements.get(part, part) for part in command]
    return subprocess.run(
        expanded,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _openai_client(config: VoiceConfig) -> OpenAI:
    if not config.api_key:
        raise RuntimeError("No voice API key is configured.")
    options: dict[str, str] = {"api_key": config.api_key}
    if config.base_url:
        options["base_url"] = config.base_url
    return OpenAI(**options)


def _environment_command(name: str) -> tuple[str, ...] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        parts = tuple(shlex.split(raw))
    except ValueError as exc:
        raise RuntimeError(f"{name} is not a valid command: {exc}") from exc
    if not parts:
        return None
    return parts


def _voice_provider_from_environment() -> str | None:
    raw = (
        os.environ.get("AGENT_VOICE_PROVIDER", "").strip()
        or os.environ.get("AGENT_VOICE_STT_PROVIDER", "").strip()
    )
    return raw.casefold() or None


def _voice_mode_from_environment() -> str:
    provider = _voice_provider_from_environment()
    if provider in _HF_REALTIME_VOICE_PROVIDERS:
        return "realtime"
    return _environment_choice(
        "AGENT_VOICE_MODE",
        default="turn",
        choices={"turn", "realtime"},
    )


def _realtime_voice_url_from_environment() -> str | None:
    url = (
        os.environ.get("AGENT_REALTIME_VOICE_URL", "").strip()
        or os.environ.get("AGENT_REALTIME_VOICE_SESSION_URL", "").strip()
    )
    if url:
        return url
    provider = _voice_provider_from_environment()
    if provider in _HF_REALTIME_VOICE_PROVIDERS:
        return _HF_REALTIME_VOICE_SPACE_URL
    return None


def _environment_choice(name: str, *, default: str, choices: set[str]) -> str:
    raw = os.environ.get(name, "").strip().casefold()
    if not raw:
        return default
    if raw not in choices:
        return default
    return raw


def _environment_json_object(name: str) -> dict[str, object] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _environment_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() not in {"0", "false", "no", "off"}


def _environment_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def _require_placeholder(command: tuple[str, ...], placeholder: str, name: str) -> None:
    if placeholder not in command:
        raise RuntimeError(f"{name} must include {placeholder} as a separate argument.")


def _require_any_placeholder(
    command: tuple[str, ...], placeholders: tuple[str, ...], name: str
) -> None:
    if not any(placeholder in command for placeholder in placeholders):
        options = " or ".join(placeholders)
        raise RuntimeError(f"{name} must include {options} as a separate argument.")


def _command_available(command: tuple[str, ...]) -> bool:
    executable = command[0]
    return Path(executable).is_file() if "/" in executable else shutil.which(executable) is not None


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(errors="ignore").casefold()
    except OSError:
        return False


def _voice_error(prefix: str, exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return f"{prefix}: {detail}"
