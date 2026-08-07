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
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Iterable

from openai import OpenAI


_DEFAULT_RECORD_SECONDS = 20
_MAX_RECORD_SECONDS = 120
_REALTIME_CHUNK_BYTES = 1280
_HF_REALTIME_VOICE_SPACE_URL = "https://huggingface.co/spaces/smolagents/hf-realtime-voice"
_HF_LOCAL_REALTIME_URL = "ws://127.0.0.1:8765/v1/realtime"
_HF_REALTIME_VOICE_PROVIDERS = {
    "hf",
    "hf-local",
    "hf-realtime",
    "hf-space",
    "huggingface",
    "huggingface-local",
    "huggingface-realtime",
    "huggingface-space",
}
_HF_SPACE_REALTIME_VOICE_PROVIDERS = {
    "hf-space-public",
    "huggingface-space-public",
    "huggingface-public",
}
_OPENAI_REALTIME_VOICE_PROVIDERS = {
    "openai-realtime",
    "openai-realtime-api",
    "realtime-openai",
}
_OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
_OPENAI_REALTIME_TRANSCRIPTION_MODEL = "gpt-live-transcribe"


@dataclass(frozen=True)
class VoiceStatus:
    input_ready: bool
    input_reason: str | None
    tts_ready: bool
    tts_reason: str | None
    auto_speak: bool
    stt_provider: str | None
    tts_provider: str | None
    input_secret_provider: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "input_ready": self.input_ready,
            "input_reason": self.input_reason,
            "tts_ready": self.tts_ready,
            "tts_reason": self.tts_reason,
            "auto_speak": self.auto_speak,
            "stt_provider": self.stt_provider,
            "tts_provider": self.tts_provider,
            "input_secret_provider": self.input_secret_provider,
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
    realtime_provider: str | None
    realtime_model: str
    realtime_session_update: dict[str, object] | None
    realtime_timeout_seconds: int
    realtime_api_key: str | None
    realtime_local_autostart: bool
    realtime_start_command: tuple[str, ...] | None
    realtime_start_timeout_seconds: int
    api_key: str | None
    base_url: str | None
    stt_model: str
    language: str | None
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
            realtime_provider=_voice_provider_from_environment(),
            realtime_model=_realtime_voice_model_from_environment(),
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
            realtime_api_key=_realtime_voice_api_key_from_environment(),
            realtime_local_autostart=_realtime_local_autostart_from_environment(),
            realtime_start_command=_environment_command("AGENT_REALTIME_VOICE_START_COMMAND"),
            realtime_start_timeout_seconds=_environment_int(
                "AGENT_REALTIME_VOICE_START_TIMEOUT_SECONDS",
                default=180,
                minimum=5,
                maximum=900,
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
            language=_voice_language_from_environment(),
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
    except RuntimeError as exc:
        return VoiceStatus(False, str(exc), False, str(exc), False, None, None)

    try:
        stt_provider = _stt_provider(config)
    except RuntimeError as exc:
        reason = str(exc)
        return VoiceStatus(
            False,
            reason,
            False,
            reason,
            False,
            None,
            None,
            input_secret_provider=_voice_input_secret_provider_for_error(config, reason)
            if recorder is not None
            else None,
        )

    try:
        tts_provider = _tts_provider(config)
    except RuntimeError as exc:
        return VoiceStatus(False, str(exc), False, str(exc), False, None, None)

    input_reason = None
    input_secret_provider = None
    if recorder is None:
        input_reason = "No supported microphone recorder was found."
    elif stt_provider is None:
        input_secret_provider = _voice_input_secret_provider_for_missing_stt(config)
        input_reason = (
            "Voice needs an API key. Press the mic again and paste your key."
            if input_secret_provider
            else "No speech-to-text provider is available."
        )

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
        input_secret_provider=input_secret_provider,
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


def bridge_voice_stream() -> int:
    """Emit live microphone transcript frames for the Rust TUI bridge."""
    def emit(payload: dict[str, object]) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    try:
        config = VoiceConfig.from_environment()
        status = voice_status()
        if not status.input_ready:
            raise RuntimeError(status.input_reason or "Voice input is unavailable.")

        stream_command = _stream_recorder_command(
            config,
            sample_rate=_realtime_input_sample_rate(config),
        )
        if config.mode == "realtime" and stream_command is not None:
            transcript = asyncio.run(
                _transcribe_realtime_microphone(
                    config,
                    stream_command,
                    lambda delta: emit({"kind": "delta", "delta": delta}),
                )
            )
        else:
            transcript = transcribe_microphone()
        emit({"kind": "final", "transcript": transcript})
        return 0
    except Exception as exc:
        emit({"kind": "error", "error": _voice_error("Voice input failed", exc)})
        return 1


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
    options: dict[str, object] = {
        "model": config.stt_model,
        "response_format": "text",
    }
    if config.language:
        options["language"] = config.language
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(file=audio_file, **options)
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


def _stream_recorder_command(
    config: VoiceConfig,
    *,
    sample_rate: int,
) -> tuple[str, ...] | None:
    """Return a recorder command that writes raw mono PCM16 to stdout."""
    if config.recorder_command:
        return None
    if shutil.which("ffmpeg"):
        if sys.platform == "darwin":
            source = ("-f", "avfoundation", "-i", ":0")
        else:
            source_format = "pulse" if os.environ.get("PULSE_SERVER") or _is_wsl() else "alsa"
            source = ("-f", source_format, "-i", "default")
        return (
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            *source,
            "-ac", "1", "-ar", str(sample_rate),
            "-t", str(config.record_seconds),
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        )
    if shutil.which("arecord"):
        return (
            "arecord", "--quiet", "--format=S16_LE", "--file-type=raw",
            f"--rate={sample_rate}", "--channels=1",
            f"--duration={config.record_seconds}",
        )
    return None


def _stt_provider(config: VoiceConfig) -> str | None:
    if config.mode == "realtime":
        if not config.realtime_url:
            raise RuntimeError(
                "AGENT_REALTIME_VOICE_URL is required when AGENT_VOICE_MODE=realtime."
            )
        if _is_openai_realtime_provider(config) and not _openai_realtime_api_key(config):
            raise RuntimeError(
                "OpenAI Realtime voice requires AGENT_REALTIME_VOICE_API_KEY, "
                "AGENT_VOICE_API_KEY, or OPENAI_API_KEY."
            )
        if importlib.util.find_spec("websockets") is None:
            raise RuntimeError(
                "Realtime voice requires the Python package 'websockets'."
            )
        if _is_openai_realtime_provider(config):
            return "openai-realtime"
        if config.realtime_local_autostart and _is_local_realtime_url(config.realtime_url):
            if _local_realtime_start_command(config) is None:
                raise RuntimeError(
                    "Local Hugging Face voice requires the 'speech-to-speech' command. "
                    "Install local voice dependencies, then retry."
                )
            return "huggingface-local"
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


def _voice_input_secret_provider_for_missing_stt(config: VoiceConfig) -> str | None:
    if config.stt_command or config.api_key:
        return None
    if config.mode != "turn":
        return None
    return "voice"


def _voice_input_secret_provider_for_error(
    config: VoiceConfig,
    reason: str,
) -> str | None:
    if _is_openai_realtime_provider(config) and not _openai_realtime_api_key(config):
        return "voice"
    lowered = reason.casefold()
    if "api_key" in lowered and "voice" in lowered:
        return "voice"
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
    """Send one recorded turn through a local OpenAI-Realtime-style speech server.

    The Hugging Face speech-to-speech backend exposes the same WebSocket wire
    shape as the smolagents HF realtime voice Space. Nym uses only the local
    speech transport/transcription part of that protocol: the transcript is
    submitted to the normal Agent runtime, so desktop actions and approvals keep
    the same behavior as typed text.
    """
    if not config.realtime_url:
        raise RuntimeError("AGENT_REALTIME_VOICE_URL is not configured.")
    if config.realtime_local_autostart and _is_local_realtime_url(config.realtime_url):
        _ensure_local_realtime_server(config)
    chunks = list(_pcm16_chunks(audio_path, sample_rate=_realtime_input_sample_rate(config)))
    if not chunks:
        raise RuntimeError("Recorded audio did not contain usable PCM data.")
    return asyncio.run(_transcribe_realtime_chunks(config, chunks))


async def _transcribe_realtime_chunks(
    config: VoiceConfig,
    chunks: list[bytes],
) -> str:
    connect_url = _realtime_connect_url(
        config.realtime_url or "",
        api_key=config.realtime_api_key,
    )
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Realtime voice requires the Python package 'websockets'.") from exc

    transcript_parts: list[str] = []
    final_transcript: str | None = None
    headers = _realtime_headers(config)
    try:
        connection = websockets.connect(
            connect_url,
            open_timeout=config.realtime_timeout_seconds,
            additional_headers=headers,
        )
    except TypeError:
        connection = websockets.connect(
            connect_url,
            open_timeout=config.realtime_timeout_seconds,
            extra_headers=headers,
        )
    async with connection as ws:
        await _realtime_handshake(ws, config)
        for chunk in chunks:
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
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
            if event_type.endswith("input_audio_transcription.delta"):
                fragment = str(event.get("delta") or "")
                if fragment:
                    transcript_parts.append(fragment)
            elif event_type.endswith("input_audio_transcription.completed"):
                final_transcript = str(event.get("transcript") or "")
            if _realtime_done(event_type):
                break
    transcript = " ".join((final_transcript or "".join(transcript_parts)).split())
    if not transcript:
        raise RuntimeError("Realtime voice server returned no transcript.")
    return transcript


async def _transcribe_realtime_microphone(
    config: VoiceConfig,
    command: tuple[str, ...],
    on_delta: Callable[[str], None],
) -> str:
    """Stream raw microphone PCM to Realtime and forward transcript deltas."""
    if not config.realtime_url:
        raise RuntimeError("AGENT_REALTIME_VOICE_URL is not configured.")
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Realtime voice requires the Python package 'websockets'.") from exc

    connect_url = _realtime_connect_url(
        config.realtime_url,
        api_key=config.realtime_api_key,
    )
    headers = _realtime_headers(config)
    try:
        connection = websockets.connect(
            connect_url,
            open_timeout=config.realtime_timeout_seconds,
            additional_headers=headers,
        )
    except TypeError:
        connection = websockets.connect(
            connect_url,
            open_timeout=config.realtime_timeout_seconds,
            extra_headers=headers,
        )

    process: subprocess.Popen[bytes] | None = None
    sender: asyncio.Task[None] | None = None
    transcript_parts: list[str] = []
    final_transcript: str | None = None
    try:
        async with connection as ws:
            await _realtime_handshake(ws, config)
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if process.stdout is None:
                raise RuntimeError("Microphone recorder stdout was not captured.")

            async def send_audio() -> None:
                assert process is not None and process.stdout is not None
                sent_audio = False
                while True:
                    chunk = await asyncio.to_thread(
                        process.stdout.read,
                        _REALTIME_CHUNK_BYTES,
                    )
                    if not chunk:
                        break
                    sent_audio = True
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }))
                if not sent_audio:
                    detail = _recorder_error(process)
                    raise RuntimeError(detail or "The recorder did not produce usable audio.")
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            sender = asyncio.create_task(send_audio())
            deadline = time.monotonic() + config.record_seconds + config.realtime_timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Realtime voice server did not finish transcription.")
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(1.0, remaining))
                except asyncio.TimeoutError:
                    if sender.done():
                        sender.result()
                    continue
                event = _json_event(raw)
                event_type = str(event.get("type") or "")
                if event_type == "error":
                    raise RuntimeError(_realtime_error_message(event))
                if event_type.endswith("input_audio_transcription.failed"):
                    raise RuntimeError(_realtime_error_message(event))
                if event_type.endswith("input_audio_transcription.delta"):
                    delta = str(event.get("delta") or "")
                    if delta:
                        transcript_parts.append(delta)
                        on_delta(delta)
                elif event_type.endswith("input_audio_transcription.completed"):
                    final_transcript = str(event.get("transcript") or "")
                    break
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if sender is not None and not sender.done():
            sender.cancel()
        if sender is not None:
            await asyncio.gather(sender, return_exceptions=True)

    transcript = " ".join((final_transcript or "".join(transcript_parts)).split())
    if not transcript:
        raise RuntimeError("Realtime voice server returned no transcript.")
    return transcript


def _recorder_error(process: subprocess.Popen[bytes]) -> str:
    if process.stderr is None:
        return ""
    try:
        return process.stderr.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


async def _realtime_handshake(ws: object, config: VoiceConfig) -> None:
    if config.realtime_session_update:
        update = config.realtime_session_update
    elif _is_openai_realtime_provider(config):
        transcription_model = _openai_realtime_transcription_model()
        transcription: dict[str, object] = {
            "model": transcription_model,
        }
        if config.language:
            if transcription_model == "gpt-live-transcribe":
                transcription["languages"] = [config.language]
                transcription["delay"] = "low"
            else:
                transcription["language"] = config.language
            if config.language == "en":
                transcription["prompt"] = (
                    "The speaker is speaking English. Transcribe in English only. "
                    "Do not translate or emit text in another language."
                )
        update = {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "noise_reduction": {"type": "near_field"},
                        "transcription": transcription,
                        "turn_detection": None
                        if transcription_model == "gpt-live-transcribe"
                        else {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 700,
                        },
                    },
                },
            },
        }
    else:
        update = {
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


def _realtime_headers(config: VoiceConfig) -> dict[str, str] | None:
    if not _is_openai_realtime_provider(config):
        return None
    api_key = _openai_realtime_api_key(config)
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


def _openai_realtime_api_key(config: VoiceConfig) -> str | None:
    if config.realtime_api_key:
        return config.realtime_api_key
    if config.api_key:
        return config.api_key
    return (
        os.environ.get("AGENT_VOICE_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or None
    )


def _openai_realtime_transcription_model() -> str:
    return (
        os.environ.get("AGENT_REALTIME_VOICE_TRANSCRIPTION_MODEL", "").strip()
        or os.environ.get("AGENT_VOICE_REALTIME_TRANSCRIPTION_MODEL", "").strip()
        or _OPENAI_REALTIME_TRANSCRIPTION_MODEL
    )


def _realtime_input_sample_rate(config: VoiceConfig) -> int:
    return 24000 if _is_openai_realtime_provider(config) else 16000


def _realtime_connect_url(raw_url: str, *, api_key: str | None = None) -> str:
    url = raw_url.strip()
    if not url:
        raise RuntimeError("AGENT_REALTIME_VOICE_URL is not configured.")
    if url.startswith(("ws://", "wss://")):
        return url

    errors: list[str] = []
    for session_url in _realtime_session_url_candidates(url):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            session_url,
            data=b"{}",
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = f"{session_url}: HTTP Error {exc.code}: {exc.reason}"
            if exc.code in {401, 403} and _is_huggingface_realtime_url(url):
                if api_key:
                    message += " (Hugging Face rejected the realtime voice bearer token)"
                else:
                    message += " (set AGENT_REALTIME_VOICE_API_KEY or HF_TOKEN for this Hugging Face Space)"
                errors.append(message)
                break
            errors.append(message)
            continue
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


def _ensure_local_realtime_server(config: VoiceConfig) -> None:
    url = config.realtime_url or _HF_LOCAL_REALTIME_URL
    if _local_realtime_server_ready(url):
        return
    command = _local_realtime_start_command(config)
    if command is None:
        raise RuntimeError(
            "Local Hugging Face voice requires the 'speech-to-speech' command. "
            "Install local voice dependencies, then retry."
        )
    log_path = Path(tempfile.gettempdir()) / "nym-hf-realtime-voice.log"
    try:
        log_handle = log_path.open("ab")
    except OSError:
        log_handle = subprocess.DEVNULL  # type: ignore[assignment]
    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start local Hugging Face voice server: {exc}") from exc
    finally:
        close = getattr(log_handle, "close", None)
        if callable(close):
            close()

    deadline = time.monotonic() + config.realtime_start_timeout_seconds
    while time.monotonic() < deadline:
        if _local_realtime_server_ready(url):
            return
        time.sleep(0.5)
    raise RuntimeError(
        "Local Hugging Face voice server did not become ready. "
        f"Check {log_path} for startup details."
    )


def _local_realtime_start_command(config: VoiceConfig) -> tuple[str, ...] | None:
    if config.realtime_start_command:
        return config.realtime_start_command
    executable = shutil.which("speech-to-speech")
    if executable is None:
        return None
    return (
        executable,
        "--mode",
        "realtime",
        "--stt",
        os.environ.get("AGENT_HF_VOICE_STT", "parakeet-tdt").strip() or "parakeet-tdt",
        "--llm_backend",
        os.environ.get("AGENT_HF_VOICE_LLM_BACKEND", "transformers").strip() or "transformers",
        "--tts",
        os.environ.get("AGENT_HF_VOICE_TTS", "qwen3").strip() or "qwen3",
        "--model_name",
        os.environ.get("AGENT_HF_VOICE_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507").strip()
        or "Qwen/Qwen3-4B-Instruct-2507",
        "--enable_live_transcription",
    )


def _local_realtime_server_ready(raw_url: str) -> bool:
    host_port = _local_realtime_host_port(raw_url)
    if host_port is None:
        return False
    host, port = host_port
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _local_realtime_host_port(raw_url: str) -> tuple[str, int] | None:
    parsed = urllib.parse.urlparse(raw_url if "://" in raw_url else f"ws://{raw_url}")
    host = parsed.hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return None
    scheme = parsed.scheme.casefold()
    default_port = 443 if scheme == "wss" else 80
    return host, parsed.port or default_port


def _is_local_realtime_url(raw_url: str | None) -> bool:
    return bool(raw_url and _local_realtime_host_port(raw_url) is not None)


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


def _pcm16_chunks(audio_path: Path, *, sample_rate: int = 16000) -> Iterable[bytes]:
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
            str(sample_rate),
            "pipe:1",
        ]
        completed = subprocess.run(command, check=True, capture_output=True)
        yield from _chunk_bytes(completed.stdout, _REALTIME_CHUNK_BYTES)
        return
    yield from _wav_pcm16_chunks(audio_path, sample_rate=sample_rate)


def _wav_pcm16_chunks(audio_path: Path, *, sample_rate: int = 16000) -> Iterable[bytes]:
    import wave

    with wave.open(str(audio_path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != sample_rate or audio.getsampwidth() != 2:
            raise RuntimeError(
                f"Realtime voice requires ffmpeg or a {sample_rate // 1000} kHz mono PCM16 WAV recorder."
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


def _realtime_voice_api_key_from_environment() -> str | None:
    explicit = os.environ.get("AGENT_REALTIME_VOICE_API_KEY", "").strip()
    if explicit:
        return explicit
    provider = _voice_provider_from_environment()
    if provider not in _HF_REALTIME_VOICE_PROVIDERS | _HF_SPACE_REALTIME_VOICE_PROVIDERS:
        return None
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
        or None
    )


def _voice_provider_from_environment() -> str | None:
    raw = (
        os.environ.get("AGENT_VOICE_PROVIDER", "").strip()
        or os.environ.get("AGENT_VOICE_STT_PROVIDER", "").strip()
    )
    if raw:
        return raw.casefold()
    if os.environ.get("AGENT_VOICE_MODE", "").strip():
        return None
    if any(
        os.environ.get(name, "").strip()
        for name in (
            "AGENT_VOICE_RECORDER",
            "AGENT_VOICE_STT_COMMAND",
            "AGENT_VOICE_BASE_URL",
        )
    ):
        return None
    return "openai-realtime"


def _voice_language_from_environment() -> str | None:
    language = os.environ.get("AGENT_VOICE_LANGUAGE", "en").strip().casefold()
    if language in {"", "auto", "detect"}:
        return None
    return language


def _voice_mode_from_environment() -> str:
    provider = _voice_provider_from_environment()
    if (
        provider in _HF_REALTIME_VOICE_PROVIDERS
        or provider in _HF_SPACE_REALTIME_VOICE_PROVIDERS
        or provider in _OPENAI_REALTIME_VOICE_PROVIDERS
    ):
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
        or os.environ.get("AGENT_HF_VOICE_LOCAL_URL", "").strip()
    )
    if url:
        return url
    provider = _voice_provider_from_environment()
    if provider in _OPENAI_REALTIME_VOICE_PROVIDERS:
        return _openai_realtime_url()
    if provider in _HF_REALTIME_VOICE_PROVIDERS:
        return _HF_LOCAL_REALTIME_URL
    if provider in _HF_SPACE_REALTIME_VOICE_PROVIDERS:
        return _HF_REALTIME_VOICE_SPACE_URL
    return None


def _realtime_local_autostart_from_environment() -> bool:
    provider = _voice_provider_from_environment()
    default = provider in _HF_REALTIME_VOICE_PROVIDERS
    return _environment_flag("AGENT_REALTIME_VOICE_AUTOSTART", default=default)


def _realtime_voice_model_from_environment() -> str:
    return (
        os.environ.get("AGENT_REALTIME_VOICE_MODEL", "").strip()
        or os.environ.get("AGENT_VOICE_REALTIME_MODEL", "").strip()
        or _OPENAI_REALTIME_TRANSCRIPTION_MODEL
    )


def _openai_realtime_url() -> str:
    query = urllib.parse.urlencode({"intent": "transcription"})
    return f"{_OPENAI_REALTIME_URL}?{query}"


def _is_openai_realtime_provider(config: VoiceConfig) -> bool:
    if config.realtime_provider in _OPENAI_REALTIME_VOICE_PROVIDERS:
        return True
    url = (config.realtime_url or "").strip()
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc.casefold() == "api.openai.com" and parsed.path.rstrip("/") == "/v1/realtime"


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
