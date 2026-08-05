"""Optional voice input and output for the terminal UI.

Voice is deliberately independent from model setup.  It uses an already
configured OpenAI-compatible key when one is available, or local commands when
the host provides them.  Nothing in this module prompts for credentials,
installs software, or contacts a provider while determining availability.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

from openai import OpenAI


_DEFAULT_RECORD_SECONDS = 20
_MAX_RECORD_SECONDS = 120


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
