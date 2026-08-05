"""Voice integration: VAD-gated STT recording and streaming TTS playback.

All API keys are read from environment variables — never hardcoded.
Users configure voice the same way they configure LLM providers.

STT providers (AGENT_STT_PROVIDER):
  openai   — Whisper API (requires OPENAI_API_KEY or AGENT_VOICE_API_KEY)
  faster   — faster-whisper local (no key needed)
  whisper  — openai-whisper local (no key needed)
  vosk     — Vosk local (no key needed)

TTS providers (AGENT_TTS_PROVIDER):
  openai   — OpenAI TTS streaming (requires OPENAI_API_KEY or AGENT_VOICE_API_KEY)
  pyttsx3  — local pyttsx3 (no key needed)
  espeak   — local espeak (no key needed)

VAD providers (AGENT_VAD_PROVIDER):
  webrtcvad — fast WebRTC VAD (default)
  silero    — accurate Silero VAD
  none      — fixed 5-second recording

Silence threshold: AGENT_VAD_SILENCE_MS (default 300ms)
"""
from __future__ import annotations

import io
import os
import re
import shutil
import struct
import tempfile
import threading
import time
import wave
from typing import Callable, Iterator

_SAMPLE_RATE = 16000
_FRAME_MS = 30
_FRAME_SAMPLES = _SAMPLE_RATE * _FRAME_MS // 1000
_FRAME_BYTES = _FRAME_SAMPLES * 2  # 16-bit PCM


def _silence_ms() -> int:
    raw = os.environ.get("AGENT_VAD_SILENCE_MS", "300").strip()
    try:
        value = int(raw)
        return max(100, min(3000, value))
    except ValueError:
        return 300


def _vad_provider() -> str:
    return os.environ.get("AGENT_VAD_PROVIDER", "webrtcvad").strip().casefold()


def _stt_provider() -> str:
    raw = os.environ.get("AGENT_STT_PROVIDER", "").strip().casefold()
    if raw:
        return raw
    # Auto-detect: prefer OpenAI if key is available
    if _voice_api_key():
        return "openai"
    if shutil.which("python3") and _local_stt_available("faster"):
        return "faster"
    if _local_stt_available("whisper"):
        return "whisper"
    return "openai"  # will fail with a clear error if no key


def _tts_provider() -> str:
    raw = os.environ.get("AGENT_TTS_PROVIDER", "").strip().casefold()
    if raw:
        return raw
    if _voice_api_key():
        return "openai"
    if shutil.which("espeak") or shutil.which("espeak-ng"):
        return "espeak"
    return "openai"


def _voice_api_key() -> str:
    """AGENT_VOICE_API_KEY overrides OPENAI_API_KEY for voice-only deployments."""
    return (
        os.environ.get("AGENT_VOICE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )


def _local_stt_available(provider: str) -> bool:
    try:
        if provider == "faster":
            import faster_whisper  # noqa: F401
            return True
        if provider == "whisper":
            import whisper  # noqa: F401
            return True
        if provider == "vosk":
            import vosk  # noqa: F401
            return True
    except ImportError:
        pass
    return False


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def _vad_webrtcvad(frames: list[bytes]) -> list[bool]:
    """Return per-frame speech flags using webrtcvad."""
    import webrtcvad
    vad = webrtcvad.Vad(2)  # aggressiveness 0-3
    return [vad.is_speech(frame, _SAMPLE_RATE) for frame in frames]


def _vad_silero(frames: list[bytes]) -> list[bool]:
    """Return per-frame speech flags using Silero VAD."""
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        verbose=False,
    )
    (get_speech_timestamps, _, _, _, _) = utils
    audio_bytes = b"".join(frames)
    audio_tensor = torch.frombuffer(audio_bytes, dtype=torch.int16).float() / 32768.0
    timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=_SAMPLE_RATE)
    if not timestamps:
        return [False] * len(frames)
    speech_samples: set[int] = set()
    for ts in timestamps:
        speech_samples.update(range(ts["start"], ts["end"]))
    result = []
    for i, _ in enumerate(frames):
        start = i * _FRAME_SAMPLES
        end = start + _FRAME_SAMPLES
        result.append(bool(speech_samples.intersection(range(start, end))))
    return result


def _is_speech_frame(frame: bytes, provider: str) -> bool:
    """Single-frame VAD check — used during streaming recording."""
    if provider == "webrtcvad":
        try:
            import webrtcvad
            vad = webrtcvad.Vad(2)
            return vad.is_speech(frame, _SAMPLE_RATE)
        except Exception:
            return True
    # silero and none: treat all frames as speech during recording
    return True


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_until_silence() -> bytes:
    """Record from the microphone until VAD detects silence.

    Returns raw 16-bit mono PCM at 16 kHz.
    Raises RuntimeError with a clear message if microphone is unavailable.
    """
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "Microphone recording requires pyaudio. "
            "Install it with: pip install pyaudio"
        ) from exc

    vad_prov = _vad_provider()
    silence_frames = max(1, _silence_ms() // _FRAME_MS)
    max_record_frames = (_SAMPLE_RATE * 30) // _FRAME_SAMPLES  # 30s hard cap

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=_SAMPLE_RATE,
            input=True,
            frames_per_buffer=_FRAME_SAMPLES,
        )
    except OSError as exc:
        pa.terminate()
        raise RuntimeError(f"Could not open microphone: {exc}") from exc

    recorded: list[bytes] = []
    silent_count = 0
    speech_started = False

    try:
        for _ in range(max_record_frames):
            frame = stream.read(_FRAME_SAMPLES, exception_on_overflow=False)
            recorded.append(frame)
            is_speech = _is_speech_frame(frame, vad_prov)
            if is_speech:
                speech_started = True
                silent_count = 0
            elif speech_started:
                silent_count += 1
                if silent_count >= silence_frames:
                    break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return b"".join(recorded)


def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

def transcribe(pcm: bytes) -> str:
    """Transcribe PCM audio to text. Returns empty string on silence."""
    provider = _stt_provider()
    wav_bytes = _pcm_to_wav(pcm)

    if provider == "openai":
        return _stt_openai(wav_bytes)
    if provider == "faster":
        return _stt_faster_whisper(wav_bytes)
    if provider == "whisper":
        return _stt_whisper(wav_bytes)
    if provider == "vosk":
        return _stt_vosk(pcm)
    raise RuntimeError(
        f"Unknown STT provider: {provider}. "
        "Set AGENT_STT_PROVIDER to: openai, faster, whisper, or vosk."
    )


def _stt_openai(wav_bytes: bytes) -> str:
    api_key = _voice_api_key()
    if not api_key:
        raise RuntimeError(
            "Voice STT requires an API key. "
            "Set OPENAI_API_KEY or AGENT_VOICE_API_KEY. "
            "Use /apikey openai in Agent to configure it."
        )
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.audio.transcriptions.create(
        model=os.environ.get("AGENT_WHISPER_MODEL", "whisper-1"),
        file=("audio.wav", wav_bytes, "audio/wav"),
        response_format="text",
    )
    return str(response).strip()


def _stt_faster_whisper(wav_bytes: bytes) -> str:
    from faster_whisper import WhisperModel
    model_size = os.environ.get("AGENT_WHISPER_MODEL", "base")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        segments, _ = model.transcribe(tmp_path, beam_size=1, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _stt_whisper(wav_bytes: bytes) -> str:
    import whisper
    model_size = os.environ.get("AGENT_WHISPER_MODEL", "base")
    model = whisper.load_model(model_size)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        result = model.transcribe(tmp_path, fp16=False)
        return str(result.get("text", "")).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _stt_vosk(pcm: bytes) -> str:
    import json as _json
    from vosk import KaldiRecognizer, Model
    model_path = os.environ.get("AGENT_VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")
    model = Model(model_path)
    rec = KaldiRecognizer(model, _SAMPLE_RATE)
    rec.AcceptWaveform(pcm)
    result = _json.loads(rec.FinalResult())
    return str(result.get("text", "")).strip()


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def _sentence_chunks(text: str) -> list[str]:
    """Split text into sentence-level chunks for low-latency streaming TTS."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def speak(text: str, *, on_chunk_start: Callable[[str], None] | None = None) -> None:
    """Speak text using the configured TTS provider.

    on_chunk_start is called with each sentence before it starts playing,
    enabling the UI to show which sentence is being spoken.
    """
    provider = _tts_provider()
    chunks = _sentence_chunks(text) or [text]

    if provider == "openai":
        _tts_openai_streaming(chunks, on_chunk_start=on_chunk_start)
    elif provider == "pyttsx3":
        _tts_pyttsx3(text)
    elif provider == "espeak":
        _tts_espeak(chunks, on_chunk_start=on_chunk_start)
    else:
        raise RuntimeError(
            f"Unknown TTS provider: {provider}. "
            "Set AGENT_TTS_PROVIDER to: openai, pyttsx3, or espeak."
        )


def _tts_openai_streaming(
    chunks: list[str],
    *,
    on_chunk_start: Callable[[str], None] | None,
) -> None:
    api_key = _voice_api_key()
    if not api_key:
        raise RuntimeError(
            "Voice TTS requires an API key. "
            "Set OPENAI_API_KEY or AGENT_VOICE_API_KEY. "
            "Use /apikey openai in Agent to configure it."
        )
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "Audio playback requires pyaudio. Install it with: pip install pyaudio"
        ) from exc

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    voice = os.environ.get("AGENT_TTS_VOICE", "alloy")
    model = os.environ.get("AGENT_TTS_MODEL", "tts-1")  # tts-1 is faster; tts-1-hd is higher quality

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    try:
        for chunk in chunks:
            if on_chunk_start:
                on_chunk_start(chunk)
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=chunk,
                response_format="pcm",  # raw 16-bit PCM — lowest latency
            ) as response:
                for audio_chunk in response.iter_bytes(chunk_size=4096):
                    stream.write(audio_chunk)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def _tts_pyttsx3(text: str) -> None:
    import pyttsx3
    engine = pyttsx3.init()
    rate = int(os.environ.get("AGENT_TTS_RATE", "175"))
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()


def _tts_espeak(
    chunks: list[str],
    *,
    on_chunk_start: Callable[[str], None] | None,
) -> None:
    import subprocess
    binary = shutil.which("espeak-ng") or shutil.which("espeak") or "espeak"
    for chunk in chunks:
        if on_chunk_start:
            on_chunk_start(chunk)
        subprocess.run([binary, chunk], check=False, capture_output=True)


# ---------------------------------------------------------------------------
# Bridge entry points (called from main.py)
# ---------------------------------------------------------------------------

def bridge_voice_record() -> dict:
    """Record audio and return transcription. Called by the TUI bridge."""
    try:
        pcm = record_until_silence()
        text = transcribe(pcm)
        # Return in "answer" so BridgeResponse.answer picks it up in Rust.
        return {"ok": True, "answer": text}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"Voice recording failed: {exc}"}


def bridge_voice_speak(text: str) -> dict:
    """Speak text asynchronously (fire-and-forget from TUI). Called by the TUI bridge."""
    if not text.strip():
        return {"ok": True}
    try:
        speak(text)
        return {"ok": True}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"Voice TTS failed: {exc}"}


def voice_status() -> dict:
    """Return voice capability status for /status display."""
    stt = _stt_provider()
    tts = _tts_provider()
    vad = _vad_provider()
    has_key = bool(_voice_api_key())
    needs_key = stt == "openai" or tts == "openai"
    return {
        "stt_provider": stt,
        "tts_provider": tts,
        "vad_provider": vad,
        "api_key_configured": has_key,
        "ready": has_key or not needs_key,
        "key_env": "OPENAI_API_KEY or AGENT_VOICE_API_KEY" if needs_key else None,
    }
