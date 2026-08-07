import os
import unittest
import urllib.error
from unittest.mock import patch

from agent.voice import (
    VoiceConfig,
    _chunk_bytes,
    _local_realtime_start_command,
    _realtime_connect_url,
    _realtime_headers,
    _realtime_session_url,
    _realtime_session_url_candidates,
    _transcript_fragment,
    bridge_voice_record,
    bridge_voice_speak,
    voice_status,
)


class VoiceTests(unittest.TestCase):
    def test_voice_uses_existing_openai_key_without_network_probe(self) -> None:
        def no_local_voice(name: str) -> str | None:
            if name == "speech-to-speech":
                return None
            return "/usr/bin/" + name

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "existing-key"}, clear=True),
            patch("agent.voice.shutil.which", side_effect=no_local_voice),
        ):
            status = voice_status()

        self.assertTrue(status.input_ready)
        self.assertTrue(status.tts_ready)
        self.assertEqual(status.stt_provider, "openai-compatible")
        self.assertEqual(status.tts_provider, "system")
        self.assertFalse(status.auto_speak)

    def test_voice_can_use_local_commands_without_a_key(self) -> None:
        environment = {
            "AGENT_VOICE_RECORDER": "capture {output}",
            "AGENT_VOICE_STT_COMMAND": "transcribe {input}",
            "AGENT_VOICE_TTS_COMMAND": "speak {text}",
            "AGENT_TTS_ENABLED": "1",
        }
        with (
            patch.dict("os.environ", environment, clear=True),
            patch("agent.voice.shutil.which", side_effect=lambda name: "/usr/bin/" + name),
        ):
            status = voice_status()

        self.assertTrue(status.input_ready)
        self.assertTrue(status.tts_ready)
        self.assertTrue(status.auto_speak)
        self.assertEqual(status.stt_provider, "command")
        self.assertEqual(status.tts_provider, "command")

    def test_invalid_command_template_disables_voice_instead_of_raising(self) -> None:
        with patch.dict(
            "os.environ",
            {"AGENT_VOICE_STT_COMMAND": "transcribe"},
            clear=True,
        ):
            status = voice_status()

        self.assertFalse(status.input_ready)
        self.assertIn("AGENT_VOICE_STT_COMMAND", status.input_reason or "")

    def test_voice_configuration_bounds_record_duration(self) -> None:
        with patch.dict(
            "os.environ", {"AGENT_VOICE_RECORD_SECONDS": "900"}, clear=True
        ):
            config = VoiceConfig.from_environment()

        self.assertEqual(config.record_seconds, 120)

    def test_bridge_returns_structured_voice_failures(self) -> None:
        with patch("agent.voice.transcribe_microphone", side_effect=RuntimeError("no microphone")):
            result = bridge_voice_record()

        self.assertEqual(result, {"ok": False, "error": "Voice input failed: no microphone"})

    def test_speech_bridge_does_not_require_text(self) -> None:
        with patch("agent.voice.speak") as speak:
            result = bridge_voice_speak("")

        self.assertEqual(result, {"ok": True})
        speak.assert_called_once_with("")

    def test_realtime_voice_status_uses_websocket_provider_without_key_prompt(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "AGENT_VOICE_MODE": "realtime",
                    "AGENT_REALTIME_VOICE_URL": "wss://voice.example/session",
                },
                clear=True,
            ),
            patch("agent.voice.shutil.which", side_effect=lambda name: "/usr/bin/" + name),
            patch("agent.voice.importlib.util.find_spec", return_value=object()),
        ):
            status = voice_status()

        self.assertTrue(status.input_ready)
        self.assertEqual(status.stt_provider, "realtime-websocket")

    def test_openai_realtime_voice_uses_user_api_key(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "AGENT_VOICE_PROVIDER": "openai-realtime",
                    "OPENAI_API_KEY": "user-key",
                },
                clear=True,
            ),
            patch("agent.voice.shutil.which", side_effect=lambda name: "/usr/bin/" + name),
            patch("agent.voice.importlib.util.find_spec", return_value=object()),
        ):
            config = VoiceConfig.from_environment()
            status = voice_status()

        self.assertEqual(config.mode, "realtime")
        self.assertEqual(config.realtime_provider, "openai-realtime")
        self.assertEqual(config.realtime_url, "wss://api.openai.com/v1/realtime?model=gpt-realtime")
        self.assertTrue(status.input_ready)
        self.assertEqual(status.stt_provider, "openai-realtime")
        self.assertEqual(_realtime_headers(config), {"Authorization": "Bearer user-key"})

    def test_openai_realtime_voice_requires_user_api_key(self) -> None:
        with (
            patch.dict("os.environ", {"AGENT_VOICE_PROVIDER": "openai-realtime"}, clear=True),
            patch("agent.voice.shutil.which", side_effect=lambda name: "/usr/bin/" + name),
            patch("agent.voice.importlib.util.find_spec", return_value=object()),
        ):
            status = voice_status()

        self.assertFalse(status.input_ready)
        self.assertIn("OpenAI Realtime voice requires", status.input_reason or "")
        self.assertEqual(status.input_secret_provider, "voice")

    def test_missing_turn_voice_key_requests_masked_voice_key_prompt(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("agent.voice.shutil.which", side_effect=lambda name: "/usr/bin/" + name),
        ):
            status = voice_status()

        self.assertFalse(status.input_ready)
        self.assertEqual(status.input_secret_provider, "voice")
        self.assertIn("Voice needs an API key", status.input_reason or "")

    def test_huggingface_voice_provider_uses_local_backend_only_when_explicit(self) -> None:
        with (
            patch.dict("os.environ", {"AGENT_VOICE_PROVIDER": "huggingface"}, clear=True),
            patch("agent.voice.shutil.which", side_effect=lambda name: "/usr/bin/" + name),
            patch("agent.voice.importlib.util.find_spec", return_value=object()),
        ):
            config = VoiceConfig.from_environment()
            status = voice_status()

        self.assertEqual(config.mode, "realtime")
        self.assertEqual(
            config.realtime_url,
            "ws://127.0.0.1:8765/v1/realtime",
        )
        self.assertTrue(config.realtime_local_autostart)
        self.assertTrue(status.input_ready)
        self.assertEqual(status.stt_provider, "huggingface-local")

    def test_huggingface_public_space_provider_remains_explicit(self) -> None:
        with (
            patch.dict("os.environ", {"AGENT_VOICE_PROVIDER": "huggingface-public"}, clear=True),
            patch("agent.voice.shutil.which", side_effect=lambda name: "/usr/bin/" + name),
            patch("agent.voice.importlib.util.find_spec", return_value=object()),
        ):
            config = VoiceConfig.from_environment()
            status = voice_status()

        self.assertEqual(config.mode, "realtime")
        self.assertEqual(
            config.realtime_url,
            "https://huggingface.co/spaces/smolagents/hf-realtime-voice",
        )
        self.assertFalse(config.realtime_local_autostart)
        self.assertTrue(status.input_ready)
        self.assertEqual(status.stt_provider, "huggingface-realtime")

    def test_huggingface_local_provider_requires_local_backend_command(self) -> None:
        def no_speech_to_speech(name: str) -> str | None:
            if name == "speech-to-speech":
                return None
            return "/usr/bin/" + name

        with (
            patch.dict("os.environ", {"AGENT_VOICE_PROVIDER": "huggingface"}, clear=True),
            patch("agent.voice.shutil.which", side_effect=no_speech_to_speech),
            patch("agent.voice.importlib.util.find_spec", return_value=object()),
        ):
            status = voice_status()

        self.assertFalse(status.input_ready)
        self.assertIn("speech-to-speech", status.input_reason or "")

    def test_huggingface_local_start_command_uses_open_source_backend(self) -> None:
        with (
            patch.dict("os.environ", {"AGENT_VOICE_PROVIDER": "huggingface"}, clear=True),
            patch("agent.voice.shutil.which", return_value="/usr/bin/speech-to-speech"),
        ):
            command = _local_realtime_start_command(VoiceConfig.from_environment())

        self.assertIsNotNone(command)
        self.assertIn("--llm_backend", command or ())
        self.assertIn("transformers", command or ())
        self.assertNotIn("responses-api", command or ())

    def test_realtime_voice_requires_url(self) -> None:
        with (
            patch.dict("os.environ", {"AGENT_VOICE_MODE": "realtime"}, clear=True),
            patch("agent.voice.shutil.which", side_effect=lambda name: "/usr/bin/" + name),
            patch("agent.voice.importlib.util.find_spec", return_value=object()),
        ):
            status = voice_status()

        self.assertFalse(status.input_ready)
        self.assertIn("AGENT_REALTIME_VOICE_URL", status.input_reason or "")

    def test_realtime_session_url_normalizes_hosts(self) -> None:
        self.assertEqual(
            _realtime_session_url("https://voice.example"),
            "https://voice.example/session",
        )
        self.assertEqual(
            _realtime_session_url("voice.example/api/session"),
            "https://voice.example/api/session",
        )
        self.assertEqual(
            _realtime_session_url("https://huggingface.co/spaces/smolagents/hf-realtime-voice"),
            "https://smolagents-hf-realtime-voice.hf.space/session",
        )

    def test_realtime_session_url_candidates_include_api_fallback(self) -> None:
        self.assertEqual(
            _realtime_session_url_candidates("https://voice.example"),
            ["https://voice.example/session", "https://voice.example/api/session"],
        )
        self.assertEqual(
            _realtime_session_url_candidates("https://voice.example/api/session"),
            ["https://voice.example/api/session"],
        )

    def test_realtime_connect_url_falls_back_to_api_session(self) -> None:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"connect_url": "wss://voice.example/realtime"}'

        def urlopen(request: object, timeout: int = 0) -> Response:
            self.assertEqual(timeout, 15)
            url = getattr(request, "full_url")
            if url == "https://voice.example/session":
                raise urllib.error.HTTPError(url, 405, "Method Not Allowed", {}, None)
            self.assertEqual(url, "https://voice.example/api/session")
            return Response()

        with patch("agent.voice.urllib.request.urlopen", side_effect=urlopen):
            self.assertEqual(
                _realtime_connect_url("https://voice.example"),
                "wss://voice.example/realtime",
            )

    def test_realtime_connect_url_sends_realtime_bearer_token(self) -> None:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"connect_url": "wss://voice.example/realtime"}'

        def urlopen(request: object, timeout: int = 0) -> Response:
            self.assertEqual(timeout, 15)
            self.assertEqual(request.get_header("Authorization"), "Bearer hf-secret")
            return Response()

        with patch("agent.voice.urllib.request.urlopen", side_effect=urlopen):
            self.assertEqual(
                _realtime_connect_url("https://voice.example", api_key="hf-secret"),
                "wss://voice.example/realtime",
            )

    def test_realtime_voice_api_key_uses_huggingface_token_environment(self) -> None:
        with patch.dict("os.environ", {"HF_TOKEN": "hf-secret"}, clear=True):
            config = VoiceConfig.from_environment()

        self.assertEqual(config.realtime_api_key, "hf-secret")

    def test_realtime_connect_url_accepts_direct_websocket(self) -> None:
        self.assertEqual(
            _realtime_connect_url("wss://voice.example/realtime"),
            "wss://voice.example/realtime",
        )

    def test_transcript_fragment_accepts_only_input_transcription_events(self) -> None:
        self.assertEqual(
            _transcript_fragment({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "close Spark",
            }),
            "close Spark",
        )
        self.assertEqual(
            _transcript_fragment({
                "type": "response.audio_transcript.delta",
                "delta": "done",
            }),
            "",
        )

    def test_chunk_bytes_splits_pcm_for_realtime_append_events(self) -> None:
        self.assertEqual(list(_chunk_bytes(b"abcdef", 2)), [b"ab", b"cd", b"ef"])


if __name__ == "__main__":
    unittest.main()
