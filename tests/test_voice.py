import os
import unittest
from unittest.mock import patch

from agent.voice import VoiceConfig, bridge_voice_record, bridge_voice_speak, voice_status


class VoiceTests(unittest.TestCase):
    def test_voice_uses_existing_openai_key_without_network_probe(self) -> None:
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "existing-key"}, clear=True),
            patch("agent.voice.shutil.which", side_effect=lambda name: "/usr/bin/" + name),
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


if __name__ == "__main__":
    unittest.main()
