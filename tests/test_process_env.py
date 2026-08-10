import os
import unittest
from unittest.mock import patch

from agent.process_env import credential_free_environment


class ProcessEnvironmentTests(unittest.TestCase):
    def test_credentials_are_not_forwarded_to_helper_processes(self) -> None:
        environment = {
            "OPENAI_API_KEY": "openai-secret",
            "HF_TOKEN": "hf-secret",
            "AGENT_CREDENTIAL_ENCRYPTION_KEY": "store-secret",
            "AWS_ACCESS_KEY_ID": "aws-id",
            "PATH": "/usr/bin",
            "PULSE_SERVER": "unix:/run/user/1000/pulse/native",
        }
        with patch.dict(os.environ, environment, clear=True):
            sanitized = credential_free_environment()

        self.assertEqual(sanitized["PATH"], "/usr/bin")
        self.assertEqual(sanitized["PULSE_SERVER"], "unix:/run/user/1000/pulse/native")
        self.assertNotIn("OPENAI_API_KEY", sanitized)
        self.assertNotIn("HF_TOKEN", sanitized)
        self.assertNotIn("AGENT_CREDENTIAL_ENCRYPTION_KEY", sanitized)
        self.assertNotIn("AWS_ACCESS_KEY_ID", sanitized)


if __name__ == "__main__":
    unittest.main()
