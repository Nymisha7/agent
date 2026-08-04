import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.attachments import import_attachment, maintain_attachment_store


class AttachmentStoreTests(unittest.TestCase):
    def test_import_uses_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.txt"
            source.write_text("private", encoding="utf-8")
            data_home = root / "data"
            with patch.dict("os.environ", {"XDG_DATA_HOME": str(data_home)}, clear=True):
                attachment = import_attachment(source, source="test")

            stored = Path(attachment.storage_path)
            self.assertEqual(stored.stat().st_mode & 0o777, 0o600)
            self.assertEqual(stored.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(stored.parents[1].stat().st_mode & 0o777, 0o700)

    def test_maintenance_removes_expired_attachments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old.txt"
            source.write_text("expired", encoding="utf-8")
            environment = {
                "XDG_DATA_HOME": str(root / "data"),
                "AGENT_ATTACHMENT_RETENTION_DAYS": "1",
            }
            with patch.dict("os.environ", environment, clear=True):
                attachment = import_attachment(source, source="test")
                stored = Path(attachment.storage_path)
                old = 1_700_000_000.0
                os.utime(stored, (old, old))
                report = maintain_attachment_store(force=True, now=old + 2 * 24 * 60 * 60)

            self.assertFalse(stored.exists())
            self.assertEqual(report["removed"], 1)
            self.assertEqual(report["bytes_removed"], len("expired"))

    def test_maintenance_enforces_configured_store_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            environment = {
                "XDG_DATA_HOME": str(root / "data"),
                "AGENT_ATTACHMENT_STORE_MAX_BYTES": "6",
                "AGENT_ATTACHMENT_RETENTION_DAYS": "30",
            }
            with patch.dict("os.environ", environment, clear=True):
                first_attachment = import_attachment(first, source="test")
                first_path = Path(first_attachment.storage_path)
                os.utime(first_path, (1_700_000_000.0, 1_700_000_000.0))
                second_attachment = import_attachment(second, source="test")
                second_path = Path(second_attachment.storage_path)
                report = maintain_attachment_store(force=True)

            self.assertFalse(first_path.exists())
            self.assertTrue(second_path.exists())
            self.assertLessEqual(report["bytes_kept"], 6)


if __name__ == "__main__":
    unittest.main()
