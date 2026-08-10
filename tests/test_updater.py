from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.main import _handle_local_command
from agent.updater import UpdateResult, UpdateStatus, apply_update, check_for_update, update_source_root


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


class UpdaterTests(unittest.TestCase):
    def _repositories(self, root: Path) -> tuple[Path, Path]:
        remote = root / "remote.git"
        author = root / "author"
        checkout = root / "checkout"
        _git("init", "--bare", str(remote), cwd=root)
        _git("init", "-b", "main", str(author), cwd=root)
        _git("config", "user.email", "test@example.com", cwd=author)
        _git("config", "user.name", "Updater Test", cwd=author)
        (author / "pyproject.toml").write_text("[project]\nname='agent'\nversion='0.1.0'\n")
        _git("add", "pyproject.toml", cwd=author)
        _git("commit", "-m", "initial", cwd=author)
        _git("remote", "add", "origin", str(remote), cwd=author)
        _git("push", "-u", "origin", "main", cwd=author)
        _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)
        _git("clone", str(remote), str(checkout), cwd=root)
        return author, checkout

    def test_check_detects_remote_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            author, checkout = self._repositories(Path(tmp))
            (author / "change.txt").write_text("new behavior\n")
            _git("add", "change.txt", cwd=author)
            _git("commit", "-m", "add updater behavior", cwd=author)
            _git("push", cwd=author)

            with patch.dict("os.environ", {"AGENT_UPDATE_ROOT": str(checkout)}, clear=False):
                status = check_for_update()

        self.assertTrue(status.supported)
        self.assertTrue(status.available)
        self.assertEqual(status.count, 1)
        self.assertIn("add updater behavior", status.changes[0])

    def test_apply_update_fast_forwards_and_installs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            author, checkout = self._repositories(Path(tmp))
            (author / "change.txt").write_text("new behavior\n")
            _git("add", "change.txt", cwd=author)
            _git("commit", "-m", "ship update", cwd=author)
            _git("push", cwd=author)
            install = subprocess.CompletedProcess(["pip"], 0, "installed", "")
            state_path = Path(tmp) / "state" / "update.json"

            with (
                patch.dict("os.environ", {"AGENT_UPDATE_ROOT": str(checkout)}, clear=False),
                patch("agent.updater._run_install", return_value=install) as run_install,
                patch("agent.updater._update_state_path", return_value=state_path),
            ):
                result = apply_update()

            self.assertTrue(result.ok)
            self.assertTrue(result.updated)
            self.assertTrue((checkout / "change.txt").is_file())
            run_install.assert_called_once_with(checkout.resolve())

    def test_apply_update_preserves_dirty_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            author, checkout = self._repositories(Path(tmp))
            (author / "change.txt").write_text("upstream\n")
            _git("add", "change.txt", cwd=author)
            _git("commit", "-m", "ship update", cwd=author)
            _git("push", cwd=author)
            (checkout / "pyproject.toml").write_text("local edit\n")
            install = subprocess.CompletedProcess(["pip"], 0, "installed", "")
            state_path = Path(tmp) / "state" / "update.json"

            with (
                patch.dict("os.environ", {"AGENT_UPDATE_ROOT": str(checkout)}, clear=False),
                patch("agent.updater._run_install", return_value=install) as run_install,
                patch("agent.updater._update_state_path", return_value=state_path),
            ):
                result = apply_update()

            self.assertTrue(result.ok)
            self.assertTrue(result.updated)
            self.assertEqual((checkout / "pyproject.toml").read_text(), "local edit\n")
            installed_from = run_install.call_args.args[0]
            self.assertNotEqual(installed_from, checkout.resolve())
            self.assertFalse(installed_from.exists())
            self.assertTrue(state_path.is_file())

    def test_update_source_root_honors_explicit_install_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            (root / "pyproject.toml").write_text("[project]\n")
            with patch.dict("os.environ", {"AGENT_UPDATE_ROOT": str(root)}, clear=False):
                self.assertEqual(update_source_root(), root.resolve())

    def test_update_command_returns_confirmation_contract(self) -> None:
        status = UpdateStatus(
            supported=True,
            available=True,
            current="111111111111",
            latest="222222222222",
            count=1,
            changes=("2222222 update behavior",),
        )
        ctx = SimpleNamespace(last_local_command_result=None)
        with patch("agent.updater.check_for_update", return_value=status):
            text = _handle_local_command(ctx, "/update")

        self.assertIn("Update available", text or "")
        self.assertEqual(ctx.last_local_command_result["code"], "update_confirmation_required")
        self.assertEqual(ctx.last_local_command_result["next_command"], "/update --yes")

    def test_confirmed_update_requests_restart(self) -> None:
        status = UpdateStatus(supported=True, current="222222222222", latest="222222222222")
        update = UpdateResult(ok=True, updated=True, status=status)
        ctx = SimpleNamespace(last_local_command_result=None)
        with patch("agent.updater.apply_update", return_value=update):
            text = _handle_local_command(ctx, "/update --yes")

        self.assertIn("Restart Nym", text or "")
        self.assertEqual(ctx.last_local_command_result["code"], "update_complete")


if __name__ == "__main__":
    unittest.main()
