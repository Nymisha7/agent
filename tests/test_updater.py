from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent.main import _tui_update_required, main
from agent.updater import (
    UPDATE_COMMAND,
    UpdateStatus,
    _install_updated_runtime,
    apply_update,
    check_for_update,
    update_source_root,
)


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

    def test_build_revision_detects_old_runtime_after_checkout_was_pulled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            author, checkout = self._repositories(Path(tmp))
            initial = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            (author / "change.txt").write_text("new behavior\n")
            _git("add", "change.txt", cwd=author)
            _git("commit", "-m", "ship new runtime", cwd=author)
            _git("push", cwd=author)
            _git("pull", "--ff-only", cwd=checkout)

            with (
                patch.dict("os.environ", {"AGENT_UPDATE_ROOT": str(checkout)}, clear=False),
                patch("agent.updater._build_revision_marker", return_value=initial),
            ):
                status = check_for_update()

        self.assertTrue(status.available)
        self.assertEqual(status.count, 1)

    def test_update_source_root_honors_explicit_install_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            (root / "pyproject.toml").write_text("[project]\n")
            with patch.dict("os.environ", {"AGENT_UPDATE_ROOT": str(root)}, clear=False):
                self.assertEqual(update_source_root(), root.resolve())

    def test_startup_gate_prints_update_command_for_outdated_tui(self) -> None:
        status = UpdateStatus(
            supported=True,
            available=True,
            current="111111111111",
            latest="222222222222",
            count=1,
            changes=("2222222 update behavior",),
        )
        output = io.StringIO()
        with (
            patch("agent.updater.check_for_update", return_value=status),
            redirect_stdout(output),
        ):
            blocked = _tui_update_required()

        self.assertTrue(blocked)
        self.assertIn(UPDATE_COMMAND, output.getvalue())
        self.assertIn("nym --tui", output.getvalue())

    def test_apply_update_fast_forwards_and_refreshes_current_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            author, checkout = self._repositories(Path(tmp))
            (author / "change.txt").write_text("new behavior\n")
            _git("add", "change.txt", cwd=author)
            _git("commit", "-m", "ship update command", cwd=author)
            _git("push", cwd=author)

            with (
                patch.dict("os.environ", {"AGENT_UPDATE_ROOT": str(checkout)}, clear=False),
                patch(
                    "agent.updater._install_updated_runtime",
                    return_value=subprocess.CompletedProcess([], 0),
                ) as install,
                redirect_stdout(io.StringIO()),
            ):
                result = apply_update()

            author_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=author,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            checkout_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        self.assertEqual(result, 0)
        self.assertEqual(checkout_head, author_head)
        install.assert_called_once_with(
            checkout,
            cargo_target_dir=checkout / "agent-rust" / "target",
        )

    def test_runtime_refresh_reuses_managed_cargo_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkout"
            target = Path(tmp) / "managed" / "agent-rust" / "target"
            root.mkdir()
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch("agent.updater._git_output", return_value="a" * 40),
                patch("agent.updater.subprocess.run", return_value=completed) as run,
            ):
                result = _install_updated_runtime(root, cargo_target_dir=target)

        self.assertIs(result, completed)
        self.assertEqual(
            run.call_args.kwargs["env"]["CARGO_TARGET_DIR"],
            str(target),
        )
        self.assertEqual(run.call_args.kwargs["cwd"], root)

    def test_diverged_checkout_updates_from_isolated_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            author, checkout = self._repositories(Path(tmp))
            _git("config", "user.email", "local@example.com", cwd=checkout)
            _git("config", "user.name", "Local User", cwd=checkout)
            (checkout / "local.txt").write_text("preserve me\n")
            _git("add", "local.txt", cwd=checkout)
            _git("commit", "-m", "local checkout change", cwd=checkout)
            local_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            (author / "remote.txt").write_text("update\n")
            _git("add", "remote.txt", cwd=author)
            _git("commit", "-m", "remote update", cwd=author)
            _git("push", cwd=author)
            installed_heads: list[str] = []

            build_targets: list[Path] = []

            def install(
                root: Path,
                *,
                cargo_target_dir: Path,
            ) -> subprocess.CompletedProcess[str]:
                installed_heads.append(
                    subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=root,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                )
                build_targets.append(cargo_target_dir)
                return subprocess.CompletedProcess([], 0)

            with (
                patch.dict("os.environ", {"AGENT_UPDATE_ROOT": str(checkout)}, clear=False),
                patch("agent.updater._install_updated_runtime", side_effect=install),
                redirect_stdout(io.StringIO()),
            ):
                status = check_for_update()
                result = apply_update()

            checkout_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            remote_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=author,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        self.assertTrue(status.available)
        self.assertTrue(status.isolated_install)
        self.assertEqual(result, 0)
        self.assertEqual(checkout_head, local_head)
        self.assertEqual(installed_heads, [remote_head])
        self.assertEqual(build_targets, [checkout / "agent-rust" / "target"])

    def test_update_cli_runs_before_tui_or_session_startup(self) -> None:
        with (
            patch("agent.updater.apply_update", return_value=0) as update,
            patch("agent.main._tui_update_required") as update_gate,
            patch("agent.main.SessionStore.default") as open_store,
        ):
            result = main(["--update"])

        self.assertEqual(result, 0)
        update.assert_called_once_with()
        update_gate.assert_not_called()
        open_store.assert_not_called()

    def test_startup_gate_allows_current_or_unverifiable_install(self) -> None:
        for status in (
            UpdateStatus(supported=True, available=False),
            UpdateStatus(supported=True, error="network unavailable"),
        ):
            with patch("agent.updater.check_for_update", return_value=status):
                self.assertFalse(_tui_update_required())

    def test_outdated_tui_exits_before_opening_session_store(self) -> None:
        with (
            patch("agent.main._tui_update_required", return_value=True),
            patch("agent.main.SessionStore.default") as open_store,
        ):
            result = main(["--tui"])

        self.assertEqual(result, 3)
        open_store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
