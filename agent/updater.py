from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .process_env import credential_free_environment


UPDATE_REMOTE = "origin"
UPDATE_COMMAND_TIMEOUT_SECONDS = 12
UPDATE_COMMAND = "nym --update"
# Kept as an import-compatible alias for older integrations.
UPDATE_INSTALL_COMMAND = UPDATE_COMMAND


@dataclass(frozen=True)
class UpdateStatus:
    supported: bool
    available: bool = False
    current: str | None = None
    latest: str | None = None
    count: int = 0
    changes: tuple[str, ...] = ()
    source_root: str | None = None
    upstream: str | None = None
    isolated_install: bool = False
    error: str | None = None


def apply_update() -> int:
    status = check_for_update()
    if not status.supported:
        print(f"Nym cannot update this installation automatically: {status.error}")
        return 1
    if status.error:
        print(f"Nym update could not start: {status.error}")
        return 1
    if not status.available:
        current = f" ({status.current})" if status.current else ""
        print(f"Nym is already up to date{current}.")
        return 0
    if not status.source_root or not status.upstream:
        print("Nym update could not identify the managed checkout or upstream branch.")
        return 1

    root = Path(status.source_root)
    dirty = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    isolated = status.isolated_install or bool(dirty)
    install_root = root
    temporary_root: tempfile.TemporaryDirectory[str] | None = None
    if isolated:
        print("Local checkout changes detected; updating Nym from an isolated clean revision.")
        temporary_root = tempfile.TemporaryDirectory(prefix="nym-update-")
        install_root = Path(temporary_root.name) / "checkout"
        worktree = _run_git(
            root,
            "worktree",
            "add",
            "--detach",
            str(install_root),
            status.upstream,
        )
        if worktree.returncode != 0:
            temporary_root.cleanup()
            print(f"Nym could not prepare a clean update: {_command_error(worktree, 'git worktree failed.')}")
            return 1
    else:
        print(f"Updating Nym: {status.current or 'current'} -> {status.latest or 'latest'}")
        merge = _run_git(root, "merge", "--ff-only", status.upstream)
        if merge.returncode != 0:
            print(f"Nym could not update its checkout: {_command_error(merge, 'git merge failed.')}")
            return 1

    expected_revision = _git_output(install_root, "rev-parse", "HEAD")
    if expected_revision is None:
        if temporary_root is not None:
            _run_git(root, "worktree", "remove", "--force", str(install_root))
            temporary_root.cleanup()
        print("Nym could not identify the revision prepared for installation.")
        return 1

    print("Refreshing the installed Nym runtime...")
    install = _refresh_installed_runtime(
        install_root,
        cargo_target_dir=root / "agent-rust" / "target",
        expected_revision=expected_revision,
    )
    if temporary_root is not None:
        _run_git(root, "worktree", "remove", "--force", str(install_root))
        temporary_root.cleanup()
    if install.returncode != 0:
        print("Nym files were downloaded, but the installed runtime could not be refreshed.")
        print("Run `nym --update` again after resolving the installation error above.")
        return 1

    revision = _git_output(root, "rev-parse", "--short=12", "HEAD")
    installed_revision = status.latest if isolated else revision
    print(f"Nym update complete{f' ({installed_revision})' if installed_revision else ''}.")
    if isolated:
        print(f"Local checkout preserved unchanged: {root}")
    print("Run: nym --tui")
    return 0


def _install_updated_runtime(
    root: Path,
    *,
    cargo_target_dir: Path,
    force: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = credential_free_environment()
    environment["CARGO_TARGET_DIR"] = str(cargo_target_dir.resolve(strict=False))
    revision = _git_output(root, "rev-parse", "HEAD")
    if revision:
        environment["NYM_BUILD_REVISION"] = revision
    try:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--upgrade-strategy",
            "only-if-needed",
            "--no-cache-dir",
        ]
        if force:
            command.extend(("--force-reinstall", "--no-deps"))
        command.append(str(root))
        return subprocess.run(
            command,
            cwd=root,
            env=environment,
            text=True,
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess([sys.executable, "-m", "pip"], 1, "", str(exc))


def _refresh_installed_runtime(
    root: Path,
    *,
    cargo_target_dir: Path,
    expected_revision: str,
) -> subprocess.CompletedProcess[str]:
    install = _install_updated_runtime(root, cargo_target_dir=cargo_target_dir)
    if install.returncode != 0 or _installed_package_revision() == expected_revision:
        return install

    print("The installed runtime remained stale; forcing a package refresh...")
    install = _install_updated_runtime(root, cargo_target_dir=cargo_target_dir, force=True)
    if install.returncode != 0 or _installed_package_revision() == expected_revision:
        return install
    return subprocess.CompletedProcess(
        install.args,
        1,
        install.stdout,
        "The installed runtime revision does not match the downloaded update.",
    )


def _installed_package_revision() -> str | None:
    try:
        marker = importlib.metadata.distribution("agent").locate_file("agent/_build_revision")
        revision = Path(marker).read_text(encoding="ascii").strip()
    except (importlib.metadata.PackageNotFoundError, OSError):
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return None
    return revision.lower()

def check_for_update() -> UpdateStatus:
    root = update_source_root()
    if root is None:
        return UpdateStatus(
            supported=False,
            error="This installation is not connected to an update checkout.",
        )

    current_ref = _installed_revision(root)
    current = _git_output(root, "rev-parse", "--short=12", current_ref)
    if current is None:
        return _status_error(root, "Could not read the installed revision.")

    fetch = _run_git(root, "fetch", "--quiet", "--prune", UPDATE_REMOTE)
    if fetch.returncode != 0:
        return _status_error(root, _command_error(fetch, "Could not check for updates."), current=current)

    upstream = _upstream_ref(root)
    if upstream is None:
        return _status_error(root, "The checkout has no tracked upstream branch.", current=current)
    latest = _git_output(root, "rev-parse", "--short=12", upstream)
    if latest is None:
        return _status_error(root, "Could not read the latest upstream revision.", current=current)

    counts = _git_output(root, "rev-list", "--left-right", "--count", f"{current_ref}...{upstream}")
    if counts is None:
        return _status_error(root, "Could not compare installed and upstream revisions.", current=current)
    try:
        ahead, behind = (int(value) for value in counts.split())
    except (TypeError, ValueError):
        return _status_error(root, "Git returned an invalid update comparison.", current=current)

    checkout_counts = _git_output(root, "rev-list", "--left-right", "--count", f"HEAD...{upstream}")
    if checkout_counts is None:
        return _status_error(root, "Could not compare the update checkout with upstream.", current=current)
    try:
        checkout_ahead, _checkout_behind = (int(value) for value in checkout_counts.split())
    except (TypeError, ValueError):
        return _status_error(root, "Git returned an invalid checkout comparison.", current=current)

    changes: tuple[str, ...] = ()
    if behind:
        log = _git_output(root, "log", "--format=%h %s", "-n", "5", f"{current_ref}..{upstream}")
        changes = tuple(line.strip() for line in (log or "").splitlines() if line.strip())
    return UpdateStatus(
        supported=True,
        available=behind > 0,
        current=current,
        latest=latest,
        count=behind,
        changes=changes,
        source_root=str(root),
        upstream=upstream,
        isolated_install=checkout_ahead > 0,
    )

def update_source_root() -> Path | None:
    candidates: list[Path] = []
    override = os.environ.get("AGENT_UPDATE_ROOT") or os.environ.get("NYM_INSTALL_ROOT")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(Path(__file__).resolve().parents[1])
    direct_root = _direct_url_source_root()
    if direct_root is not None:
        candidates.append(direct_root)
    candidates.append(Path.home() / ".local" / "share" / "nym")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / ".git").exists() and (resolved / "pyproject.toml").is_file():
            return resolved
    return None


def _direct_url_source_root() -> Path | None:
    try:
        raw = importlib.metadata.distribution("agent").read_text("direct_url.json")
        payload = json.loads(raw or "{}")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
        return None
    url = payload.get("url") if isinstance(payload, dict) else None
    if not isinstance(url, str):
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file":
        return None
    path = urllib.parse.unquote(parsed.path)
    if platform.system().casefold() == "windows" and path.startswith("/"):
        path = path[1:]
    return Path(path)


def _installed_revision(root: Path) -> str:
    revision = _build_revision_marker()
    if revision is None:
        return "HEAD"
    exists = _run_git(root, "cat-file", "-e", f"{revision}^{{commit}}")
    return revision if exists.returncode == 0 else "HEAD"


def _build_revision_marker() -> str | None:
    try:
        revision = Path(__file__).with_name("_build_revision").read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return None
    return revision


def _upstream_ref(root: Path) -> str | None:
    tracked = _git_output(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    if tracked:
        return tracked
    fallback = f"{UPDATE_REMOTE}/main"
    exists = _run_git(root, "show-ref", "--verify", "--quiet", f"refs/remotes/{fallback}")
    return fallback if exists.returncode == 0 else None


def _status_error(root: Path, error: str, *, current: str | None = None) -> UpdateStatus:
    return UpdateStatus(
        supported=True,
        current=current,
        source_root=str(root),
        error=error,
    )


def _git_output(root: Path, *args: str) -> str | None:
    result = _run_git(root, *args)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = credential_free_environment()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=UPDATE_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(["git", *args], 1, "", str(exc))

def _command_error(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    detail = " ".join((result.stderr or result.stdout).split())
    if not detail:
        return fallback
    return f"{fallback} {detail[-600:]}"
