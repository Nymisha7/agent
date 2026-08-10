from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
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
    preserved_changes = bool(dirty)
    if preserved_changes:
        stash = _run_git(
            root,
            "stash",
            "push",
            "--include-untracked",
            "-m",
            "Nym automatic update backup",
        )
        if stash.returncode != 0:
            print(f"Nym could not preserve local checkout changes: {_command_error(stash, 'git stash failed.')}")
            return 1
        print("Preserved local checkout changes in the Git stash.")

    print(f"Updating Nym: {status.current or 'current'} -> {status.latest or 'latest'}")
    merge = _run_git(root, "merge", "--ff-only", status.upstream)
    if merge.returncode != 0:
        print(f"Nym could not update its checkout: {_command_error(merge, 'git merge failed.')}")
        return 1

    print("Refreshing the installed Nym runtime...")
    install = _install_updated_runtime(root)
    if install.returncode != 0:
        print("Nym files were downloaded, but the installed runtime could not be refreshed.")
        print("Run `nym --update` again after resolving the installation error above.")
        return 1

    revision = _git_output(root, "rev-parse", "--short=12", "HEAD")
    print(f"Nym update complete{f' ({revision})' if revision else ''}.")
    if preserved_changes:
        print(f"Your previous checkout changes remain available with: git -C {root} stash list")
    print("Run: nym --tui")
    return 0


def _install_updated_runtime(root: Path) -> subprocess.CompletedProcess[str]:
    environment = credential_free_environment()
    revision = _git_output(root, "rev-parse", "HEAD")
    if revision:
        environment["NYM_BUILD_REVISION"] = revision
    try:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--upgrade-strategy",
                "only-if-needed",
                "--no-cache-dir",
                str(root),
            ],
            cwd=root,
            env=environment,
            text=True,
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess([sys.executable, "-m", "pip"], 1, "", str(exc))

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

    if ahead and behind:
        return UpdateStatus(
            supported=True,
            current=current,
            latest=latest,
            source_root=str(root),
            upstream=upstream,
            error="The local checkout and upstream branch have diverged.",
        )

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
