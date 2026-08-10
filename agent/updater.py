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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .process_env import credential_free_environment


UPDATE_REMOTE = "origin"
UPDATE_COMMAND_TIMEOUT_SECONDS = 12
UPDATE_INSTALL_TIMEOUT_SECONDS = 30 * 60


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

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UpdateResult:
    ok: bool
    updated: bool
    status: UpdateStatus
    error: str | None = None


def check_for_update() -> UpdateStatus:
    root = update_source_root()
    if root is None:
        return UpdateStatus(
            supported=False,
            error="This installation is not connected to an update checkout.",
        )

    current_ref = _installed_revision(root) or "HEAD"
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
            error="The local checkout and upstream branch have diverged; automatic update is unavailable.",
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


def apply_update(progress: Callable[[str], None] | None = None) -> UpdateResult:
    status = check_for_update()
    if not status.supported or status.error:
        return UpdateResult(False, False, status, status.error)
    if not status.available:
        return UpdateResult(True, False, status)

    root = Path(status.source_root or "")
    dirty_result = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty_result.returncode != 0:
        return UpdateResult(
            False,
            False,
            status,
            _command_error(dirty_result, "Could not inspect the update checkout."),
        )
    dirty = bool(dirty_result.stdout.strip())
    install_root = root
    temporary_root: tempfile.TemporaryDirectory[str] | None = None
    if dirty:
        if progress:
            progress("Local checkout has changes; preparing an isolated update")
        temporary_root = tempfile.TemporaryDirectory(prefix="nym-update-")
        install_root = Path(temporary_root.name) / "checkout"
        worktree = _run_git(root, "worktree", "add", "--detach", str(install_root), status.upstream or "")
        if worktree.returncode != 0:
            temporary_root.cleanup()
            return UpdateResult(
                False,
                False,
                status,
                _command_error(worktree, "Could not prepare an isolated update checkout."),
            )
    else:
        if progress:
            progress(f"Updating source to {status.latest}")
        merge = _run_git(root, "merge", "--ff-only", status.upstream or "")
        if merge.returncode != 0:
            return UpdateResult(False, False, status, _command_error(merge, "Fast-forward update failed."))

    if progress:
        progress("Rebuilding the Nym runtime")
    install = _run_install(install_root)
    if temporary_root is not None:
        _run_git(root, "worktree", "remove", "--force", str(install_root))
        temporary_root.cleanup()
    if install.returncode != 0:
        return UpdateResult(
            False,
            False,
            status,
            _command_error(install, "The update was prepared, but runtime installation failed."),
        )

    latest_revision = _git_output(root, "rev-parse", status.upstream or "")
    if latest_revision:
        _write_installed_revision(root, latest_revision)
    installed = UpdateStatus(
        supported=True,
        available=False,
        current=status.latest,
        latest=status.latest,
        source_root=status.source_root,
        upstream=status.upstream,
    )
    if progress:
        progress("Update installed; restart Nym to use it")
    return UpdateResult(True, True, installed)


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


def _update_state_path() -> Path:
    state_home = Path(
        os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
    ).expanduser()
    return state_home / "agent" / "update.json"


def _installed_revision(root: Path) -> str | None:
    try:
        payload = json.loads(_update_state_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("source_root") != str(root):
        return None
    revision = payload.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return None
    exists = _run_git(root, "cat-file", "-e", f"{revision}^{{commit}}")
    return revision if exists.returncode == 0 else None


def _write_installed_revision(root: Path, revision: str) -> None:
    path = _update_state_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"source_root": str(root), "revision": revision}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        temporary.replace(path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


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


def _run_install(root: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                str(root),
            ],
            env=credential_free_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=UPDATE_INSTALL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess([sys.executable, "-m", "pip"], 1, "", str(exc))


def _command_error(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    detail = " ".join((result.stderr or result.stdout).split())
    if not detail:
        return fallback
    return f"{fallback} {detail[-600:]}"
