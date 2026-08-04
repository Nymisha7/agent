from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SESSION_PRUNE_AFTER_DAYS = 30
DEFAULT_SESSION_MAX_ENTRIES = 500
DEFAULT_SESSION_WRITE_LOCK_ACQUIRE_TIMEOUT_MS = 60_000


@dataclass(frozen=True)
class ProjectInfo:
    id: str
    root: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class WorkspaceInfo:
    id: str
    project_id: str
    root: str
    cwd: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TokenUsage:
    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0


@dataclass(frozen=True)
class SessionInfo:
    id: str
    project_id: str | None
    workspace_id: str | None
    title: str
    workspace_root: str
    cwd: str | None
    created_at: str
    updated_at: str
    provider: str | None
    model: str | None
    agent: str | None
    permission: dict[str, Any] | None
    cost_usd: float
    tokens: TokenUsage
    summary: str | None
    active_root: str | None
    focus_path: str | None
    last_prompt: str | None
    state: dict[str, Any] | None


@dataclass(frozen=True)
class MessageInfo:
    id: int
    session_id: str
    seq: int
    role: str
    content: str
    created_at: str
    attachments: tuple["AttachmentInfo", ...] = ()


@dataclass(frozen=True)
class AttachmentInfo:
    id: str
    filename: str
    mime: str
    size_bytes: int
    sha256: str
    storage_path: str
    source: str
    created_at: str


@dataclass(frozen=True)
class EventInfo:
    id: int
    session_id: str
    seq: int
    event_type: str
    tool: str | None
    args: dict[str, Any] | None
    path: str | None
    summary: str
    data: dict[str, Any] | None
    created_at: str


@dataclass(frozen=True)
class SessionRouteInfo:
    route_key: str
    session_id: str
    agent_id: str
    scope: str
    channel: str
    account_id: str
    peer_kind: str | None
    peer_id: str | None
    sender_id: str | None
    guild_id: str | None
    team_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SessionMaintenanceReport:
    mode: str
    before_count: int
    after_count: int
    pruned: int
    capped: int
    applied: bool


def default_db_path() -> Path:
    return data_home() / "agent" / "sessions.sqlite3"


def workspace_db_path() -> Path:
    return Path.cwd() / ".agent-session.sqlite3"


def data_home() -> Path:
    if path := _env_path("XDG_DATA_HOME"):
        return path
    return Path.home() / ".local" / "share"


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_title(value: str | None, *, max_length: int = 80) -> str | None:
    if value is None:
        return None
    text = " ".join(value.split())
    if not text:
        return None
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1]}..."


def _maintenance_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def _is_unusable_default_db_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return True
    text = str(exc).casefold()
    return any(
        marker in text
        for marker in (
            "attempt to write a readonly database",
            "readonly database",
            "unable to open database file",
            "permission denied",
            "disk i/o error",
        )
    )


_sqlx_module = sys.modules.get(f"{__package__}.sqlx_session_store")
if _sqlx_module is None or hasattr(_sqlx_module, "SessionStore"):
    from .sqlx_session_store import SessionStore as SessionStore
