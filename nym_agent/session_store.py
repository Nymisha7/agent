from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PERMISSION = {
    "read_files": "allow",
    "write_files": "ask",
    "shell": "ask",
    "network": "ask",
}


SCHEMA = """
create table if not exists projects (
    id text primary key,
    root text not null unique,
    title text not null,
    created_at text not null,
    updated_at text not null
);

create table if not exists workspaces (
    id text primary key,
    project_id text not null,
    root text not null,
    cwd text not null,
    created_at text not null,
    updated_at text not null,
    foreign key (project_id) references projects(id) on delete cascade,
    unique (project_id, root, cwd)
);

create table if not exists sessions (
    id text primary key,
    project_id text,
    workspace_id text,
    title text not null,
    workspace_root text not null,
    cwd text,
    created_at text not null,
    updated_at text not null,
    provider text,
    model text,
    agent text,
    permission_json text,
    cost_usd real not null default 0,
    tokens_input integer not null default 0,
    tokens_output integer not null default 0,
    tokens_reasoning integer not null default 0,
    tokens_cache_read integer not null default 0,
    tokens_cache_write integer not null default 0,
    summary text,
    active_root text,
    focus_path text,
    last_prompt text,
    state_json text,
    foreign key (project_id) references projects(id),
    foreign key (workspace_id) references workspaces(id)
);

create table if not exists messages (
    id integer primary key autoincrement,
    session_id text not null,
    seq integer not null,
    role text not null,
    content text not null,
    created_at text not null,
    foreign key (session_id) references sessions(id) on delete cascade
);

create table if not exists events (
    id integer primary key autoincrement,
    session_id text not null,
    seq integer not null,
    event_type text not null,
    tool text,
    path text,
    summary text not null,
    args_json text,
    data_json text,
    created_at text not null,
    foreign key (session_id) references sessions(id) on delete cascade
);

create index if not exists idx_sessions_updated_at
    on sessions(updated_at desc);

create index if not exists idx_messages_session_seq
    on messages(session_id, seq);

create index if not exists idx_events_session_seq
    on events(session_id, seq);

create index if not exists idx_events_session_created_at
    on events(session_id, created_at desc);
"""


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


class SessionStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def default(cls) -> SessionStore:
        explicit_path = _env_path("NYM_SESSION_DB")
        if explicit_path is not None:
            return cls(explicit_path)

        preferred_path = default_db_path()
        try:
            return cls(preferred_path)
        except (OSError, sqlite3.OperationalError) as exc:
            if not _is_unusable_default_db_error(exc):
                raise
            return cls(workspace_db_path())

    def create_session(
        self,
        *,
        workspace_root: Path,
        provider: str | None = None,
        model: str | None = None,
        title: str | None = None,
    ) -> SessionInfo:
        now = utc_now()
        session_id = new_session_id()
        title = clean_title(title) or "New session"
        workspace_root = workspace_root.expanduser().resolve()

        with self._connect() as conn:
            project = ensure_project(conn, workspace_root, now)
            workspace = ensure_workspace(conn, project.id, workspace_root, workspace_root, now)
            conn.execute(
                """
                insert into sessions (
                    id, project_id, workspace_id, title, workspace_root, cwd,
                    created_at, updated_at, provider, model, agent, permission_json,
                    cost_usd, tokens_input, tokens_output, tokens_reasoning,
                    tokens_cache_read, tokens_cache_write,
                    summary, active_root, focus_path, last_prompt, state_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0,
                        null, null, null, null, null)
                """,
                (
                    session_id,
                    project.id,
                    workspace.id,
                    title,
                    str(workspace_root),
                    str(workspace_root),
                    now,
                    now,
                    provider,
                    model,
                    "nym",
                    json.dumps(DEFAULT_PERMISSION, ensure_ascii=False),
                ),
            )

        return self.get_session(session_id)

    def get_session(self, session_id: str) -> SessionInfo:
        with self._connect() as conn:
            row = conn.execute(
                """
                select id, project_id, workspace_id, title, workspace_root, cwd,
                       created_at, updated_at, provider, model, agent, permission_json,
                       cost_usd, tokens_input, tokens_output, tokens_reasoning,
                       tokens_cache_read, tokens_cache_write,
                       summary, active_root, focus_path, last_prompt, state_json
                from sessions
                where id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Session not found: {session_id}")

        return session_from_row(row)

    def list_sessions(self, *, limit: int = 50) -> list[SessionInfo]:
        limit = max(1, min(limit, 500))

        with self._connect() as conn:
            rows = conn.execute(
                """
                select id, project_id, workspace_id, title, workspace_root, cwd,
                       created_at, updated_at, provider, model, agent, permission_json,
                       cost_usd, tokens_input, tokens_output, tokens_reasoning,
                       tokens_cache_read, tokens_cache_write,
                       summary, active_root, focus_path, last_prompt, state_json
                from sessions
                order by updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()

        return [session_from_row(row) for row in rows]

    def update_llm_config(
        self,
        session_id: str,
        *,
        provider: str | None,
        model: str | None,
    ) -> None:
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                update sessions
                set provider = ?,
                    model = ?,
                    updated_at = ?
                where id = ?
                """,
                (provider, model, now, session_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Session not found: {session_id}")

    def resolve_session_id(self, prefix: str) -> str:
        prefix = prefix.strip()
        if not prefix:
            raise ValueError("Session id prefix cannot be empty.")

        with self._connect() as conn:
            rows = conn.execute(
                "select id from sessions where id like ? order by updated_at desc",
                (f"{prefix}%",),
            ).fetchall()

        if not rows:
            raise KeyError(f"No session matches prefix: {prefix}")
        if len(rows) > 1:
            matches = ", ".join(row["id"] for row in rows[:10])
            raise ValueError(f"Session prefix is ambiguous: {prefix}. Matches: {matches}")

        return str(rows[0]["id"])

    def add_message(self, session_id: str, role: str, content: str) -> MessageInfo:
        if role not in {"user", "assistant", "tool", "system"}:
            raise ValueError(f"Unsupported message role: {role}")

        now = utc_now()
        with self._connect() as conn:
            seq = next_message_seq(conn, session_id)
            cursor = conn.execute(
                """
                insert into messages (session_id, seq, role, content, created_at)
                values (?, ?, ?, ?, ?)
                """,
                (session_id, seq, role, content, now),
            )
            conn.execute(
                "update sessions set updated_at = ? where id = ?",
                (now, session_id),
            )
            message_id = int(cursor.lastrowid)

        return MessageInfo(
            id=message_id,
            session_id=session_id,
            seq=seq,
            role=role,
            content=content,
            created_at=now,
        )

    def list_messages(self, session_id: str, *, limit: int | None = 20) -> list[MessageInfo]:
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    """
                    select id, session_id, seq, role, content, created_at
                    from messages
                    where session_id = ?
                    order by seq asc
                    """,
                    (session_id,),
                ).fetchall()
            else:
                limit = max(1, min(limit, 500))
                rows = conn.execute(
                    """
                    select id, session_id, seq, role, content, created_at
                    from messages
                    where session_id = ?
                    order by seq asc
                    limit ?
                    """,
                    (session_id, limit),
                ).fetchall()

        return [message_from_row(row) for row in rows]

    def update_last_prompt(self, session_id: str, prompt: str) -> None:
        title = clean_title(prompt)
        now = utc_now()

        with self._connect() as conn:
            row = conn.execute(
                "select title from sessions where id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")

            current_title = str(row["title"])
            next_title = title if current_title == "New session" and title else current_title
            conn.execute(
                """
                update sessions
                set title = ?, last_prompt = ?, updated_at = ?
                where id = ?
                """,
                (next_title, prompt, now, session_id),
            )

    def save_agent_state(self, session_id: str, state: dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                update sessions
                set state_json = ?,
                    active_root = ?,
                    focus_path = ?,
                    updated_at = ?
                where id = ?
                """,
                (
                    json.dumps(state, ensure_ascii=False),
                    state.get("active_root"),
                    state.get("focus_path"),
                    now,
                    session_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Session not found: {session_id}")

    def add_usage(
        self,
        session_id: str,
        *,
        tokens: TokenUsage,
        cost_usd: float = 0.0,
    ) -> None:
        if (
            tokens.input == 0
            and tokens.output == 0
            and tokens.reasoning == 0
            and tokens.cache_read == 0
            and tokens.cache_write == 0
            and cost_usd == 0
        ):
            return

        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                update sessions
                set tokens_input = tokens_input + ?,
                    tokens_output = tokens_output + ?,
                    tokens_reasoning = tokens_reasoning + ?,
                    tokens_cache_read = tokens_cache_read + ?,
                    tokens_cache_write = tokens_cache_write + ?,
                    cost_usd = cost_usd + ?,
                    updated_at = ?
                where id = ?
                """,
                (
                    tokens.input,
                    tokens.output,
                    tokens.reasoning,
                    tokens.cache_read,
                    tokens.cache_write,
                    cost_usd,
                    now,
                    session_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Session not found: {session_id}")

    def add_event(
        self,
        session_id: str,
        *,
        event_type: str,
        summary: str,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        path: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> EventInfo:
        event_type = event_type.strip()
        summary = " ".join(summary.split())
        if not event_type:
            raise ValueError("Event type cannot be empty.")
        if not summary.strip():
            raise ValueError("Event summary cannot be empty.")

        now = utc_now()
        args_json = json.dumps(args, ensure_ascii=False) if args is not None else None
        data_json = json.dumps(data, ensure_ascii=False) if data is not None else None

        with self._connect() as conn:
            seq = next_event_seq(conn, session_id)
            cursor = conn.execute(
                """
                insert into events (
                    session_id, seq, event_type, tool, path, summary,
                    args_json, data_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    seq,
                    event_type,
                    tool,
                    path,
                    summary,
                    args_json,
                    data_json,
                    now,
                ),
            )
            conn.execute(
                "update sessions set updated_at = ? where id = ?",
                (now, session_id),
            )
            event_id = int(cursor.lastrowid)

        return EventInfo(
            id=event_id,
            session_id=session_id,
            seq=seq,
            event_type=event_type,
            tool=tool,
            args=args,
            path=path,
            summary=summary,
            data=data,
            created_at=now,
        )

    def list_events(self, session_id: str, *, limit: int = 20) -> list[EventInfo]:
        limit = max(1, min(limit, 500))

        with self._connect() as conn:
            rows = conn.execute(
                """
                select id, session_id, seq, event_type, tool, args_json,
                       path, summary, data_json, created_at
                from events
                where session_id = ?
                order by seq desc
                limit ?
                """,
                (session_id, limit),
            ).fetchall()

        return [event_from_row(row) for row in rows]

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            migrate_schema(conn)
            backfill_session_context(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = on")
        conn.execute("pragma journal_mode = wal")
        conn.execute("pragma busy_timeout = 5000")
        return conn


def default_db_path() -> Path:
    return data_home() / "nym" / "sessions.sqlite3"


def workspace_db_path() -> Path:
    return Path.cwd() / ".nym-session.sqlite3"


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


def migrate_schema(conn: sqlite3.Connection) -> None:
    session_columns = table_columns(conn, "sessions")
    desired_columns = {
        "project_id": "text",
        "workspace_id": "text",
        "cwd": "text",
        "provider": "text",
        "agent": "text",
        "permission_json": "text",
        "cost_usd": "real not null default 0",
        "tokens_input": "integer not null default 0",
        "tokens_output": "integer not null default 0",
        "tokens_reasoning": "integer not null default 0",
        "tokens_cache_read": "integer not null default 0",
        "tokens_cache_write": "integer not null default 0",
    }

    for column, definition in desired_columns.items():
        if column not in session_columns:
            conn.execute(f"alter table sessions add column {column} {definition}")

    conn.execute(
        """
        create index if not exists idx_sessions_project_updated_at
            on sessions(project_id, updated_at desc)
        """
    )


def backfill_session_context(conn: sqlite3.Connection) -> None:
    now = utc_now()
    rows = conn.execute(
        """
        select id, workspace_root, cwd, project_id, workspace_id, permission_json
        from sessions
        """
    ).fetchall()

    for row in rows:
        workspace_root = Path(str(row["workspace_root"])).expanduser().resolve()
        cwd_value = row["cwd"]
        cwd = Path(cwd_value).expanduser().resolve() if cwd_value else workspace_root
        project = ensure_project(conn, workspace_root, now)
        workspace = ensure_workspace(conn, project.id, workspace_root, cwd, now)
        permission_json = row["permission_json"]
        if not permission_json:
            permission_json = json.dumps(DEFAULT_PERMISSION, ensure_ascii=False)

        conn.execute(
            """
            update sessions
            set project_id = coalesce(project_id, ?),
                workspace_id = coalesce(workspace_id, ?),
                cwd = coalesce(cwd, ?),
                agent = coalesce(agent, ?),
                permission_json = coalesce(permission_json, ?)
            where id = ?
            """,
            (
                project.id,
                workspace.id,
                str(cwd),
                "nym",
                permission_json,
                row["id"],
            ),
        )


def ensure_project(conn: sqlite3.Connection, root: Path, now: str) -> ProjectInfo:
    root_text = str(root.expanduser().resolve())
    row = conn.execute(
        "select id, root, title, created_at, updated_at from projects where root = ?",
        (root_text,),
    ).fetchone()
    if row is None:
        project_id = new_session_id()
        title = root.name or root_text
        conn.execute(
            """
            insert into projects (id, root, title, created_at, updated_at)
            values (?, ?, ?, ?, ?)
            """,
            (project_id, root_text, title, now, now),
        )
        row = conn.execute(
            "select id, root, title, created_at, updated_at from projects where id = ?",
            (project_id,),
        ).fetchone()

    return project_from_row(row)


def ensure_workspace(
    conn: sqlite3.Connection,
    project_id: str,
    root: Path,
    cwd: Path,
    now: str,
) -> WorkspaceInfo:
    root_text = str(root.expanduser().resolve())
    cwd_text = str(cwd.expanduser().resolve())
    row = conn.execute(
        """
        select id, project_id, root, cwd, created_at, updated_at
        from workspaces
        where project_id = ? and root = ? and cwd = ?
        """,
        (project_id, root_text, cwd_text),
    ).fetchone()
    if row is None:
        workspace_id = new_session_id()
        conn.execute(
            """
            insert into workspaces (id, project_id, root, cwd, created_at, updated_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (workspace_id, project_id, root_text, cwd_text, now, now),
        )
        row = conn.execute(
            """
            select id, project_id, root, cwd, created_at, updated_at
            from workspaces
            where id = ?
            """,
            (workspace_id,),
        ).fetchone()

    return workspace_from_row(row)


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"pragma table_info({table})")}


def project_from_row(row: sqlite3.Row) -> ProjectInfo:
    return ProjectInfo(
        id=str(row["id"]),
        root=str(row["root"]),
        title=str(row["title"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def workspace_from_row(row: sqlite3.Row) -> WorkspaceInfo:
    return WorkspaceInfo(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        root=str(row["root"]),
        cwd=str(row["cwd"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def session_from_row(row: sqlite3.Row) -> SessionInfo:
    state_json = row["state_json"]
    state = json.loads(state_json) if isinstance(state_json, str) and state_json else None
    permission_json = row["permission_json"]
    permission = (
        json.loads(permission_json)
        if isinstance(permission_json, str) and permission_json
        else None
    )

    return SessionInfo(
        id=str(row["id"]),
        project_id=row["project_id"],
        workspace_id=row["workspace_id"],
        title=str(row["title"]),
        workspace_root=str(row["workspace_root"]),
        cwd=row["cwd"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        provider=row["provider"],
        model=row["model"],
        agent=row["agent"],
        permission=permission if isinstance(permission, dict) else None,
        cost_usd=float(row["cost_usd"] or 0),
        tokens=TokenUsage(
            input=int(row["tokens_input"] or 0),
            output=int(row["tokens_output"] or 0),
            reasoning=int(row["tokens_reasoning"] or 0),
            cache_read=int(row["tokens_cache_read"] or 0),
            cache_write=int(row["tokens_cache_write"] or 0),
        ),
        summary=row["summary"],
        active_root=row["active_root"],
        focus_path=row["focus_path"],
        last_prompt=row["last_prompt"],
        state=state,
    )


def message_from_row(row: sqlite3.Row) -> MessageInfo:
    return MessageInfo(
        id=int(row["id"]),
        session_id=str(row["session_id"]),
        seq=int(row["seq"]),
        role=str(row["role"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
    )


def event_from_row(row: sqlite3.Row) -> EventInfo:
    args_json = row["args_json"]
    data_json = row["data_json"]
    args = json.loads(args_json) if isinstance(args_json, str) and args_json else None
    data = json.loads(data_json) if isinstance(data_json, str) and data_json else None

    return EventInfo(
        id=int(row["id"]),
        session_id=str(row["session_id"]),
        seq=int(row["seq"]),
        event_type=str(row["event_type"]),
        tool=row["tool"],
        args=args if isinstance(args, dict) else None,
        path=row["path"],
        summary=str(row["summary"]),
        data=data if isinstance(data, dict) else None,
        created_at=str(row["created_at"]),
    )


def next_message_seq(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "select coalesce(max(seq), 0) + 1 as next_seq from messages where session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return 1
    return int(row["next_seq"])


def next_event_seq(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        """
        select coalesce(max(seq), 0) + 1 as next_seq
        from events
        where session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return 1
    return int(row["next_seq"])


def _env_path(name: str) -> Path | None:
    import os

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
