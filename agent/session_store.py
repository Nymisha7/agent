from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_SESSION_PRUNE_AFTER_DAYS = 30
DEFAULT_SESSION_MAX_ENTRIES = 500
DEFAULT_SESSION_WRITE_LOCK_ACQUIRE_TIMEOUT_MS = 60_000
STRICT_ENTRY_MAINTENANCE_MAX_ENTRIES = 49
MIN_BATCHED_ENTRY_MAINTENANCE_SLACK = 25
BATCHED_ENTRY_MAINTENANCE_SLACK_RATIO = 0.1

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

create table if not exists session_routes (
    route_key text primary key,
    session_id text not null,
    agent_id text not null,
    scope text not null,
    channel text not null,
    account_id text not null,
    peer_kind text,
    peer_id text,
    sender_id text,
    guild_id text,
    team_id text,
    created_at text not null,
    updated_at text not null,
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

create index if not exists idx_session_routes_session_id
    on session_routes(session_id);

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


class SessionStore:
    def __init__(
        self,
        db_path: Path,
        *,
        write_lock_timeout_ms: int = DEFAULT_SESSION_WRITE_LOCK_ACQUIRE_TIMEOUT_MS,
    ) -> None:
        self.db_path = db_path.expanduser()
        self.write_lock_timeout_ms = _maintenance_positive_int(
            write_lock_timeout_ms,
            "write_lock_timeout_ms",
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def default(cls) -> SessionStore:
        explicit_path = _env_path("AGENT_SESSION_DB")
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
        agent_id: str = "agent",
    ) -> SessionInfo:
        now = utc_now()
        session_id = new_session_id()
        title = clean_title(title) or "New session"
        workspace_root = workspace_root.expanduser().resolve()

        with self._connect() as conn:
            project = ensure_project(conn, workspace_root, now)
            workspace = ensure_workspace(conn, project.id, workspace_root, workspace_root, now)
            insert_session(
                conn,
                session_id=session_id,
                project_id=project.id,
                workspace_id=workspace.id,
                title=title,
                workspace_root=workspace_root,
                now=now,
                provider=provider,
                model=model,
                agent_id=agent_id,
                )

        self.apply_maintenance(active_session_id=session_id)
        return self.get_session(session_id)

    def get_route(self, route_key: str) -> SessionRouteInfo:
        with self._connect() as conn:
            row = conn.execute(
                """
                select route_key, session_id, agent_id, scope, channel, account_id,
                       peer_kind, peer_id, sender_id, guild_id, team_id,
                       created_at, updated_at
                from session_routes
                where route_key = ?
                """,
                (route_key,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Session route not found: {route_key}")
        return session_route_from_row(row)

    def list_routes_for_session(self, session_id: str) -> list[SessionRouteInfo]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select route_key, session_id, agent_id, scope, channel, account_id,
                       peer_kind, peer_id, sender_id, guild_id, team_id,
                       created_at, updated_at
                from session_routes
                where session_id = ?
                order by created_at
                """,
                (session_id,),
            ).fetchall()
        return [session_route_from_row(row) for row in rows]

    def list_routes(self, *, limit: int = 100) -> list[SessionRouteInfo]:
        limit = max(1, min(limit, 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                select route_key, session_id, agent_id, scope, channel, account_id,
                       peer_kind, peer_id, sender_id, guild_id, team_id,
                       created_at, updated_at
                from session_routes
                order by updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [session_route_from_row(row) for row in rows]

    def get_or_create_routed_session(
        self,
        *,
        route_key: str,
        workspace_root: Path,
        agent_id: str,
        scope: str,
        channel: str,
        account_id: str,
        peer_kind: str | None = None,
        peer_id: str | None = None,
        sender_id: str | None = None,
        guild_id: str | None = None,
        team_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        title: str | None = None,
    ) -> tuple[SessionInfo, bool]:
        now = utc_now()
        workspace_root = workspace_root.expanduser().resolve()
        created = False

        with self._connect() as conn:
            conn.execute("begin immediate")
            existing = conn.execute(
                "select session_id from session_routes where route_key = ?",
                (route_key,),
            ).fetchone()
            if existing is not None:
                session_id = str(existing["session_id"])
                conn.execute(
                    "update session_routes set updated_at = ? where route_key = ?",
                    (now, route_key),
                )
            else:
                session_id = new_session_id()
                project = ensure_project(conn, workspace_root, now)
                workspace = ensure_workspace(
                    conn,
                    project.id,
                    workspace_root,
                    workspace_root,
                    now,
                )
                insert_session(
                    conn,
                    session_id=session_id,
                    project_id=project.id,
                    workspace_id=workspace.id,
                    title=clean_title(title) or "New session",
                    workspace_root=workspace_root,
                    now=now,
                    provider=provider,
                    model=model,
                    agent_id=agent_id,
                )
                conn.execute(
                    """
                    insert into session_routes (
                        route_key, session_id, agent_id, scope, channel, account_id,
                        peer_kind, peer_id, sender_id, guild_id, team_id,
                        created_at, updated_at
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        route_key,
                        session_id,
                        agent_id,
                        scope,
                        channel,
                        account_id,
                        peer_kind,
                        peer_id,
                        sender_id,
                        guild_id,
                        team_id,
                        now,
                        now,
                    ),
                )
                created = True

        self.apply_maintenance(active_session_id=session_id)
        return self.get_session(session_id), created

    def apply_maintenance(
        self,
        *,
        max_entries: int = DEFAULT_SESSION_MAX_ENTRIES,
        prune_after_days: int = DEFAULT_SESSION_PRUNE_AFTER_DAYS,
        active_session_id: str | None = None,
        force: bool = False,
        mode: str = "enforce",
    ) -> SessionMaintenanceReport:
        max_entries = _maintenance_positive_int(max_entries, "max_entries")
        prune_after_days = _maintenance_positive_int(prune_after_days, "prune_after_days")
        mode = mode.strip().casefold()
        if mode not in {"enforce", "warn"}:
            raise ValueError("Session maintenance mode must be 'enforce' or 'warn'.")

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=prune_after_days)
        ).isoformat()

        with self._connect() as conn:
            conn.execute("begin immediate")
            before_count = _session_count(conn)
            if not force and not _should_run_session_entry_maintenance(
                entry_count=before_count,
                max_entries=max_entries,
            ):
                return SessionMaintenanceReport(
                    mode=mode,
                    before_count=before_count,
                    after_count=before_count,
                    pruned=0,
                    capped=0,
                    applied=False,
                )

            preserve_ids = _maintenance_preserve_session_ids(
                conn,
                active_session_id=active_session_id,
            )
            stale_ids = _stale_session_ids(
                conn,
                cutoff=cutoff,
                preserve_ids=preserve_ids,
            )
            after_stale_count = before_count - len(stale_ids)
            capped_ids = _capped_session_ids(
                conn,
                max_entries=max_entries,
                preserve_ids=preserve_ids,
                excluded_ids=set(stale_ids),
            )

            if mode == "enforce":
                _delete_session_ids(conn, [*stale_ids, *capped_ids])
                after_count = _session_count(conn)
            else:
                after_count = after_stale_count - len(capped_ids)

        return SessionMaintenanceReport(
            mode=mode,
            before_count=before_count,
            after_count=after_count,
            pruned=len(stale_ids),
            capped=len(capped_ids),
            applied=bool(stale_ids or capped_ids),
        )

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

    def list_sessions(
        self,
        *,
        limit: int | None = 50,
        agent_id: str | None = None,
        updated_after: str | None = None,
    ) -> list[SessionInfo]:
        if limit is not None:
            limit = max(1, min(limit, 500))

        with self._connect() as conn:
            filters, values = _session_list_filters(
                agent_id=agent_id,
                updated_after=updated_after,
            )
            limit_clause = "" if limit is None else "limit ?"
            if limit is not None:
                values.append(limit)
            rows = conn.execute(
                f"""
                select id, project_id, workspace_id, title, workspace_root, cwd,
                       created_at, updated_at, provider, model, agent, permission_json,
                       cost_usd, tokens_input, tokens_output, tokens_reasoning,
                       tokens_cache_read, tokens_cache_write,
                       summary, active_root, focus_path, last_prompt, state_json
                from sessions
                {filters}
                order by updated_at desc
                {limit_clause}
                """,
                tuple(values),
            ).fetchall()

        return [session_from_row(row) for row in rows]

    def count_sessions(
        self,
        *,
        agent_id: str | None = None,
        updated_after: str | None = None,
    ) -> int:
        with self._connect() as conn:
            filters, values = _session_list_filters(
                agent_id=agent_id,
                updated_after=updated_after,
            )
            row = conn.execute(
                f"select count(*) as count from sessions {filters}",
                tuple(values),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

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

    def patch_session_metadata(
        self,
        session_id: str,
        *,
        title: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        state_patch: dict[str, Any] | None = None,
    ) -> SessionInfo:
        now = utc_now()
        with self._connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                """
                select title, provider, model, state_json, active_root, focus_path
                from sessions
                where id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")

            next_title = str(row["title"])
            if title is not None:
                next_title = clean_title(title) or ""
                if not next_title:
                    raise ValueError("Session title cannot be empty.")

            state_json = row["state_json"]
            state = (
                json.loads(state_json)
                if isinstance(state_json, str) and state_json
                else None
            )
            if not isinstance(state, dict):
                state = {}
            if state_patch:
                state.update(state_patch)

            next_state_json = json.dumps(state, ensure_ascii=False) if state else None
            cursor = conn.execute(
                """
                update sessions
                set title = ?,
                    provider = ?,
                    model = ?,
                    state_json = ?,
                    active_root = ?,
                    focus_path = ?,
                    updated_at = ?
                where id = ?
                """,
                (
                    next_title,
                    provider if provider is not None else row["provider"],
                    model if model is not None else row["model"],
                    next_state_json,
                    state.get("active_root", row["active_root"]),
                    state.get("focus_path", row["focus_path"]),
                    now,
                    session_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Session not found: {session_id}")

        return self.get_session(session_id)

    def reset_routed_session(
        self,
        route_key: str,
        *,
        reason: str = "reset",
    ) -> tuple[SessionInfo, SessionInfo]:
        route_key = route_key.strip()
        if not route_key:
            raise ValueError("Session route key cannot be empty.")
        if reason not in {"new", "reset"}:
            raise ValueError("Session reset reason must be 'new' or 'reset'.")

        now = utc_now()
        fresh_session_id = new_session_id()
        with self._connect() as conn:
            conn.execute("begin immediate")
            route = conn.execute(
                "select session_id from session_routes where route_key = ?",
                (route_key,),
            ).fetchone()
            if route is None:
                raise KeyError(f"Session route not found: {route_key}")
            old_session_id = str(route["session_id"])
            old = conn.execute(
                """
                select project_id, workspace_id, title, workspace_root, cwd,
                       provider, model, agent, permission_json, state_json
                from sessions
                where id = ?
                """,
                (old_session_id,),
            ).fetchone()
            if old is None:
                raise KeyError(f"Session not found: {old_session_id}")

            preserved_state: dict[str, Any] = {}
            state_json = old["state_json"]
            if isinstance(state_json, str) and state_json:
                state = json.loads(state_json)
                if isinstance(state, dict) and state.get("reasoning_effort") is not None:
                    preserved_state["reasoning_effort"] = state["reasoning_effort"]

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
                        null, null, null, null, ?)
                """,
                (
                    fresh_session_id,
                    old["project_id"],
                    old["workspace_id"],
                    str(old["title"]),
                    str(old["workspace_root"]),
                    old["cwd"],
                    now,
                    now,
                    old["provider"],
                    old["model"],
                    old["agent"],
                    old["permission_json"],
                    json.dumps(preserved_state, ensure_ascii=False) if preserved_state else None,
                ),
            )
            conn.execute(
                "update session_routes set session_id = ?, updated_at = ? where route_key = ?",
                (fresh_session_id, now, route_key),
            )

        self.apply_maintenance(active_session_id=fresh_session_id)
        return self.get_session(old_session_id), self.get_session(fresh_session_id)

    def delete_routed_session(self, route_key: str) -> dict[str, int | str]:
        route_key = route_key.strip()
        if not route_key:
            raise ValueError("Session route key cannot be empty.")

        with self._connect() as conn:
            conn.execute("begin immediate")
            route = conn.execute(
                "select session_id from session_routes where route_key = ?",
                (route_key,),
            ).fetchone()
            if route is None:
                raise KeyError(f"Session route not found: {route_key}")
            session_id = str(route["session_id"])
            session = conn.execute(
                "select id from sessions where id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            message_count = _count_rows(conn, "messages", session_id)
            event_count = _count_rows(conn, "events", session_id)
            route_count = _count_rows(conn, "session_routes", session_id)
            cursor = conn.execute("delete from sessions where id = ?", (session_id,))
            if cursor.rowcount == 0:
                raise KeyError(f"Session not found: {session_id}")

        return {
            "session_id": session_id,
            "messages_deleted": message_count,
            "events_deleted": event_count,
            "routes_deleted": route_count,
        }

    def compact_routed_session(
        self,
        route_key: str,
        *,
        max_messages: int,
    ) -> dict[str, int | str | bool | None]:
        route_key = route_key.strip()
        if not route_key:
            raise ValueError("Session route key cannot be empty.")
        if isinstance(max_messages, bool) or not isinstance(max_messages, int) or max_messages < 1:
            raise ValueError("max_messages must be a positive integer.")

        now = utc_now()
        with self._connect() as conn:
            conn.execute("begin immediate")
            route = conn.execute(
                "select session_id from session_routes where route_key = ?",
                (route_key,),
            ).fetchone()
            if route is None:
                raise KeyError(f"Session route not found: {route_key}")
            session_id = str(route["session_id"])
            session = conn.execute(
                "select id from sessions where id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            rows = conn.execute(
                """
                select id, seq, role, content, created_at
                from messages
                where session_id = ?
                order by seq asc
                """,
                (session_id,),
            ).fetchall()
            lines_before = len(rows)
            if lines_before <= max_messages:
                return {
                    "session_id": session_id,
                    "compacted": False,
                    "lines_before": lines_before,
                    "lines_after": lines_before,
                    "kept": lines_before,
                    "pruned": 0,
                    "archived_event_id": None,
                }

            pruned_rows = rows[: lines_before - max_messages]
            kept_rows = rows[lines_before - max_messages :]
            archive_payload = {
                "kind": "sessions.compact.maxLines.archive",
                "route_key": route_key,
                "session_id": session_id,
                "max_messages": max_messages,
                "lines_before": lines_before,
                "lines_after": len(kept_rows),
                "first_kept_seq": int(kept_rows[0]["seq"]) if kept_rows else None,
                "messages": [
                    {
                        "id": int(row["id"]),
                        "seq": int(row["seq"]),
                        "role": str(row["role"]),
                        "content": str(row["content"]),
                        "created_at": str(row["created_at"]),
                    }
                    for row in pruned_rows
                ],
            }
            event_seq = next_event_seq(conn, session_id)
            event_summary = (
                f"Compacted transcript to last {max_messages} message(s); "
                f"archived {len(pruned_rows)} pruned message(s)"
            )
            event_cursor = conn.execute(
                """
                insert into events (
                    session_id, seq, event_type, tool, path, summary,
                    args_json, data_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    event_seq,
                    "session_compacted",
                    None,
                    None,
                    event_summary,
                    json.dumps({"max_messages": max_messages}, ensure_ascii=False),
                    json.dumps(archive_payload, ensure_ascii=False),
                    now,
                ),
            )
            conn.executemany(
                "delete from messages where id = ?",
                [(int(row["id"]),) for row in pruned_rows],
            )
            conn.execute(
                "update sessions set updated_at = ? where id = ?",
                (now, session_id),
            )

        return {
            "session_id": session_id,
            "compacted": True,
            "lines_before": lines_before,
            "lines_after": len(kept_rows),
            "kept": len(kept_rows),
            "pruned": len(pruned_rows),
            "archived_event_id": int(event_cursor.lastrowid),
        }

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

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        expected_route_key: str | None = None,
    ) -> MessageInfo:
        return self.add_messages(
            session_id,
            [(role, content)],
            expected_route_key=expected_route_key,
        )[0]

    def add_messages(
        self,
        session_id: str,
        messages: list[tuple[str, str]] | tuple[tuple[str, str], ...],
        *,
        last_prompt: str | None = None,
        expected_route_key: str | None = None,
    ) -> list[MessageInfo]:
        if not messages:
            return []
        for role, _content in messages:
            if role not in {"user", "assistant", "tool", "system"}:
                raise ValueError(f"Unsupported message role: {role}")

        now = utc_now()
        inserted: list[MessageInfo] = []
        with self._connect() as conn:
            conn.execute("begin immediate")
            if expected_route_key:
                route = conn.execute(
                    "select session_id from session_routes where route_key = ?",
                    (expected_route_key,),
                ).fetchone()
                if route is None or str(route["session_id"]) != session_id:
                    current = str(route["session_id"]) if route is not None else "missing"
                    raise RuntimeError(
                        "Session route rebound: "
                        f"{expected_route_key} now points to {current}, not {session_id}"
                    )
            seq = next_message_seq(conn, session_id)
            for offset, (role, content) in enumerate(messages):
                message_seq = seq + offset
                cursor = conn.execute(
                    """
                    insert into messages (session_id, seq, role, content, created_at)
                    values (?, ?, ?, ?, ?)
                    """,
                    (session_id, message_seq, role, content, now),
                )
                inserted.append(MessageInfo(
                    id=int(cursor.lastrowid),
                    session_id=session_id,
                    seq=message_seq,
                    role=role,
                    content=content,
                    created_at=now,
                ))
            if last_prompt is not None:
                row = conn.execute(
                    "select title from sessions where id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Session not found: {session_id}")
                current_title = str(row["title"])
                title = clean_title(last_prompt)
                next_title = title if current_title == "New session" and title else current_title
                cursor = conn.execute(
                    """
                    update sessions
                    set title = ?, last_prompt = ?, updated_at = ?
                    where id = ?
                    """,
                    (next_title, last_prompt, now, session_id),
                )
            else:
                cursor = conn.execute(
                    "update sessions set updated_at = ? where id = ?",
                    (now, session_id),
                )
            if cursor.rowcount == 0:
                raise KeyError(f"Session not found: {session_id}")

        return inserted

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
                    from (
                        select id, session_id, seq, role, content, created_at
                        from messages
                        where session_id = ?
                        order by seq desc
                        limit ?
                    )
                    order by seq asc
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
            conn.execute("begin immediate")
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

    def count_session_compaction_checkpoints(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                select count(*) as count
                from events
                where session_id = ? and event_type = 'session_compacted'
                """,
                (session_id,),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def list_session_compaction_checkpoints(
        self,
        session_id: str,
        *,
        session_key: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        session_key = session_key.strip()
        if not session_key:
            raise ValueError("Session key cannot be empty.")

        with self._connect() as conn:
            session = conn.execute(
                "select id from sessions where id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            rows = conn.execute(
                """
                select id, session_id, seq, event_type, tool, args_json,
                       path, summary, data_json, created_at
                from events
                where session_id = ? and event_type = 'session_compacted'
                order by created_at desc, id desc
                limit ?
                """,
                (session_id, limit),
            ).fetchall()

        return [
            _compaction_checkpoint_from_event(row, session_key=session_key)
            for row in rows
        ]

    def get_session_compaction_checkpoint(
        self,
        session_id: str,
        *,
        session_key: str,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        checkpoint_id = checkpoint_id.strip()
        if not checkpoint_id:
            raise ValueError("checkpoint_id cannot be empty.")
        event_id = _checkpoint_event_id(checkpoint_id)

        with self._connect() as conn:
            session = conn.execute(
                "select id from sessions where id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            row = conn.execute(
                """
                select id, session_id, seq, event_type, tool, args_json,
                       path, summary, data_json, created_at
                from events
                where id = ? and session_id = ? and event_type = 'session_compacted'
                """,
                (event_id, session_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Compaction checkpoint not found: {checkpoint_id}")
        return _compaction_checkpoint_from_event(row, session_key=session_key)

    def branch_routed_session_from_compaction_checkpoint(
        self,
        route_key: str,
        *,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        route_key = route_key.strip()
        if not route_key:
            raise ValueError("Session route key cannot be empty.")
        checkpoint_id = checkpoint_id.strip()
        if not checkpoint_id:
            raise ValueError("checkpoint_id cannot be empty.")
        event_id = _checkpoint_event_id(checkpoint_id)
        now = utc_now()

        with self._connect() as conn:
            conn.execute("begin immediate")
            route = conn.execute(
                """
                select route_key, session_id, agent_id, scope, channel, account_id,
                       peer_kind, peer_id, sender_id, guild_id, team_id
                from session_routes
                where route_key = ?
                """,
                (route_key,),
            ).fetchone()
            if route is None:
                raise KeyError(f"Session route not found: {route_key}")
            source_session_id = str(route["session_id"])
            source = conn.execute(
                """
                select project_id, workspace_id, title, workspace_root, cwd,
                       provider, model, agent, permission_json, state_json
                from sessions
                where id = ?
                """,
                (source_session_id,),
            ).fetchone()
            if source is None:
                raise KeyError(f"Session not found: {source_session_id}")
            checkpoint_row = conn.execute(
                """
                select id, session_id, seq, event_type, tool, args_json,
                       path, summary, data_json, created_at
                from events
                where id = ? and session_id = ? and event_type = 'session_compacted'
                """,
                (event_id, source_session_id),
            ).fetchone()
            if checkpoint_row is None:
                raise KeyError(f"Compaction checkpoint not found: {checkpoint_id}")

            snapshot_messages = _reconstruct_compaction_snapshot_messages(
                conn,
                source_session_id,
                checkpoint_row,
            )
            branch_session_id = new_session_id()
            branch_key = f"{route_key}:checkpoint:{branch_session_id}"
            title = clean_title(f"{str(source['title'])} (checkpoint)") or "Checkpoint branch"
            state = _branch_session_state(
                source["state_json"],
                parent_session_key=route_key,
                source_session_id=source_session_id,
                checkpoint_id=checkpoint_id,
            )
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
                        null, null, null, null, ?)
                """,
                (
                    branch_session_id,
                    source["project_id"],
                    source["workspace_id"],
                    title,
                    str(source["workspace_root"]),
                    source["cwd"],
                    now,
                    now,
                    source["provider"],
                    source["model"],
                    source["agent"],
                    source["permission_json"],
                    json.dumps(state, ensure_ascii=False),
                ),
            )
            conn.execute(
                """
                insert into session_routes (
                    route_key, session_id, agent_id, scope, channel, account_id,
                    peer_kind, peer_id, sender_id, guild_id, team_id,
                    created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    branch_key,
                    branch_session_id,
                    route["agent_id"],
                    route["scope"],
                    route["channel"],
                    route["account_id"],
                    route["peer_kind"],
                    route["peer_id"],
                    route["sender_id"],
                    route["guild_id"],
                    route["team_id"],
                    now,
                    now,
                ),
            )
            conn.executemany(
                """
                insert into messages (session_id, seq, role, content, created_at)
                values (?, ?, ?, ?, ?)
                """,
                [
                    (
                        branch_session_id,
                        message["seq"],
                        message["role"],
                        message["content"],
                        message["created_at"],
                    )
                    for message in snapshot_messages
                ],
            )

        self.apply_maintenance(active_session_id=branch_session_id)
        return {
            "source_session_id": source_session_id,
            "session_id": branch_session_id,
            "key": branch_key,
            "messages_copied": len(snapshot_messages),
            "checkpoint": self.get_session_compaction_checkpoint(
                source_session_id,
                session_key=route_key,
                checkpoint_id=checkpoint_id,
            ),
        }

    def restore_routed_session_from_compaction_checkpoint(
        self,
        route_key: str,
        *,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        route_key = route_key.strip()
        if not route_key:
            raise ValueError("Session route key cannot be empty.")
        checkpoint_id = checkpoint_id.strip()
        if not checkpoint_id:
            raise ValueError("checkpoint_id cannot be empty.")
        event_id = _checkpoint_event_id(checkpoint_id)
        now = utc_now()

        with self._connect() as conn:
            conn.execute("begin immediate")
            route = conn.execute(
                "select session_id from session_routes where route_key = ?",
                (route_key,),
            ).fetchone()
            if route is None:
                raise KeyError(f"Session route not found: {route_key}")
            source_session_id = str(route["session_id"])
            source = conn.execute(
                """
                select project_id, workspace_id, title, workspace_root, cwd,
                       provider, model, agent, permission_json, state_json
                from sessions
                where id = ?
                """,
                (source_session_id,),
            ).fetchone()
            if source is None:
                raise KeyError(f"Session not found: {source_session_id}")
            checkpoint_row = conn.execute(
                """
                select id, session_id, seq, event_type, tool, args_json,
                       path, summary, data_json, created_at
                from events
                where id = ? and session_id = ? and event_type = 'session_compacted'
                """,
                (event_id, source_session_id),
            ).fetchone()
            if checkpoint_row is None:
                raise KeyError(f"Compaction checkpoint not found: {checkpoint_id}")

            snapshot_messages = _reconstruct_compaction_snapshot_messages(
                conn,
                source_session_id,
                checkpoint_row,
            )
            restored_session_id = new_session_id()
            state = _restore_session_state(
                source["state_json"],
                source_session_id=source_session_id,
                checkpoint_id=checkpoint_id,
            )
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
                        null, null, null, null, ?)
                """,
                (
                    restored_session_id,
                    source["project_id"],
                    source["workspace_id"],
                    str(source["title"]),
                    str(source["workspace_root"]),
                    source["cwd"],
                    now,
                    now,
                    source["provider"],
                    source["model"],
                    source["agent"],
                    source["permission_json"],
                    json.dumps(state, ensure_ascii=False),
                ),
            )
            conn.executemany(
                """
                insert into messages (session_id, seq, role, content, created_at)
                values (?, ?, ?, ?, ?)
                """,
                [
                    (
                        restored_session_id,
                        message["seq"],
                        message["role"],
                        message["content"],
                        message["created_at"],
                    )
                    for message in snapshot_messages
                ],
            )
            conn.execute(
                """
                update events
                set session_id = ?
                where session_id = ? and event_type = 'session_compacted'
                """,
                (restored_session_id, source_session_id),
            )
            conn.execute(
                "update session_routes set session_id = ?, updated_at = ? where route_key = ?",
                (restored_session_id, now, route_key),
            )

        self.apply_maintenance(active_session_id=restored_session_id)
        return {
            "previous_session_id": source_session_id,
            "session_id": restored_session_id,
            "key": route_key,
            "messages_restored": len(snapshot_messages),
            "checkpoint": self.get_session_compaction_checkpoint(
                restored_session_id,
                session_key=route_key,
                checkpoint_id=checkpoint_id,
            ),
        }

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
        conn.execute(f"pragma busy_timeout = {self.write_lock_timeout_ms}")
        return conn


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
                "agent",
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


def insert_session(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    project_id: str,
    workspace_id: str,
    title: str,
    workspace_root: Path,
    now: str,
    provider: str | None,
    model: str | None,
    agent_id: str,
) -> None:
    root_text = str(workspace_root.expanduser().resolve())
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
            project_id,
            workspace_id,
            title,
            root_text,
            root_text,
            now,
            now,
            provider,
            model,
            agent_id,
            json.dumps(DEFAULT_PERMISSION, ensure_ascii=False),
        ),
    )


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"pragma table_info({table})")}


def _session_list_filters(
    *,
    agent_id: str | None,
    updated_after: str | None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    values: list[Any] = []
    if agent_id is not None:
        clauses.append("coalesce(agent, ?) = ?")
        values.extend(("agent", agent_id))
    if updated_after is not None:
        clauses.append("updated_at >= ?")
        values.append(updated_after)
    if not clauses:
        return "", values
    return "where " + " and ".join(clauses), values


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


def _compaction_checkpoint_from_event(
    row: sqlite3.Row,
    *,
    session_key: str,
) -> dict[str, Any]:
    session_id = str(row["session_id"])
    data_json = row["data_json"]
    data = json.loads(data_json) if isinstance(data_json, str) and data_json else {}
    data = data if isinstance(data, dict) else {}
    messages = data.get("messages")
    pruned_messages = messages if isinstance(messages, list) else []
    last_pruned_seq = _message_seq(pruned_messages[-1]) if pruned_messages else None
    first_kept_seq = _optional_int(data.get("first_kept_seq"))
    checkpoint: dict[str, Any] = {
        "checkpointId": f"sqlite:event:{int(row['id'])}",
        "sessionKey": session_key,
        "sessionId": session_id,
        "createdAt": _iso_timestamp_ms(str(row["created_at"])),
        "reason": "manual",
        "summary": str(row["summary"]),
        "preCompaction": _drop_none({
            "sessionId": session_id,
            "entryId": str(last_pruned_seq) if last_pruned_seq is not None else None,
        }),
        "postCompaction": _drop_none({
            "sessionId": session_id,
            "entryId": str(first_kept_seq) if first_kept_seq is not None else None,
        }),
    }
    for source_key, target_key in (
        ("lines_before", "linesBefore"),
        ("lines_after", "linesAfter"),
        ("max_messages", "maxMessages"),
    ):
        value = _optional_int(data.get(source_key))
        if value is not None:
            checkpoint[target_key] = value
    if first_kept_seq is not None:
        checkpoint["firstKeptEntryId"] = str(first_kept_seq)
    checkpoint["pruned"] = len(pruned_messages)
    return checkpoint


def _reconstruct_compaction_snapshot_messages(
    conn: sqlite3.Connection,
    session_id: str,
    checkpoint_row: sqlite3.Row,
) -> list[dict[str, Any]]:
    data_json = checkpoint_row["data_json"]
    data = json.loads(data_json) if isinstance(data_json, str) and data_json else {}
    data = data if isinstance(data, dict) else {}
    first_kept_seq = _optional_int(data.get("first_kept_seq"))
    lines_after = _optional_int(data.get("lines_after"))
    if first_kept_seq is None or lines_after is None:
        raise RuntimeError("Checkpoint snapshot is not reconstructable.")
    pruned_messages = _checkpoint_archived_messages(data)
    kept_rows = conn.execute(
        """
        select seq, role, content, created_at
        from messages
        where session_id = ? and seq >= ? and seq < ?
        order by seq asc
        """,
        (session_id, first_kept_seq, first_kept_seq + lines_after),
    ).fetchall()
    if len(kept_rows) != lines_after:
        raise RuntimeError("Checkpoint snapshot tail is missing from the active transcript.")
    return [
        *pruned_messages,
        *[
            {
                "seq": int(row["seq"]),
                "role": str(row["role"]),
                "content": str(row["content"]),
                "created_at": str(row["created_at"]),
            }
            for row in kept_rows
        ],
    ]


def _checkpoint_archived_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        raise RuntimeError("Checkpoint archive does not contain message rows.")
    messages: list[dict[str, Any]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise RuntimeError("Checkpoint archive contains an invalid message row.")
        seq = _optional_int(item.get("seq"))
        role = item.get("role")
        content = item.get("content")
        created_at = item.get("created_at")
        if (
            seq is None
            or role not in {"user", "assistant", "tool", "system"}
            or not isinstance(content, str)
            or not isinstance(created_at, str)
        ):
            raise RuntimeError("Checkpoint archive contains an invalid message row.")
        messages.append({
            "seq": seq,
            "role": role,
            "content": content,
            "created_at": created_at,
        })
    return messages


def _branch_session_state(
    state_json: Any,
    *,
    parent_session_key: str,
    source_session_id: str,
    checkpoint_id: str,
) -> dict[str, Any]:
    state = json.loads(state_json) if isinstance(state_json, str) and state_json else {}
    state = state if isinstance(state, dict) else {}
    branch_state = {
        "parent_session_key": parent_session_key,
        "source_session_id": source_session_id,
        "checkpoint_id": checkpoint_id,
    }
    if state.get("reasoning_effort") is not None:
        branch_state["reasoning_effort"] = state["reasoning_effort"]
    return branch_state


def _restore_session_state(
    state_json: Any,
    *,
    source_session_id: str,
    checkpoint_id: str,
) -> dict[str, Any]:
    state = json.loads(state_json) if isinstance(state_json, str) and state_json else {}
    state = state if isinstance(state, dict) else {}
    restored_state = {
        "restored_from_session_id": source_session_id,
        "restored_checkpoint_id": checkpoint_id,
    }
    if state.get("reasoning_effort") is not None:
        restored_state["reasoning_effort"] = state["reasoning_effort"]
    if state.get("parent_session_key") is not None:
        restored_state["parent_session_key"] = state["parent_session_key"]
    return restored_state


def _message_seq(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    return _optional_int(value.get("seq"))


def _checkpoint_event_id(checkpoint_id: str) -> int:
    prefix = "sqlite:event:"
    if not checkpoint_id.startswith(prefix):
        raise KeyError(f"Compaction checkpoint not found: {checkpoint_id}")
    raw = checkpoint_id[len(prefix):]
    if not raw.isdecimal():
        raise KeyError(f"Compaction checkpoint not found: {checkpoint_id}")
    return int(raw)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _iso_timestamp_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return 0
    return int(parsed.timestamp() * 1000)


def session_route_from_row(row: sqlite3.Row) -> SessionRouteInfo:
    return SessionRouteInfo(
        route_key=str(row["route_key"]),
        session_id=str(row["session_id"]),
        agent_id=str(row["agent_id"]),
        scope=str(row["scope"]),
        channel=str(row["channel"]),
        account_id=str(row["account_id"]),
        peer_kind=row["peer_kind"],
        peer_id=row["peer_id"],
        sender_id=row["sender_id"],
        guild_id=row["guild_id"],
        team_id=row["team_id"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
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


def _session_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("select count(*) as count from sessions").fetchone()
    return int(row["count"] if row is not None else 0)


def _maintenance_high_water(max_entries: int) -> int:
    if max_entries <= STRICT_ENTRY_MAINTENANCE_MAX_ENTRIES:
        return max_entries + 1
    slack = max(
        MIN_BATCHED_ENTRY_MAINTENANCE_SLACK,
        math.ceil(max_entries * BATCHED_ENTRY_MAINTENANCE_SLACK_RATIO),
    )
    return max_entries + slack


def _should_run_session_entry_maintenance(
    *,
    entry_count: int,
    max_entries: int,
) -> bool:
    return entry_count >= _maintenance_high_water(max_entries)


def _maintenance_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _maintenance_preserve_session_ids(
    conn: sqlite3.Connection,
    *,
    active_session_id: str | None,
) -> set[str]:
    preserve = {
        str(row["session_id"])
        for row in conn.execute("select distinct session_id from session_routes")
    }
    if active_session_id:
        preserve.add(active_session_id)
    return preserve


def _stale_session_ids(
    conn: sqlite3.Connection,
    *,
    cutoff: str,
    preserve_ids: set[str],
) -> list[str]:
    rows = conn.execute(
        """
        select id
        from sessions
        where updated_at < ?
        order by updated_at asc, created_at asc, id asc
        """,
        (cutoff,),
    ).fetchall()
    return [str(row["id"]) for row in rows if str(row["id"]) not in preserve_ids]


def _capped_session_ids(
    conn: sqlite3.Connection,
    *,
    max_entries: int,
    preserve_ids: set[str],
    excluded_ids: set[str],
) -> list[str]:
    preserved_count = (
        conn.execute(
            "select count(*) as count from sessions where id in (%s)"
            % ",".join("?" for _ in preserve_ids),
            tuple(preserve_ids),
        ).fetchone()
        if preserve_ids
        else None
    )
    removable_budget = max(
        0,
        max_entries - int(preserved_count["count"] if preserved_count else 0),
    )
    rows = conn.execute(
        """
        select id
        from sessions
        order by updated_at desc, created_at desc, id desc
        """
    ).fetchall()
    removable = [
        str(row["id"])
        for row in rows
        if str(row["id"]) not in preserve_ids and str(row["id"]) not in excluded_ids
    ]
    if len(removable) <= removable_budget:
        return []
    return removable[removable_budget:]


def _delete_session_ids(conn: sqlite3.Connection, session_ids: list[str]) -> None:
    if not session_ids:
        return
    conn.executemany(
        "delete from sessions where id = ?",
        [(session_id,) for session_id in session_ids],
    )


def _count_rows(conn: sqlite3.Connection, table: str, session_id: str) -> int:
    if table == "session_routes":
        column = "session_id"
    elif table in {"messages", "events"}:
        column = "session_id"
    else:
        raise ValueError(f"Unsupported count table: {table}")
    row = conn.execute(
        f"select count(*) as count from {table} where {column} = ?",
        (session_id,),
    ).fetchone()
    return int(row["count"] if row is not None else 0)


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
