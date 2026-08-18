from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bundle import bundled_rust_binary
from .rust_tools import RustTools
from .session_store import (
    DEFAULT_SESSION_MAX_ENTRIES,
    DEFAULT_SESSION_PRUNE_AFTER_DAYS,
    DEFAULT_SESSION_WRITE_LOCK_ACQUIRE_TIMEOUT_MS,
    EventInfo,
    AttachmentInfo,
    CostUsage,
    MessageInfo,
    ProjectInfo,
    SessionInfo,
    SessionMaintenanceReport,
    SessionRouteInfo,
    TokenUsage,
    WorkspaceInfo,
    _is_unusable_default_db_error,
    _maintenance_positive_int,
    clean_title,
    default_db_path,
    workspace_db_path,
)

_RUST_CLIENTS: dict[Path, RustTools] = {}
_RUST_CLIENTS_LOCK = threading.Lock()


def _set_private_permissions(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


class SessionStore:
    """Synchronous compatibility facade over the Rust SQLx session store."""

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
        parent_existed = self.db_path.parent.exists()
        self.db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not parent_existed:
            _set_private_permissions(self.db_path.parent, 0o700)
        self._rust: RustTools | None = None
        result = self._call("initialize")
        if not result:
            raise RuntimeError("Rust session store initialization returned no result.")
        _set_private_permissions(self.db_path, 0o600)

    @classmethod
    def default(cls) -> SessionStore:
        from .session_store import _env_path

        explicit_path = _env_path("AGENT_SESSION_DB")
        if explicit_path is not None:
            return cls(explicit_path)

        try:
            return cls(default_db_path())
        except (OSError, RuntimeError, sqlite3.OperationalError) as exc:
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
        workspace_root = workspace_root.expanduser().resolve()
        result = self._call(
            "create_session",
            workspace_root=str(workspace_root),
            provider=provider,
            model=model,
            title=title,
            agent_id=agent_id,
        )
        return _session_info(result)

    def get_route(self, route_key: str) -> SessionRouteInfo:
        return _route_info(self._call("get_route", route_key=route_key))

    def list_routes_for_session(self, session_id: str) -> list[SessionRouteInfo]:
        return [
            _route_info(item)
            for item in _list_result(
                self._call("list_routes_for_session", session_id=session_id)
            )
        ]

    def list_routes(self, *, limit: int = 100) -> list[SessionRouteInfo]:
        limit = max(1, min(limit, 500))
        return [
            _route_info(item)
            for item in _list_result(self._call("list_routes", limit=limit))
        ]

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
        workspace_root = workspace_root.expanduser().resolve()
        result = _mapping_result(self._call(
            "get_or_create_routed_session",
            route_key=route_key,
            workspace_root=str(workspace_root),
            agent_id=agent_id,
            scope=scope,
            channel=channel,
            account_id=account_id,
            peer_kind=peer_kind,
            peer_id=peer_id,
            sender_id=sender_id,
            guild_id=guild_id,
            team_id=team_id,
            provider=provider,
            model=model,
            title=title,
        ))
        return _session_info(result["session"]), bool(result["created"])

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
        prune_after_days = _maintenance_positive_int(
            prune_after_days,
            "prune_after_days",
        )
        mode = mode.strip().casefold()
        if mode not in {"enforce", "warn"}:
            raise ValueError("Session maintenance mode must be 'enforce' or 'warn'.")
        return _maintenance_report(self._call(
            "apply_maintenance",
            max_entries=max_entries,
            prune_after_days=prune_after_days,
            active_session_id=active_session_id,
            force=force,
            mode=mode,
        ))

    def get_session(self, session_id: str) -> SessionInfo:
        return _session_info(self._call("get_session", session_id=session_id))

    def list_sessions(
        self,
        *,
        limit: int | None = 50,
        agent_id: str | None = None,
        updated_after: str | None = None,
    ) -> list[SessionInfo]:
        if limit is not None:
            limit = max(1, min(limit, 500))
        return [
            _session_info(item)
            for item in _list_result(self._call(
                "list_sessions",
                limit=limit,
                agent_id=agent_id,
                updated_after=updated_after,
            ))
        ]

    def count_sessions(
        self,
        *,
        agent_id: str | None = None,
        updated_after: str | None = None,
    ) -> int:
        return int(self._call(
            "count_sessions",
            agent_id=agent_id,
            updated_after=updated_after,
        ))

    def update_llm_config(
        self,
        session_id: str,
        *,
        provider: str | None,
        model: str | None,
    ) -> None:
        self._call(
            "update_llm_config",
            session_id=session_id,
            provider=provider,
            model=model,
        )

    def patch_session_metadata(
        self,
        session_id: str,
        *,
        title: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        state_patch: dict[str, Any] | None = None,
    ) -> SessionInfo:
        if title is not None and not (clean_title(title) or ""):
            raise ValueError("Session title cannot be empty.")
        return _session_info(self._call(
            "patch_session_metadata",
            session_id=session_id,
            title=title,
            provider=provider,
            model=model,
            state_patch=state_patch,
        ))

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
        result = _mapping_result(self._call(
            "reset_routed_session",
            route_key=route_key,
            reason=reason,
        ))
        return (
            _session_info(result["old_session"]),
            _session_info(result["new_session"]),
        )

    def delete_routed_session(self, route_key: str) -> dict[str, int | str]:
        route_key = route_key.strip()
        if not route_key:
            raise ValueError("Session route key cannot be empty.")
        return dict(_mapping_result(
            self._call("delete_routed_session", route_key=route_key)
        ))

    def compact_routed_session(
        self,
        route_key: str,
        *,
        max_messages: int,
    ) -> dict[str, int | str | bool | None]:
        route_key = route_key.strip()
        if not route_key:
            raise ValueError("Session route key cannot be empty.")
        if (
            isinstance(max_messages, bool)
            or not isinstance(max_messages, int)
            or max_messages < 1
        ):
            raise ValueError("max_messages must be a positive integer.")
        return dict(_mapping_result(self._call(
            "compact_routed_session",
            route_key=route_key,
            max_messages=max_messages,
        )))

    def resolve_session_id(self, prefix: str) -> str:
        prefix = prefix.strip()
        if not prefix:
            raise ValueError("Session id prefix cannot be empty.")
        return str(self._call("resolve_session_id", prefix=prefix))

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        expected_route_key: str | None = None,
    ) -> MessageInfo:
        _validate_role(role)
        return _message_info(self._call(
            "add_message",
            session_id=session_id,
            role=role,
            content=content,
            expected_route_key=expected_route_key,
        ))

    def add_messages(
        self,
        session_id: str,
        messages: list[tuple[str, str]] | tuple[tuple[str, str], ...],
        *,
        last_prompt: str | None = None,
        expected_route_key: str | None = None,
    ) -> list[MessageInfo]:
        for role, _content in messages:
            _validate_role(role)
        result = self._call(
            "add_messages",
            session_id=session_id,
            messages=[
                {"role": role, "content": content}
                for role, content in messages
            ],
            last_prompt=last_prompt,
            expected_route_key=expected_route_key,
        )
        return [_message_info(item) for item in _list_result(result)]

    def add_message_with_attachments(
        self,
        session_id: str,
        role: str,
        content: str,
        attachments: list[dict[str, object]],
        *,
        last_prompt: str | None = None,
        expected_route_key: str | None = None,
    ) -> MessageInfo:
        _validate_role(role)
        result = self._call(
            "add_messages",
            session_id=session_id,
            messages=[{"role": role, "content": content, "attachments": attachments}],
            last_prompt=last_prompt,
            expected_route_key=expected_route_key,
        )
        items = _list_result(result)
        if not items:
            raise RuntimeError("Session store did not create the attachment message.")
        return _message_info(items[-1])

    def list_messages(
        self,
        session_id: str,
        *,
        limit: int | None = 20,
    ) -> list[MessageInfo]:
        if limit is not None:
            limit = max(1, min(limit, 500))
        return [
            _message_info(item)
            for item in _list_result(self._call(
                "list_messages",
                session_id=session_id,
                limit=limit,
            ))
        ]

    def update_last_prompt(self, session_id: str, prompt: str) -> None:
        self._call(
            "update_last_prompt",
            session_id=session_id,
            prompt=prompt,
        )

    def save_agent_state(self, session_id: str, state: dict[str, Any]) -> None:
        self._call(
            "save_agent_state",
            session_id=session_id,
            state=state,
        )

    def add_usage(
        self,
        session_id: str,
        *,
        tokens: TokenUsage,
        cost_usd: float = 0.0,
        costs: CostUsage | None = None,
    ) -> None:
        self._call(
            "add_usage",
            session_id=session_id,
            tokens=asdict(tokens),
            cost_usd=cost_usd,
            costs=asdict(costs or CostUsage()),
        )

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
        return _event_info(self._call(
            "add_event",
            session_id=session_id,
            event_type=event_type,
            summary=summary,
            tool=tool,
            args=args,
            path=path,
            data=data,
        ))

    def list_events(self, session_id: str, *, limit: int = 20) -> list[EventInfo]:
        limit = max(1, min(limit, 500))
        return [
            _event_info(item)
            for item in _list_result(self._call(
                "list_events",
                session_id=session_id,
                limit=limit,
            ))
        ]

    def count_session_compaction_checkpoints(self, session_id: str) -> int:
        return int(self._call(
            "count_session_compaction_checkpoints",
            session_id=session_id,
        ))

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
        return [
            _checkpoint_to_public(item)
            for item in _list_result(self._call(
                "list_session_compaction_checkpoints",
                session_id=session_id,
                session_key=session_key,
                limit=limit,
            ))
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
        return _checkpoint_to_public(self._call(
            "get_session_compaction_checkpoint",
            session_id=session_id,
            session_key=session_key,
            checkpoint_id=checkpoint_id,
        ))

    def branch_routed_session_from_compaction_checkpoint(
        self,
        route_key: str,
        *,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        route_key, checkpoint_id = _validated_route_checkpoint(
            route_key,
            checkpoint_id,
        )
        result = dict(_mapping_result(self._call(
            "branch_routed_session_from_compaction_checkpoint",
            route_key=route_key,
            checkpoint_id=checkpoint_id,
        )))
        if "checkpoint" in result:
            result["checkpoint"] = _checkpoint_to_public(result["checkpoint"])
        return result

    def restore_routed_session_from_compaction_checkpoint(
        self,
        route_key: str,
        *,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        route_key, checkpoint_id = _validated_route_checkpoint(
            route_key,
            checkpoint_id,
        )
        result = dict(_mapping_result(self._call(
            "restore_routed_session_from_compaction_checkpoint",
            route_key=route_key,
            checkpoint_id=checkpoint_id,
        )))
        if "checkpoint" in result:
            result["checkpoint"] = _checkpoint_to_public(result["checkpoint"])
        return result

    def _call(self, operation: str, **params: Any) -> Any:
        try:
            return self._rust_tools().call_session_store(
                db_path=self.db_path,
                operation=operation,
                params=params,
                write_lock_timeout_ms=self.write_lock_timeout_ms,
            )
        except RuntimeError as exc:
            _raise_compatible_store_error(exc)

    def _rust_tools(self) -> RustTools:
        rust = self._rust
        if rust is not None:
            return rust
        binary = _resolve_rust_binary(self.db_path.parent)
        with _RUST_CLIENTS_LOCK:
            rust = _RUST_CLIENTS.get(binary)
            if rust is None:
                rust = RustTools(binary)
                _RUST_CLIENTS[binary] = rust
            self._rust = rust
        return rust

    def _connect(self) -> sqlite3.Connection:
        """Compatibility hook for diagnostics and legacy test fixtures only."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = on")
        conn.execute("pragma journal_mode = wal")
        conn.execute(f"pragma busy_timeout = {self.write_lock_timeout_ms}")
        return conn


def _resolve_rust_binary(workspace_root: Path) -> Path:
    bundled = bundled_rust_binary()
    if bundled is not None:
        return bundled.resolve()

    repo_root = Path(__file__).resolve().parents[1]
    for root in (repo_root, workspace_root.resolve(), workspace_root.resolve().parent):
        for candidate in (
            root / "agent-rust" / "target" / "release" / "agent-rust",
            root / "agent-rust" / "target" / "debug" / "agent-rust",
            root / "target" / "release" / "agent-rust",
            root / "target" / "debug" / "agent-rust",
        ):
            if candidate.is_file():
                return candidate.resolve()
    return (repo_root / "agent-rust" / "target" / "debug" / "agent-rust").resolve()


def _raise_compatible_store_error(exc: RuntimeError) -> None:
    code = getattr(exc, "error_code", None)
    message = str(exc)
    if code == "not_found":
        raise KeyError(message) from exc
    if code in {"invalid_argument", "ambiguous"}:
        raise ValueError(message) from exc
    if code == "database_busy":
        raise sqlite3.OperationalError(message) from exc
    raise exc


def _mapping_result(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Rust session store returned an invalid object.")
    return value


def _list_result(value: Any) -> Sequence[Any]:
    if not isinstance(value, list):
        raise RuntimeError("Rust session store returned an invalid list.")
    return value


def _token_usage(value: Any) -> TokenUsage:
    data = _mapping_result(value)
    return TokenUsage(
        input=int(data.get("input", 0)),
        output=int(data.get("output", 0)),
        reasoning=int(data.get("reasoning", 0)),
        cache_read=int(data.get("cache_read", 0)),
        cache_write=int(data.get("cache_write", 0)),
    )


def _cost_usage(value: Any) -> CostUsage:
    data = _mapping_result(value)
    return CostUsage(
        input=float(data.get("input", 0)),
        cached_input=float(data.get("cached_input", 0)),
        cache_write=float(data.get("cache_write", 0)),
        output=float(data.get("output", 0)),
    )


def _session_info(value: Any) -> SessionInfo:
    data = _mapping_result(value)
    tokens = data.get("tokens")
    if not isinstance(tokens, Mapping):
        tokens = {
            "input": data.get("tokens_input", 0),
            "output": data.get("tokens_output", 0),
            "reasoning": data.get("tokens_reasoning", 0),
            "cache_read": data.get("tokens_cache_read", 0),
            "cache_write": data.get("tokens_cache_write", 0),
        }
    return SessionInfo(
        id=str(data["id"]),
        project_id=data.get("project_id"),
        workspace_id=data.get("workspace_id"),
        title=str(data["title"]),
        workspace_root=str(data["workspace_root"]),
        cwd=data.get("cwd"),
        created_at=str(data["created_at"]),
        updated_at=str(data["updated_at"]),
        provider=data.get("provider"),
        model=data.get("model"),
        agent=data.get("agent"),
        permission=data.get("permission"),
        cost_usd=float(data.get("cost_usd", 0)),
        costs=_cost_usage(data.get("costs", {})),
        tokens=_token_usage(tokens),
        summary=data.get("summary"),
        active_root=data.get("active_root"),
        focus_path=data.get("focus_path"),
        last_prompt=data.get("last_prompt"),
        state=data.get("state"),
    )


def _route_info(value: Any) -> SessionRouteInfo:
    data = _mapping_result(value)
    return SessionRouteInfo(**{
        key: data.get(key)
        for key in (
            "route_key",
            "session_id",
            "agent_id",
            "scope",
            "channel",
            "account_id",
            "peer_kind",
            "peer_id",
            "sender_id",
            "guild_id",
            "team_id",
            "created_at",
            "updated_at",
        )
    })


def _message_info(value: Any) -> MessageInfo:
    data = _mapping_result(value)
    return MessageInfo(
        id=int(data["id"]),
        session_id=str(data["session_id"]),
        seq=int(data["seq"]),
        role=str(data["role"]),
        content=str(data["content"]),
        created_at=str(data["created_at"]),
        attachments=tuple(
            AttachmentInfo(
                id=str(item["id"]),
                filename=str(item["filename"]),
                mime=str(item["mime"]),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
                storage_path=str(item["storage_path"]),
                source=str(item["source"]),
                created_at=str(item["created_at"]),
            )
            for item in data.get("attachments", [])
            if isinstance(item, dict)
        ),
    )


def _event_info(value: Any) -> EventInfo:
    data = _mapping_result(value)
    return EventInfo(
        id=int(data["id"]),
        session_id=str(data["session_id"]),
        seq=int(data["seq"]),
        event_type=str(data["event_type"]),
        tool=data.get("tool"),
        args=data.get("args"),
        path=data.get("path"),
        summary=str(data["summary"]),
        data=data.get("data"),
        created_at=str(data["created_at"]),
    )


def _maintenance_report(value: Any) -> SessionMaintenanceReport:
    data = _mapping_result(value)
    return SessionMaintenanceReport(
        mode=str(data["mode"]),
        before_count=int(data["before_count"]),
        after_count=int(data["after_count"]),
        pruned=int(data["pruned"]),
        capped=int(data["capped"]),
        applied=bool(data["applied"]),
    )


def _checkpoint_to_public(value: Any) -> dict[str, Any]:
    data = dict(_mapping_result(value))
    aliases = {
        "checkpoint_id": "checkpointId",
        "session_key": "sessionKey",
        "session_id": "sessionId",
        "created_at": "createdAt",
        "pre_compaction": "preCompaction",
        "post_compaction": "postCompaction",
        "lines_before": "linesBefore",
        "lines_after": "linesAfter",
        "max_messages": "maxMessages",
        "first_kept_entry_id": "firstKeptEntryId",
    }
    for source, target in aliases.items():
        if source in data and target not in data:
            data[target] = data.pop(source)
    for key in ("preCompaction", "postCompaction"):
        nested = data.get(key)
        if isinstance(nested, Mapping):
            nested = dict(nested)
            if "session_id" in nested and "sessionId" not in nested:
                nested["sessionId"] = nested.pop("session_id")
            if "entry_id" in nested and "entryId" not in nested:
                nested["entryId"] = nested.pop("entry_id")
            data[key] = nested
    return data


def _validate_role(role: str) -> None:
    if role not in {"user", "assistant", "tool", "system"}:
        raise ValueError(f"Unsupported message role: {role}")


def _validated_route_checkpoint(
    route_key: str,
    checkpoint_id: str,
) -> tuple[str, str]:
    route_key = route_key.strip()
    if not route_key:
        raise ValueError("Session route key cannot be empty.")
    checkpoint_id = checkpoint_id.strip()
    if not checkpoint_id:
        raise ValueError("checkpoint_id cannot be empty.")
    return route_key, checkpoint_id


# Direct imports of this implementation first load the shared contract module.
# Complete that module's public re-export once this class exists.
import sys

_contract_module = sys.modules.get(f"{__package__}.session_store")
if _contract_module is not None:
    _contract_module.SessionStore = SessionStore
