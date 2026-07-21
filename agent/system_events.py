from __future__ import annotations

import time
from typing import Any

from .config import AgentConfig


MAX_EVENTS = 20

_queues: dict[str, dict[str, Any]] = {}


def resolve_main_system_event_session_key(config: AgentConfig | None = None) -> str:
    if config is not None and config.session.default_scope == "global":
        return "global"
    agent_id = (config.default_agent_id if config is not None else "main") or "main"
    return f"agent:{agent_id}:main"


def is_system_event_context_changed(session_key: str, context_key: str | None = None) -> bool:
    existing = _get_session_queue(session_key)
    normalized = _normalize_context_key(context_key)
    return normalized != (existing.get("last_context_key") if existing is not None else None)


def enqueue_system_event(
    text: str,
    *,
    session_key: str,
    context_key: str | None = None,
    trusted: bool = True,
) -> bool:
    key = _require_session_key(session_key)
    cleaned = text.strip()
    if not cleaned:
        return False
    entry = _get_or_create_session_queue(key)
    normalized_context_key = _normalize_context_key(context_key)
    entry["last_context_key"] = normalized_context_key
    if entry.get("last_text") == cleaned:
        return False
    entry["last_text"] = cleaned
    queue = entry["queue"]
    queue.append({
        "text": cleaned,
        "ts": _now_ms(),
        "contextKey": normalized_context_key,
        "trusted": trusted is not False,
    })
    if len(queue) > MAX_EVENTS:
        del queue[0]
    return True


def drain_system_event_entries(session_key: str) -> list[dict[str, Any]]:
    key = _require_session_key(session_key)
    entry = _get_session_queue(key)
    if entry is None or not entry["queue"]:
        return []
    out = [dict(event) for event in entry["queue"]]
    del _queues[key]
    return out


def drain_system_events(session_key: str) -> list[str]:
    return [event["text"] for event in drain_system_event_entries(session_key)]


def peek_system_event_entries(session_key: str) -> list[dict[str, Any]]:
    entry = _get_session_queue(session_key)
    if entry is None:
        return []
    return [dict(event) for event in entry["queue"]]


def peek_system_events(session_key: str) -> list[str]:
    return [event["text"] for event in peek_system_event_entries(session_key)]


def has_system_events(session_key: str) -> bool:
    entry = _get_session_queue(session_key)
    return bool(entry and entry["queue"])


def reset_system_events_for_test() -> None:
    _queues.clear()


def _get_session_queue(session_key: str) -> dict[str, Any] | None:
    return _queues.get(_require_session_key(session_key))


def _get_or_create_session_queue(session_key: str) -> dict[str, Any]:
    key = _require_session_key(session_key)
    entry = _queues.get(key)
    if entry is None:
        entry = {"queue": [], "last_text": None, "last_context_key": None}
        _queues[key] = entry
    return entry


def _require_session_key(key: str | None) -> str:
    trimmed = key.strip() if isinstance(key, str) else ""
    if not trimmed:
        raise ValueError("system events require a sessionKey")
    return trimmed


def _normalize_context_key(key: str | None) -> str | None:
    if not isinstance(key, str):
        return None
    trimmed = key.strip()
    return trimmed.lower() if trimmed else None


def _now_ms() -> int:
    return int(time.time() * 1000)
