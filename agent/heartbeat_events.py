from __future__ import annotations

import time
from typing import Any


HEARTBEAT_STATUSES = frozenset({"sent", "ok-empty", "ok-token", "skipped", "failed"})
HEARTBEAT_INDICATORS = {
    "sent": "alert",
    "ok-empty": "ok",
    "ok-token": "ok",
    "failed": "error",
}

_last_heartbeat: dict[str, Any] | None = None


def resolve_indicator_type(status: str) -> str | None:
    if status not in HEARTBEAT_STATUSES:
        raise ValueError("Unsupported heartbeat status")
    return HEARTBEAT_INDICATORS.get(status)


def emit_heartbeat_event(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if status not in HEARTBEAT_STATUSES:
        raise ValueError("Heartbeat status must be one of: failed, ok-empty, ok-token, sent, skipped.")
    event = {"ts": _heartbeat_timestamp_ms(), **payload}
    global _last_heartbeat
    _last_heartbeat = event
    return dict(event)


def get_last_heartbeat_event() -> dict[str, Any] | None:
    return dict(_last_heartbeat) if _last_heartbeat is not None else None


def reset_heartbeat_events_for_test() -> None:
    global _last_heartbeat
    _last_heartbeat = None


def _heartbeat_timestamp_ms() -> int:
    return int(time.time() * 1000)
