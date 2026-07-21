from __future__ import annotations


_heartbeats_enabled = True


def set_heartbeats_enabled(enabled: bool) -> None:
    global _heartbeats_enabled
    _heartbeats_enabled = enabled


def are_heartbeats_enabled() -> bool:
    return _heartbeats_enabled


def reset_heartbeat_wake_state_for_tests() -> None:
    set_heartbeats_enabled(True)
