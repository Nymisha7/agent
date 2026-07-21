from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


DEFAULT_CHANNEL_STALE_EVENT_THRESHOLD_MS = 30 * 60_000
DEFAULT_CHANNEL_CONNECT_GRACE_MS = 120_000
BUSY_ACTIVITY_STALE_THRESHOLD_MS = 25 * 60_000

ChannelHealthReason = Literal[
    "healthy",
    "unmanaged",
    "not-running",
    "terminal-disconnect",
    "busy",
    "stuck",
    "startup-connect-grace",
    "disconnected",
    "stale-socket",
]
ChannelRestartReason = Literal["gave-up", "stopped", "stale-socket", "stuck", "disconnected"]


@dataclass(frozen=True)
class ChannelHealthPolicy:
    channel_id: str
    now: int
    stale_event_threshold_ms: int = DEFAULT_CHANNEL_STALE_EVENT_THRESHOLD_MS
    channel_connect_grace_ms: int = DEFAULT_CHANNEL_CONNECT_GRACE_MS


@dataclass(frozen=True)
class ChannelHealthEvaluation:
    healthy: bool
    reason: ChannelHealthReason


def evaluate_channel_health(
    snapshot: Mapping[str, Any],
    policy: ChannelHealthPolicy,
) -> ChannelHealthEvaluation:
    if snapshot.get("enabled") is False or snapshot.get("configured") is False:
        return ChannelHealthEvaluation(True, "unmanaged")

    if not snapshot.get("running") and snapshot.get("terminalDisconnect") is True:
        return ChannelHealthEvaluation(False, "terminal-disconnect")

    if not snapshot.get("running"):
        return ChannelHealthEvaluation(False, "not-running")

    active_runs = _finite_number(snapshot.get("activeRuns"))
    active_runs_count = max(0, math.trunc(active_runs)) if active_runs is not None else 0
    is_busy = snapshot.get("busy") is True or active_runs_count > 0
    last_start_at = _finite_number(snapshot.get("lastStartAt"))
    last_run_activity_at = _finite_number(snapshot.get("lastRunActivityAt"))
    last_transport_activity_at = _finite_number(snapshot.get("lastTransportActivityAt"))

    busy_state_initialized_for_lifecycle = (
        last_start_at is None
        or (
            last_run_activity_at is not None
            and last_run_activity_at >= last_start_at
        )
    )

    if is_busy and busy_state_initialized_for_lifecycle:
        run_activity_age = (
            math.inf
            if last_run_activity_at is None
            else max(0, policy.now - last_run_activity_at)
        )
        if run_activity_age < BUSY_ACTIVITY_STALE_THRESHOLD_MS:
            return ChannelHealthEvaluation(True, "busy")
        return ChannelHealthEvaluation(False, "stuck")

    if last_start_at is not None:
        up_duration = policy.now - last_start_at
        if up_duration < policy.channel_connect_grace_ms:
            return ChannelHealthEvaluation(True, "startup-connect-grace")

    if snapshot.get("connected") is False:
        return ChannelHealthEvaluation(False, "disconnected")

    should_check_stale_socket = (
        snapshot.get("connected") is True
        and last_transport_activity_at is not None
    )
    if should_check_stale_socket:
        if last_start_at is not None and last_transport_activity_at < last_start_at:
            lifecycle_event_gap = max(0, policy.now - last_start_at)
            if lifecycle_event_gap <= policy.stale_event_threshold_ms:
                return ChannelHealthEvaluation(True, "healthy")
            return ChannelHealthEvaluation(False, "stale-socket")
        event_age = policy.now - last_transport_activity_at
        if event_age > policy.stale_event_threshold_ms:
            return ChannelHealthEvaluation(False, "stale-socket")

    return ChannelHealthEvaluation(True, "healthy")


def resolve_channel_restart_reason(
    snapshot: Mapping[str, Any],
    evaluation: ChannelHealthEvaluation,
) -> ChannelRestartReason:
    if evaluation.reason == "stale-socket":
        return "stale-socket"
    if evaluation.reason == "not-running":
        reconnect_attempts = _finite_number(snapshot.get("reconnectAttempts"))
        return "gave-up" if reconnect_attempts and reconnect_attempts >= 10 else "stopped"
    if evaluation.reason == "disconnected":
        return "disconnected"
    return "stuck"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None
