from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .channel_health_policy import (
    DEFAULT_CHANNEL_CONNECT_GRACE_MS,
    DEFAULT_CHANNEL_STALE_EVENT_THRESHOLD_MS,
    ChannelHealthPolicy,
    evaluate_channel_health,
    resolve_channel_restart_reason,
)


DEFAULT_CHECK_INTERVAL_MS = 5 * 60_000
DEFAULT_MONITOR_STARTUP_GRACE_MS = 60_000
DEFAULT_COOLDOWN_CYCLES = 2
DEFAULT_MAX_RESTARTS_PER_HOUR = 10
ONE_HOUR_MS = 60 * 60_000


class ChannelManager(Protocol):
    def get_runtime_snapshot(self) -> Mapping[str, Any]:
        ...

    def get_autostart_suppression(self) -> bool:
        ...

    def is_health_monitor_enabled(self, channel_id: str, account_id: str) -> bool:
        ...

    def is_manually_stopped(self, channel_id: str, account_id: str) -> bool:
        ...

    def stop_channel(self, channel_id: str, account_id: str, *, manual: bool) -> None:
        ...

    def reset_restart_attempts(self, channel_id: str, account_id: str) -> None:
        ...

    def start_channel(self, channel_id: str, account_id: str) -> None:
        ...


@dataclass(frozen=True)
class ChannelHealthTimingPolicy:
    monitor_startup_grace_ms: int = DEFAULT_MONITOR_STARTUP_GRACE_MS
    channel_connect_grace_ms: int = DEFAULT_CHANNEL_CONNECT_GRACE_MS
    stale_event_threshold_ms: int = DEFAULT_CHANNEL_STALE_EVENT_THRESHOLD_MS


@dataclass
class RestartRecord:
    last_restart_at: int = 0
    restarts_this_hour: list[int] = field(default_factory=list)


class ChannelHealthMonitor:
    def __init__(
        self,
        *,
        channel_manager: ChannelManager,
        check_interval_ms: int = DEFAULT_CHECK_INTERVAL_MS,
        timing: ChannelHealthTimingPolicy | None = None,
        cooldown_cycles: int = DEFAULT_COOLDOWN_CYCLES,
        max_restarts_per_hour: int = DEFAULT_MAX_RESTARTS_PER_HOUR,
        start_background: bool = True,
        clock: Any = None,
    ) -> None:
        self._channel_manager = channel_manager
        self._check_interval_ms = _positive_int(check_interval_ms, DEFAULT_CHECK_INTERVAL_MS)
        self._timing = timing or ChannelHealthTimingPolicy()
        self._cooldown_ms = cooldown_cycles * self._check_interval_ms
        self._max_restarts_per_hour = max_restarts_per_hour
        self._clock = clock or _SystemClock()
        self._started_at = self._clock.now_ms()
        self._restart_records: dict[str, RestartRecord] = {}
        self._suppressed_accounts: set[str] = set()
        self._stop_event = threading.Event()
        self._check_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if start_background:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._check_interval_ms / 1000)

    def run_check(self) -> None:
        if self._stop_event.is_set():
            return
        if not self._check_lock.acquire(blocking=False):
            return
        try:
            now = self._clock.now_ms()
            if now - self._started_at < self._timing.monitor_startup_grace_ms:
                return
            snapshot = self._channel_manager.get_runtime_snapshot()
            channel_accounts = snapshot.get("channelAccounts")
            if not isinstance(channel_accounts, Mapping):
                return
            autostart_suppression = self._channel_manager.get_autostart_suppression()
            if not autostart_suppression:
                self._suppressed_accounts.clear()
            for channel_id, accounts in channel_accounts.items():
                if not isinstance(accounts, Mapping):
                    continue
                for account_id, status in accounts.items():
                    if not isinstance(status, Mapping):
                        continue
                    channel = str(channel_id)
                    account = str(account_id)
                    if not self._channel_manager.is_health_monitor_enabled(channel, account):
                        continue
                    if self._channel_manager.is_manually_stopped(channel, account):
                        continue
                    key = _restart_key(channel, account)
                    if autostart_suppression:
                        if status.get("running") is not True:
                            self._suppressed_accounts.add(key)
                        continue
                    self._suppressed_accounts.discard(key)
                    health = evaluate_channel_health(
                        status,
                        ChannelHealthPolicy(
                            channel_id=channel,
                            now=now,
                            stale_event_threshold_ms=self._timing.stale_event_threshold_ms,
                            channel_connect_grace_ms=self._timing.channel_connect_grace_ms,
                        ),
                    )
                    if health.healthy or health.reason == "terminal-disconnect":
                        continue
                    record = self._restart_records.get(key) or RestartRecord()
                    continuing_pending_restart = (
                        status.get("running") is not True
                        and status.get("restartPending") is True
                        and (status.get("reconnectAttempts") or 0) == 0
                    )
                    if (
                        not continuing_pending_restart
                        and now - record.last_restart_at <= self._cooldown_ms
                    ):
                        continue
                    _prune_old_restarts(record, now)
                    if (
                        not continuing_pending_restart
                        and len(record.restarts_this_hour) >= self._max_restarts_per_hour
                    ):
                        continue
                    if not continuing_pending_restart:
                        record.last_restart_at = now
                        record.restarts_this_hour.append(now)
                        self._restart_records[key] = record
                    if status.get("running") is True:
                        self._channel_manager.stop_channel(channel, account, manual=False)
                    self._channel_manager.reset_restart_attempts(channel, account)
                    self._channel_manager.start_channel(channel, account)
                    resolve_channel_restart_reason(status, health)
        finally:
            self._check_lock.release()

    def _run_loop(self) -> None:
        interval_seconds = self._check_interval_ms / 1000
        while not self._stop_event.wait(interval_seconds):
            self.run_check()


def start_channel_health_monitor(
    *,
    channel_manager: ChannelManager,
    check_interval_ms: int = DEFAULT_CHECK_INTERVAL_MS,
    timing: ChannelHealthTimingPolicy | None = None,
    cooldown_cycles: int = DEFAULT_COOLDOWN_CYCLES,
    max_restarts_per_hour: int = DEFAULT_MAX_RESTARTS_PER_HOUR,
) -> ChannelHealthMonitor:
    return ChannelHealthMonitor(
        channel_manager=channel_manager,
        check_interval_ms=check_interval_ms,
        timing=timing,
        cooldown_cycles=cooldown_cycles,
        max_restarts_per_hour=max_restarts_per_hour,
    )


class _SystemClock:
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _positive_int(value: int, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


def _restart_key(channel_id: str, account_id: str) -> str:
    return f"{channel_id}:{account_id}"


def _prune_old_restarts(record: RestartRecord, now: int) -> None:
    record.restarts_this_hour = [
        item for item in record.restarts_this_hour
        if now - item < ONE_HOUR_MS
    ]
