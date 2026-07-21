from __future__ import annotations

import unittest
from typing import Any

from agent.channel_health_monitor import (
    ChannelHealthMonitor,
    ChannelHealthTimingPolicy,
)
from agent.gateway_runtime import ChannelLifecycleManager


class FakeClock:
    def __init__(self, now: int) -> None:
        self.now = now

    def now_ms(self) -> int:
        return self.now


class FakeChannelManager:
    def __init__(self, status: dict[str, Any]) -> None:
        self.status = status
        self.calls: list[tuple[str, str, str, object | None]] = []
        self.autostart_suppression = False
        self.manually_stopped = False

    def get_runtime_snapshot(self) -> dict[str, Any]:
        return {"channelAccounts": {"chat": {"default": self.status}}}

    def get_autostart_suppression(self) -> bool:
        return self.autostart_suppression

    def is_health_monitor_enabled(self, channel_id: str, account_id: str) -> bool:
        return True

    def is_manually_stopped(self, channel_id: str, account_id: str) -> bool:
        return self.manually_stopped

    def stop_channel(self, channel_id: str, account_id: str, *, manual: bool) -> None:
        self.calls.append(("stop", channel_id, account_id, manual))

    def reset_restart_attempts(self, channel_id: str, account_id: str) -> None:
        self.calls.append(("reset", channel_id, account_id, None))

    def start_channel(self, channel_id: str, account_id: str) -> None:
        self.calls.append(("start", channel_id, account_id, None))


class ChannelHealthMonitorTests(unittest.TestCase):
    def test_stale_running_channel_is_stopped_reset_and_started(self) -> None:
        clock = FakeClock(10_000_000)
        manager = FakeChannelManager({
            "running": True,
            "connected": True,
            "lastStartAt": clock.now - 2_000_000,
            "lastTransportActivityAt": clock.now - 31 * 60_000,
        })
        monitor = ChannelHealthMonitor(
            channel_manager=manager,
            timing=ChannelHealthTimingPolicy(monitor_startup_grace_ms=0),
            start_background=False,
            clock=clock,
        )

        monitor.run_check()

        self.assertEqual(manager.calls, [
            ("stop", "chat", "default", False),
            ("reset", "chat", "default", None),
            ("start", "chat", "default", None),
        ])

    def test_startup_grace_suppresses_checks(self) -> None:
        clock = FakeClock(10_000_000)
        manager = FakeChannelManager({"running": False})
        monitor = ChannelHealthMonitor(
            channel_manager=manager,
            timing=ChannelHealthTimingPolicy(monitor_startup_grace_ms=60_000),
            start_background=False,
            clock=clock,
        )

        monitor.run_check()

        self.assertEqual(manager.calls, [])

    def test_cooldown_limits_repeated_restarts(self) -> None:
        clock = FakeClock(10_000_000)
        manager = FakeChannelManager({
            "running": True,
            "connected": True,
            "lastStartAt": clock.now - 2_000_000,
            "lastTransportActivityAt": clock.now - 31 * 60_000,
        })
        monitor = ChannelHealthMonitor(
            channel_manager=manager,
            check_interval_ms=10_000,
            timing=ChannelHealthTimingPolicy(monitor_startup_grace_ms=0),
            cooldown_cycles=2,
            start_background=False,
            clock=clock,
        )

        monitor.run_check()
        clock.now += 1_000
        monitor.run_check()

        self.assertEqual(len(manager.calls), 3)

    def test_pending_restart_bypasses_fresh_restart_cooldown(self) -> None:
        clock = FakeClock(10_000_000)
        manager = FakeChannelManager({
            "running": True,
            "connected": True,
            "lastStartAt": clock.now - 2_000_000,
            "lastTransportActivityAt": clock.now - 31 * 60_000,
        })
        monitor = ChannelHealthMonitor(
            channel_manager=manager,
            check_interval_ms=10_000,
            timing=ChannelHealthTimingPolicy(monitor_startup_grace_ms=0),
            cooldown_cycles=2,
            start_background=False,
            clock=clock,
        )
        monitor.run_check()
        manager.calls.clear()
        manager.status = {
            "running": False,
            "restartPending": True,
            "reconnectAttempts": 0,
        }

        clock.now += 1_000
        monitor.run_check()

        self.assertEqual(manager.calls, [
            ("reset", "chat", "default", None),
            ("start", "chat", "default", None),
        ])

    def test_suppression_and_manual_stop_skip_restart(self) -> None:
        clock = FakeClock(10_000_000)
        manager = FakeChannelManager({"running": False})
        monitor = ChannelHealthMonitor(
            channel_manager=manager,
            timing=ChannelHealthTimingPolicy(monitor_startup_grace_ms=0),
            start_background=False,
            clock=clock,
        )

        manager.autostart_suppression = True
        monitor.run_check()
        manager.autostart_suppression = False
        manager.manually_stopped = True
        monitor.run_check()

        self.assertEqual(manager.calls, [])

    def test_channel_lifecycle_manager_satisfies_monitor_protocol(self) -> None:
        manager = ChannelLifecycleManager()
        manager.register("chat", "alerts")
        manager.set_health_monitor_enabled("chat", "alerts", enabled=True)

        manager.start_channel("chat", "alerts")
        snapshot = manager.get_runtime_snapshot()

        self.assertTrue(snapshot["channelAccounts"]["chat"]["alerts"]["running"])
        self.assertTrue(manager.is_health_monitor_enabled("chat", "alerts"))
        self.assertFalse(manager.is_manually_stopped("chat", "alerts"))

        manager.stop_channel("chat", "alerts", manual=False)
        manager.reset_restart_attempts("chat", "alerts")
        manager.start_channel("chat", "alerts")

        self.assertTrue(manager.get_runtime_snapshot()["channelAccounts"]["chat"]["alerts"]["running"])


if __name__ == "__main__":
    unittest.main()
