from __future__ import annotations

import random
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Generic, TypeVar

from .plugin_sdk import (
    PluginRegistry,
    pin_active_plugin_registry,
    release_pinned_plugin_registry,
)


T = TypeVar("T")
GatewayHandler = Callable[[Mapping[str, Any]], Any]


class GatewayRuntimeError(RuntimeError):
    pass


class LazyService(Generic[T]):
    """Thread-safe, inspectable lazy service loader with explicit failure policy."""

    def __init__(
        self,
        factory: Callable[[], T],
        *,
        retry_on_failure: bool = True,
    ) -> None:
        self._factory = factory
        self._retry_on_failure = retry_on_failure
        self._condition = threading.Condition(threading.RLock())
        self._state = "unloaded"
        self._value: T | None = None
        self._error: str | None = None

    def get(self) -> T:
        with self._condition:
            while self._state == "loading":
                self._condition.wait()
            if self._state == "ready":
                return self._value  # type: ignore[return-value]
            if self._state == "failed" and not self._retry_on_failure:
                raise GatewayRuntimeError(self._error or "Lazy service failed to load.")
            self._state = "loading"
            self._error = None

        try:
            value = self._factory()
        except Exception as exc:
            with self._condition:
                self._state = "failed"
                self._error = str(exc)
                self._condition.notify_all()
            raise

        with self._condition:
            self._value = value
            self._state = "ready"
            self._condition.notify_all()
            return value

    def peek(self) -> T | None:
        with self._condition:
            return self._value if self._state == "ready" else None

    def clear(self) -> None:
        with self._condition:
            if self._state == "loading":
                raise GatewayRuntimeError("Cannot clear a lazy service while it is loading.")
            self._state = "unloaded"
            self._value = None
            self._error = None

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "state": self._state,
                "retry_on_failure": self._retry_on_failure,
                "error": self._error,
            }


@dataclass(frozen=True)
class GatewayMethod:
    name: str
    handler: GatewayHandler
    required_scopes: frozenset[str] = frozenset({"gateway.read"})
    requires_ready: bool = True
    control_write: bool = False
    owner: str = "core"
    advertise: bool = True


class GatewayMethodRegistry:
    """Single dispatch table for gateway control-plane methods."""

    def __init__(self) -> None:
        self._methods: dict[str, GatewayMethod] = {}
        self._lock = threading.RLock()

    def register(self, method: GatewayMethod) -> None:
        name = method.name.strip()
        if not name:
            raise GatewayRuntimeError("Gateway method name cannot be empty.")
        if method.owner != "core" and method.control_write:
            raise GatewayRuntimeError(
                f"Extension method '{name}' cannot claim control-plane write access."
            )
        with self._lock:
            if name in self._methods:
                raise GatewayRuntimeError(f"Duplicate gateway method: {name}")
            self._methods[name] = GatewayMethod(
                name=name,
                handler=method.handler,
                required_scopes=method.required_scopes,
                requires_ready=method.requires_ready,
                control_write=method.control_write,
                owner=method.owner,
                advertise=method.advertise,
            )

    def get_handler(self, name: str) -> GatewayHandler | None:
        with self._lock:
            method = self._methods.get(name)
        return method.handler if method is not None else None

    def list_methods(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._methods)

    def list_advertised_methods(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                name for name, method in self._methods.items()
                if method.advertise
            )

    def get_scopes(self, name: str) -> frozenset[str] | None:
        with self._lock:
            method = self._methods.get(name)
        return method.required_scopes if method is not None else None

    def is_startup_unavailable(self, name: str) -> bool:
        with self._lock:
            method = self._methods.get(name)
        return method.requires_ready if method is not None else False

    def is_control_plane_write(self, name: str) -> bool:
        with self._lock:
            method = self._methods.get(name)
        return method.control_write if method is not None else False

    def descriptors(self) -> tuple[GatewayMethod, ...]:
        with self._lock:
            return tuple(self._methods.values())

    def dispatch(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        granted_scopes: frozenset[str] = frozenset({"gateway.read"}),
        ready: bool,
    ) -> Any:
        with self._lock:
            method = self._methods.get(name)
        if method is None:
            raise GatewayRuntimeError(f"Unknown gateway method: {name}")
        if method.requires_ready and not ready:
            raise GatewayRuntimeError(f"Gateway method is unavailable before ready: {name}")
        missing = method.required_scopes - granted_scopes
        if missing:
            raise GatewayRuntimeError(
                f"Gateway method '{name}' requires scope(s): {', '.join(sorted(missing))}"
            )
        return method.handler(params or {})

    def describe(self) -> list[dict[str, Any]]:
        with self._lock:
            methods = tuple(self._methods.values())
        return [
            {
                "name": method.name,
                "owner": method.owner,
                "scopes": sorted(method.required_scopes),
                "requires_ready": method.requires_ready,
                "control_write": method.control_write,
                "advertise": method.advertise,
            }
            for method in sorted(methods, key=lambda item: item.name)
        ]


@dataclass
class ChannelLifecycle:
    channel: str
    account_id: str
    state: str = "registered"
    generation: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    last_heartbeat: str | None = None
    retry_at: str | None = None
    start_pending: bool = False
    task_active: bool = False
    abort_requested: bool = False
    restart_pending: bool = False
    health_monitor_enabled: bool = False
    manually_stopped: bool = False
    last_start_at_ms: int | None = None
    last_transport_activity_at_ms: int | None = None
    last_run_activity_at_ms: int | None = None


class ChannelLifecycleManager:
    """Generation-safe channel state with bounded exponential restart policy."""

    def __init__(
        self,
        *,
        base_backoff_seconds: float = 5.0,
        max_backoff_seconds: float = 300.0,
        crash_loop_limit: int = 10,
        jitter_ratio: float = 0.10,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self._base_backoff = base_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._crash_loop_limit = crash_loop_limit
        self._jitter_ratio = jitter_ratio
        self._random_value = random_value
        self._channels: dict[tuple[str, str], ChannelLifecycle] = {}
        self._aborts: dict[tuple[str, str], threading.Event] = {}
        self._starting: dict[tuple[str, str], threading.Event] = {}
        self._tasks: dict[tuple[str, str], object] = {}
        self._lock = threading.RLock()

    def register(self, channel: str, account_id: str = "default") -> None:
        key = _channel_key(channel, account_id)
        with self._lock:
            if key in self._channels:
                raise GatewayRuntimeError(
                    f"Duplicate channel account: {key[0]}/{key[1]}"
                )
            self._channels[key] = ChannelLifecycle(channel=key[0], account_id=key[1])

    def begin_start(self, channel: str, account_id: str = "default") -> int:
        with self._lock:
            item = self._get(channel, account_id)
            if item.state == "crash_loop":
                raise GatewayRuntimeError(
                    f"Channel {item.channel}/{item.account_id} is crash-loop suppressed."
                )
            key = _channel_key(channel, account_id)
            if key in self._starting:
                return item.generation
            item.generation += 1
            item.state = "starting"
            item.retry_at = None
            item.start_pending = True
            item.task_active = False
            item.abort_requested = False
            item.restart_pending = True
            item.manually_stopped = False
            item.last_start_at_ms = _now_ms()
            self._starting[key] = threading.Event()
            self._aborts[key] = threading.Event()
            return item.generation

    def mark_running(self, channel: str, account_id: str, generation: int) -> bool:
        with self._lock:
            item = self._get(channel, account_id)
            if generation != item.generation:
                return False
            key = _channel_key(channel, account_id)
            item.state = "running"
            item.consecutive_failures = 0
            item.last_error = None
            item.retry_at = None
            item.last_heartbeat = _utc_now()
            item.start_pending = False
            item.task_active = True
            item.abort_requested = False
            item.restart_pending = False
            item.last_transport_activity_at_ms = _now_ms()
            start_gate = self._starting.pop(key, None)
            start_gate and start_gate.set()
            self._tasks[key] = object()
            return True

    def heartbeat(self, channel: str, account_id: str, generation: int) -> bool:
        with self._lock:
            item = self._get(channel, account_id)
            if generation != item.generation or item.state != "running":
                return False
            item.last_heartbeat = _utc_now()
            item.last_transport_activity_at_ms = _now_ms()
            return True

    def mark_failed(
        self,
        channel: str,
        account_id: str,
        generation: int,
        error: str,
    ) -> bool:
        with self._lock:
            item = self._get(channel, account_id)
            if generation != item.generation:
                return False
            key = _channel_key(channel, account_id)
            item.consecutive_failures += 1
            item.last_error = error
            item.last_heartbeat = None
            item.start_pending = False
            item.task_active = False
            item.abort_requested = False
            item.restart_pending = False
            start_gate = self._starting.pop(key, None)
            start_gate and start_gate.set()
            self._tasks.pop(key, None)
            self._aborts.pop(key, None)
            if item.consecutive_failures >= self._crash_loop_limit:
                item.state = "crash_loop"
                item.retry_at = None
                return True
            exponent = max(0, item.consecutive_failures - 1)
            base_delay = min(self._base_backoff * (2**exponent), self._max_backoff)
            jitter = (self._random_value() * 2.0 - 1.0) * self._jitter_ratio
            delay = max(0.0, base_delay * (1.0 + jitter))
            item.state = "backoff"
            item.retry_at = (
                datetime.now(timezone.utc) + timedelta(seconds=delay)
            ).isoformat()
            return True

    def stop(self, channel: str, account_id: str = "default", *, manual: bool = True) -> None:
        with self._lock:
            item = self._get(channel, account_id)
            key = _channel_key(channel, account_id)
            abort = self._aborts.get(key)
            abort and abort.set()
            item.generation += 1
            item.state = "stopped"
            item.retry_at = None
            item.last_heartbeat = None
            item.start_pending = False
            item.task_active = False
            item.abort_requested = True
            item.restart_pending = False
            item.manually_stopped = manual
            start_gate = self._starting.pop(key, None)
            start_gate and start_gate.set()
            self._tasks.pop(key, None)
            self._aborts.pop(key, None)

    def reset_crash_loop(self, channel: str, account_id: str = "default") -> None:
        with self._lock:
            item = self._get(channel, account_id)
            item.consecutive_failures = 0
            item.last_error = None
            item.retry_at = None
            item.state = "registered"
            item.start_pending = False
            item.task_active = False
            item.abort_requested = False
            item.restart_pending = False
            item.manually_stopped = False

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            values = tuple(self._channels.values())
        return [
            asdict(item)
            for item in sorted(values, key=lambda value: (value.channel, value.account_id))
        ]

    def lifecycle_store_snapshot(self) -> dict[str, list[dict[str, str]]]:
        with self._lock:
            return {
                "aborts": _channel_key_snapshots(self._aborts),
                "starting": _channel_key_snapshots(self._starting),
                "tasks": _channel_key_snapshots(self._tasks),
                "runtimes": _channel_key_snapshots(self._channels),
            }

    def get_runtime_snapshot(self) -> dict[str, Any]:
        with self._lock:
            values = tuple(self._channels.values())
        channel_accounts: dict[str, dict[str, dict[str, Any]]] = {}
        for item in values:
            accounts = channel_accounts.setdefault(item.channel, {})
            accounts[item.account_id] = {
                "configured": True,
                "enabled": True,
                "running": item.state == "running",
                "connected": item.state == "running",
                "terminalDisconnect": False,
                "restartPending": item.restart_pending,
                "reconnectAttempts": item.consecutive_failures,
                "lastStartAt": item.last_start_at_ms,
                "lastTransportActivityAt": item.last_transport_activity_at_ms,
                "lastRunActivityAt": item.last_run_activity_at_ms,
                "activeRuns": 0,
                "busy": False,
            }
        return {"channelAccounts": channel_accounts}

    def get_autostart_suppression(self) -> bool:
        return False

    def is_health_monitor_enabled(self, channel: str, account_id: str) -> bool:
        with self._lock:
            return self._get(channel, account_id).health_monitor_enabled

    def is_manually_stopped(self, channel: str, account_id: str) -> bool:
        with self._lock:
            return self._get(channel, account_id).manually_stopped

    def stop_channel(self, channel: str, account_id: str, *, manual: bool) -> None:
        self.stop(channel, account_id, manual=manual)

    def reset_restart_attempts(self, channel: str, account_id: str) -> None:
        with self._lock:
            item = self._get(channel, account_id)
            item.consecutive_failures = 0
            item.last_error = None
            if item.state == "crash_loop":
                item.state = "registered"

    def start_channel(self, channel: str, account_id: str) -> None:
        generation = self.begin_start(channel, account_id)
        self.mark_running(channel, account_id, generation)

    def set_health_monitor_enabled(
        self,
        channel: str,
        account_id: str = "default",
        *,
        enabled: bool,
    ) -> None:
        with self._lock:
            self._get(channel, account_id).health_monitor_enabled = enabled

    def _get(self, channel: str, account_id: str) -> ChannelLifecycle:
        key = _channel_key(channel, account_id)
        try:
            return self._channels[key]
        except KeyError as exc:
            raise GatewayRuntimeError(
                f"Channel account is not registered: {key[0]}/{key[1]}"
            ) from exc


class GatewayRuntimeState:
    def __init__(
        self,
        *,
        plugin_registry: PluginRegistry | None = None,
        release_plugin_registry: Callable[[], None] | None = None,
    ) -> None:
        self.started_at = _utc_now()
        self.ready = False
        self.methods = GatewayMethodRegistry()
        self.channels = ChannelLifecycleManager()
        self._lazy_services: dict[str, LazyService[Any]] = {}
        self.plugin_registry = plugin_registry
        self._release_plugin_registry = release_plugin_registry or (lambda: None)
        self._plugin_registry_released = False

    def register_lazy_service(self, name: str, service: LazyService[Any]) -> None:
        normalized = name.strip()
        if not normalized:
            raise GatewayRuntimeError("Lazy service name cannot be empty.")
        if normalized in self._lazy_services:
            raise GatewayRuntimeError(f"Duplicate lazy service: {normalized}")
        self._lazy_services[normalized] = service

    def mark_ready(self) -> None:
        self.ready = True

    def release_plugin_registry(self) -> None:
        if self._plugin_registry_released:
            return
        self._release_plugin_registry()
        self._plugin_registry_released = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": "ready" if self.ready else "starting",
            "started_at": self.started_at,
            "methods": len(self.methods.describe()),
            "channels": len(self.channels.snapshots()),
            "plugins": (
                self.plugin_registry.describe()
                if self.plugin_registry is not None
                else {"plugins": [], "channels": []}
            ),
            "lazy_services": {
                name: service.status()
                for name, service in sorted(self._lazy_services.items())
            },
        }


def create_gateway_runtime_state(
    *,
    plugin_registry: PluginRegistry,
    pin_plugin_registry: bool = True,
) -> GatewayRuntimeState:
    if pin_plugin_registry:
        pin_active_plugin_registry(plugin_registry)
    try:
        return GatewayRuntimeState(
            plugin_registry=plugin_registry,
            release_plugin_registry=(
                release_pinned_plugin_registry if pin_plugin_registry else None
            ),
        )
    except Exception:
        if pin_plugin_registry:
            release_pinned_plugin_registry()
        raise


def _channel_key_snapshots(items: Mapping[tuple[str, str], object]) -> list[dict[str, str]]:
    return [
        {"channel": channel, "account_id": account_id}
        for channel, account_id in sorted(items)
    ]


def _channel_key(channel: str, account_id: str) -> tuple[str, str]:
    normalized_channel = channel.strip().casefold()
    normalized_account = account_id.strip()
    if not normalized_channel or not normalized_account:
        raise GatewayRuntimeError("Channel and account id cannot be empty.")
    return normalized_channel, normalized_account


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
