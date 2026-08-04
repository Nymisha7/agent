from __future__ import annotations

# Implementation module loaded lazily by agent.gateway.

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol, TypeVar

from .config import AgentConfig, PEER_KINDS, RouteBindingConfig, SESSION_SCOPES
from .channel_health_policy import ChannelHealthPolicy, evaluate_channel_health
from .channel_health_monitor import start_channel_health_monitor
from .attachments import import_attachment
from .device_identity import load_or_create_device_identity, public_key_raw_base64url_from_pem
from .gateway_runtime import GatewayMethod, create_gateway_runtime_state
from .heartbeat_events import get_last_heartbeat_event
from .heartbeat_wake import set_heartbeats_enabled
from .session_store import SessionInfo, SessionStore
from .system_events import (
    enqueue_system_event,
    is_system_event_context_changed,
    peek_system_events,
    resolve_main_system_event_session_key,
)
from .system_presence import list_system_presence, update_system_presence
from .tool_groups import TOOL_GROUPS, tool_group_for


DEFAULT_ACCOUNT_ID = "default"
CHANNEL_STATUS_MAX_TIMEOUT_MS = 30_000
SESSION_PATCH_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})
SESSION_PATCH_UNSUPPORTED_FIELDS = frozenset({
    "authProfileOverride",
    "elevatedLevel",
    "execAsk",
    "execHost",
    "execNode",
    "execSecurity",
    "fastMode",
    "groupActivation",
    "reasoningLevel",
    "responseUsage",
    "sendPolicy",
    "spawnDepth",
    "spawnedBy",
    "unread",
    "verboseLevel",
})


class GatewayError(ValueError):
    pass


@dataclass(frozen=True)
class InboundAddress:
    channel: str
    account_id: str = DEFAULT_ACCOUNT_ID
    sender_id: str | None = None
    peer_kind: str | None = None
    peer_id: str | None = None
    guild_id: str | None = None
    team_id: str | None = None

    def __post_init__(self) -> None:
        _validate_identity(self.channel, "channel")
        _validate_identity(self.account_id, "account_id")
        if (self.peer_kind is None) != (self.peer_id is None):
            raise GatewayError("peer_kind and peer_id must be provided together")
        if self.peer_kind is not None and self.peer_kind not in PEER_KINDS:
            raise GatewayError(f"peer_kind must be one of: {', '.join(sorted(PEER_KINDS))}")
        for field_name in ("sender_id", "peer_id", "guild_id", "team_id"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_identity(value, field_name)


@dataclass(frozen=True)
class InboundMessage:
    address: InboundAddress
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise GatewayError("message text cannot be empty")
        if len(self.text) > 1_000_000:
            raise GatewayError("message text exceeds the 1,000,000 character ingress limit")


@dataclass(frozen=True)
class RouteDecision:
    agent_id: str
    scope: str
    route_key: str
    matched_binding: int | None


@dataclass(frozen=True)
class RoutedSession:
    decision: RouteDecision
    session: SessionInfo
    created: bool


class ChannelAdapter(Protocol):
    channel_id: str

    def normalize(self, payload: Mapping[str, Any]) -> InboundMessage:
        ...


Hook = Callable[[Mapping[str, Any]], None]
TurnResult = TypeVar("TurnResult")


@dataclass
class HookResult:
    event: str
    errors: list[str] = field(default_factory=list)


class LifecycleHooks:
    """Synchronous hooks that observe gateway events without changing routing decisions."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[Hook]] = defaultdict(list)
        self._lock = threading.RLock()

    def register(self, event: str, callback: Hook) -> None:
        event = event.strip()
        if not event:
            raise GatewayError("hook event cannot be empty")
        with self._lock:
            self._hooks[event].append(callback)

    def emit(self, event: str, payload: Mapping[str, Any]) -> HookResult:
        with self._lock:
            callbacks = tuple(self._hooks.get(event, ()))
        result = HookResult(event=event)
        for callback in callbacks:
            try:
                callback(payload)
            except Exception as exc:  # Hooks are observability extensions, not control flow.
                result.errors.append(str(exc))
        return result


class ChannelRegistry:
    """Programmatic extension point for channel adapters; no dynamic code is imported."""

    def __init__(self) -> None:
        self._adapters: dict[str, ChannelAdapter] = {}

    def register(self, adapter: ChannelAdapter) -> None:
        channel_id = adapter.channel_id.strip().casefold()
        if not channel_id:
            raise GatewayError("channel adapter id cannot be empty")
        if channel_id in self._adapters:
            raise GatewayError(f"duplicate channel adapter: {channel_id}")
        self._adapters[channel_id] = adapter

    def normalize(self, channel_id: str, payload: Mapping[str, Any]) -> InboundMessage:
        normalized = channel_id.strip().casefold()
        try:
            adapter = self._adapters[normalized]
        except KeyError as exc:
            raise GatewayError(f"channel adapter is not registered: {normalized}") from exc
        message = adapter.normalize(payload)
        if message.address.channel.casefold() != normalized:
            raise GatewayError("channel adapter returned a message for a different channel")
        return message

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


class LocalTuiChannel:
    """Built-in adapter for trusted local TUI/CLI ingress payloads."""

    channel_id = "tui"

    def normalize(self, payload: Mapping[str, Any]) -> InboundMessage:
        text = payload.get("text")
        if not isinstance(text, str):
            raise GatewayError("tui channel payload requires string field 'text'")
        return InboundMessage(
            address=InboundAddress(
                channel=self.channel_id,
                account_id=_payload_text(payload, "account_id") or DEFAULT_ACCOUNT_ID,
                sender_id=_payload_text(payload, "sender_id") or "local-user",
                peer_kind=_payload_text(payload, "peer_kind"),
                peer_id=_payload_text(payload, "peer_id"),
                guild_id=_payload_text(payload, "guild_id"),
                team_id=_payload_text(payload, "team_id"),
            ),
            text=text,
            metadata={
                str(key): value
                for key, value in payload.items()
                if key not in {
                    "text", "account_id", "sender_id", "peer_kind", "peer_id",
                    "guild_id", "team_id",
                }
            },
        )


class AgentGateway:
    """Agent's local control plane for channel normalization and durable session routing."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        store: SessionStore,
        channels: ChannelRegistry | None = None,
        hooks: LifecycleHooks | None = None,
    ) -> None:
        self.config = config
        self.store = store
        from .channel_plugins import builtin_plugin_registry

        plugin_registry = builtin_plugin_registry()
        self.channels = channels or ChannelRegistry()
        if channels is None:
            for channel_id in plugin_registry.channel_ids():
                self.channels.register(plugin_registry.create_channel(channel_id))
        self.hooks = hooks or LifecycleHooks()
        self.runtime = create_gateway_runtime_state(plugin_registry=plugin_registry)
        for channel_id in self.channels.ids():
            self.runtime.channels.register(channel_id)
        self._register_core_methods()
        self.runtime.mark_ready()
        self.health_monitor = start_channel_health_monitor(
            channel_manager=self.runtime.channels,
        )
        self._lease_guard = threading.Lock()
        self._session_leases: dict[str, threading.Lock] = {}
        self._send_idempotency: dict[str, dict[str, Any]] = {}
        self._send_idempotency_lock = threading.Lock()
        self._suspend_lock = threading.Lock()
        self._suspend_lease_id: str | None = None
        self._suspend_expires_at_ms: int | None = None

    def close(self) -> None:
        self.health_monitor.stop()
        self.runtime.release_plugin_registry()

    def _register_core_methods(self) -> None:
        methods = (
            GatewayMethod(
                name="health",
                handler=self._health_method,
            ),
            GatewayMethod(
                name="status",
                handler=self._status_method,
            ),
            GatewayMethod(
                name="diagnostics.stability",
                handler=self._diagnostics_stability_method,
            ),
            GatewayMethod(
                name="logs.tail",
                handler=self._logs_tail_method,
            ),
            GatewayMethod(
                name="gateway.suspend.prepare",
                handler=self._gateway_suspend_prepare_method,
                required_scopes=frozenset({"gateway.admin"}),
                control_write=True,
            ),
            GatewayMethod(
                name="gateway.suspend.status",
                handler=self._gateway_suspend_status_method,
            ),
            GatewayMethod(
                name="gateway.suspend.resume",
                handler=self._gateway_suspend_resume_method,
                required_scopes=frozenset({"gateway.admin"}),
            ),
            GatewayMethod(
                name="usage.status",
                handler=self._usage_status_method,
            ),
            GatewayMethod(
                name="usage.cost",
                handler=self._usage_cost_method,
            ),
            GatewayMethod(
                name="gateway.status",
                handler=lambda _params: self.status(),
                requires_ready=False,
            ),
            GatewayMethod(
                name="gateway.routes",
                handler=self._routes_method,
            ),
            GatewayMethod(
                name="gateway.bindings",
                handler=lambda _params: [asdict(binding) for binding in self.config.bindings],
            ),
            GatewayMethod(
                name="gateway.channels",
                handler=lambda _params: self.runtime.channels.snapshots(),
            ),
            GatewayMethod(
                name="channels.status",
                handler=self._channels_status_method,
            ),
            GatewayMethod(
                name="channels.start",
                handler=self._channels_start_method,
                required_scopes=frozenset({"gateway.write"}),
                control_write=True,
            ),
            GatewayMethod(
                name="channels.stop",
                handler=self._channels_stop_method,
                required_scopes=frozenset({"gateway.write"}),
                control_write=True,
            ),
            GatewayMethod(
                name="channels.logout",
                handler=self._channels_logout_method,
                required_scopes=frozenset({"gateway.write"}),
                control_write=True,
            ),
            GatewayMethod(
                name="models.list",
                handler=self._models_list_method,
            ),
            GatewayMethod(
                name="models.authStatus",
                handler=self._models_auth_status_method,
            ),
            GatewayMethod(
                name="commands.list",
                handler=self._commands_list_method,
            ),
            GatewayMethod(
                name="tools.catalog",
                handler=self._tools_catalog_method,
            ),
            GatewayMethod(
                name="tools.effective",
                handler=self._tools_effective_method,
            ),
            GatewayMethod(
                name="agents.list",
                handler=self._agents_list_method,
            ),
            GatewayMethod(
                name="skills.status",
                handler=self._skills_status_method,
            ),
            GatewayMethod(
                name="tasks.list",
                handler=self._tasks_list_method,
            ),
            GatewayMethod(
                name="tasks.get",
                handler=self._tasks_get_method,
            ),
            GatewayMethod(
                name="tasks.cancel",
                handler=self._tasks_cancel_method,
                required_scopes=frozenset({"gateway.write"}),
            ),
            GatewayMethod(
                name="config.get",
                handler=self._config_get_method,
            ),
            GatewayMethod(
                name="config.schema",
                handler=self._config_schema_method,
                required_scopes=frozenset({"gateway.admin"}),
            ),
            GatewayMethod(
                name="config.schema.lookup",
                handler=self._config_schema_lookup_method,
            ),
            GatewayMethod(
                name="chat.history",
                handler=self._chat_history_method,
            ),
            GatewayMethod(
                name="chat.metadata",
                handler=self._chat_metadata_method,
            ),
            GatewayMethod(
                name="chat.message.get",
                handler=self._chat_message_get_method,
            ),
            GatewayMethod(
                name="gateway.sessions",
                handler=self._sessions_method,
            ),
            GatewayMethod(
                name="gateway.identity.get",
                handler=self._gateway_identity_get_method,
            ),
            GatewayMethod(
                name="last-heartbeat",
                handler=lambda _params: get_last_heartbeat_event(),
            ),
            GatewayMethod(
                name="set-heartbeats",
                handler=self._set_heartbeats_method,
                required_scopes=frozenset({"gateway.admin"}),
                control_write=True,
            ),
            GatewayMethod(
                name="system-presence",
                handler=lambda _params: list_system_presence(),
            ),
            GatewayMethod(
                name="system-event",
                handler=self._system_event_method,
                required_scopes=frozenset({"gateway.admin"}),
            ),
            GatewayMethod(
                name="sessions.list",
                handler=self._sessions_list_method,
            ),
            GatewayMethod(
                name="sessions.preview",
                handler=self._sessions_preview_method,
            ),
            GatewayMethod(
                name="sessions.compaction.list",
                handler=self._sessions_compaction_list_method,
            ),
            GatewayMethod(
                name="sessions.compaction.get",
                handler=self._sessions_compaction_get_method,
            ),
            GatewayMethod(
                name="sessions.compaction.branch",
                handler=self._sessions_compaction_branch_method,
                required_scopes=frozenset({"gateway.write"}),
                control_write=True,
            ),
            GatewayMethod(
                name="sessions.compaction.restore",
                handler=self._sessions_compaction_restore_method,
                required_scopes=frozenset({"gateway.admin"}),
                control_write=True,
            ),
            GatewayMethod(
                name="sessions.describe",
                handler=self._sessions_describe_method,
            ),
            GatewayMethod(
                name="sessions.resolve",
                handler=self._sessions_resolve_method,
            ),
            GatewayMethod(
                name="sessions.create",
                handler=self._sessions_create_method,
                required_scopes=frozenset({"gateway.write"}),
                control_write=True,
            ),
            GatewayMethod(
                name="sessions.send",
                handler=self._sessions_send_method,
                required_scopes=frozenset({"gateway.write"}),
                control_write=True,
            ),
            GatewayMethod(
                name="sessions.steer",
                handler=self._sessions_steer_method,
                required_scopes=frozenset({"gateway.write"}),
                control_write=True,
            ),
            GatewayMethod(
                name="sessions.abort",
                handler=self._sessions_abort_method,
                required_scopes=frozenset({"gateway.write"}),
                control_write=True,
            ),
            GatewayMethod(
                name="sessions.patch",
                handler=self._sessions_patch_method,
                required_scopes=frozenset({"gateway.admin"}),
                control_write=True,
            ),
            GatewayMethod(
                name="sessions.reset",
                handler=self._sessions_reset_method,
                required_scopes=frozenset({"gateway.admin"}),
                control_write=True,
            ),
            GatewayMethod(
                name="sessions.delete",
                handler=self._sessions_delete_method,
                required_scopes=frozenset({"gateway.admin"}),
                control_write=True,
            ),
            GatewayMethod(
                name="sessions.compact",
                handler=self._sessions_compact_method,
                required_scopes=frozenset({"gateway.admin"}),
                control_write=True,
            ),
            GatewayMethod(
                name="gateway.methods",
                handler=lambda _params: self.runtime.methods.describe(),
            ),
        )
        for method in methods:
            self.runtime.methods.register(method)
        for method in _unsupported_gateway_methods():
            self.runtime.methods.register(method)

    def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        granted_scopes: frozenset[str] = frozenset({"gateway.read"}),
    ) -> Any:
        return self.runtime.methods.dispatch(
            method,
            params,
            granted_scopes=granted_scopes,
            ready=self.runtime.ready,
        )

    def control_snapshot(self, *, route_limit: int = 100, session_limit: int = 50) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overview": self.call("gateway.status"),
            "routes": self.call("gateway.routes", {"limit": route_limit}),
            "bindings": self.call("gateway.bindings"),
            "channels": self.call("gateway.channels"),
            "sessions": self.call("gateway.sessions", {"limit": session_limit}),
            "methods": self.call("gateway.methods"),
        }

    def _set_heartbeats_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        enabled = params.get("enabled")
        if type(enabled) is not bool:
            raise GatewayError("invalid set-heartbeats params: enabled (boolean) required")
        set_heartbeats_enabled(enabled)
        return {"ok": True, "enabled": enabled}

    def _gateway_identity_get_method(self, _params: Mapping[str, Any]) -> dict[str, str]:
        identity = load_or_create_device_identity()
        return {
            "deviceId": identity.device_id,
            "publicKey": public_key_raw_base64url_from_pem(identity.public_key_pem),
        }

    def _health_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        start = _now_epoch_ms()
        _health_probe(params)
        _health_timeout_ms(params)
        channels_status = self._channels_status_method({})
        sessions = self._health_sessions_summary(self.config.default_agent_id)
        agents = [
            {
                "agentId": agent_id,
                "isDefault": agent_id == self.config.default_agent_id,
                "sessions": self._health_sessions_summary(agent_id),
            }
            for agent_id in sorted(self.config.agents)
        ]
        return {
            "ok": True,
            "ts": _now_epoch_ms(),
            "durationMs": max(0, _now_epoch_ms() - start),
            "plugins": self._health_plugins_summary(),
            "channels": _health_channels_from_status(channels_status),
            "channelOrder": channels_status["channelOrder"],
            "channelLabels": channels_status["channelLabels"],
            "heartbeatSeconds": 0,
            "defaultAgentId": self.config.default_agent_id,
            "agents": agents,
            "sessions": sessions,
        }

    def _status_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        include_channel_summary = _status_include_channel_summary(params)
        channel_summary = []
        if include_channel_summary:
            channel_summary = _status_channel_summary(self._channels_status_method({}))
        return {
            "runtimeVersion": _agent_runtime_version(),
            "heartbeat": {
                "defaultAgentId": self.config.default_agent_id,
                "agents": [
                    {
                        "agentId": agent_id,
                        "enabled": False,
                        "every": "off",
                        "everyMs": None,
                    }
                    for agent_id in sorted(self.config.agents)
                ],
            },
            "channelSummary": channel_summary,
            "queuedSystemEvents": peek_system_events(resolve_main_system_event_session_key(self.config)),
            "tasks": _empty_task_registry_summary(),
            "taskAudit": _empty_task_audit_summary(),
            "sessions": self._status_sessions_summary(),
        }

    def _diagnostics_stability_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        _diagnostics_stability_params(params)
        runtime = self.runtime.snapshot()
        channels_status = self._channels_status_method({})
        return {
            "ok": True,
            "generatedAt": _now_epoch_ms(),
            "runtime": {
                "state": runtime["state"],
                "startedAt": runtime["started_at"],
                "methods": runtime["methods"],
                "lazyServices": runtime["lazy_services"],
            },
            "channels": {
                "count": len(channels_status["channelOrder"]),
                "order": channels_status["channelOrder"],
                "accounts": sum(
                    len(accounts)
                    for accounts in channels_status["channelAccounts"].values()
                    if isinstance(accounts, list)
                ),
            },
            "sessions": {
                "count": self.store.count_sessions(),
                "recent": [
                    {
                        "id": session.id,
                        "agentId": session.agent or self.config.default_agent_id,
                        "updatedAt": _iso_to_epoch_ms(session.updated_at),
                    }
                    for session in self.store.list_sessions(limit=5)
                ],
            },
            "plugins": self._health_plugins_summary(),
            "privacy": {
                "prompts": False,
                "messageBodies": False,
                "toolOutputs": False,
                "secrets": False,
            },
        }

    def _logs_tail_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        _logs_tail_params(params)
        return {
            "available": False,
            "entries": [],
            "cursor": None,
            "reason": "Agent Gateway file logging is not configured.",
        }

    def _gateway_suspend_prepare_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        timeout_ms = _suspend_timeout_ms(params)
        idle, reason = self._gateway_suspend_idle_state()
        if not idle:
            return {
                "ok": False,
                "prepared": False,
                "reason": reason,
                "leaseId": None,
            }
        lease_id = uuid.uuid4().hex
        expires_at = _now_epoch_ms() + timeout_ms
        with self._suspend_lock:
            self._suspend_lease_id = lease_id
            self._suspend_expires_at_ms = expires_at
        return {
            "ok": True,
            "prepared": True,
            "leaseId": lease_id,
            "expiresAt": expires_at,
            "timeoutMs": timeout_ms,
        }

    def _gateway_suspend_status_method(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        return self._gateway_suspend_status_payload()

    def _gateway_suspend_resume_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        requested_lease = _optional_text(params, "leaseId") or _optional_text(params, "lease_id")
        with self._suspend_lock:
            active = self._active_suspend_lease_unlocked()
            if requested_lease and active and requested_lease != self._suspend_lease_id:
                raise GatewayError("gateway.suspend.resume leaseId does not match active lease")
            resumed = active
            self._suspend_lease_id = None
            self._suspend_expires_at_ms = None
        return {
            "ok": True,
            "resumed": resumed,
            "active": False,
        }

    def _gateway_suspend_idle_state(self) -> tuple[bool, str | None]:
        for snapshot in self.runtime.channels.snapshots():
            if snapshot.get("task_active") is True:
                return False, f"channel {snapshot.get('channel') or 'unknown'} has active work"
            if snapshot.get("start_pending") is True:
                return False, f"channel {snapshot.get('channel') or 'unknown'} is starting"
        return True, None

    def _gateway_suspend_status_payload(self) -> dict[str, Any]:
        with self._suspend_lock:
            active = self._active_suspend_lease_unlocked()
            return {
                "active": active,
                "leaseId": self._suspend_lease_id if active else None,
                "expiresAt": self._suspend_expires_at_ms if active else None,
            }

    def _active_suspend_lease_unlocked(self) -> bool:
        if not self._suspend_lease_id or not self._suspend_expires_at_ms:
            self._suspend_lease_id = None
            self._suspend_expires_at_ms = None
            return False
        if self._suspend_expires_at_ms <= _now_epoch_ms():
            self._suspend_lease_id = None
            self._suspend_expires_at_ms = None
            return False
        return True

    def _config_get_method(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "config": _config_payload(self.config),
            "sourcePaths": [str(path) for path in self.config.source_paths],
            "configRevisionHash": _stable_digest(_config_payload(self.config)),
            "secretsIncluded": False,
        }

    def _config_schema_method(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        return _config_schema_payload()

    def _config_schema_lookup_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        path = _optional_text(params, "path")
        if not path:
            raise GatewayError("config.schema.lookup requires path")
        schema = _config_schema_payload()
        node: Any = schema["schema"]
        for part in path.split("."):
            if isinstance(node, Mapping) and "properties" in node and part in node["properties"]:
                node = node["properties"][part]
            else:
                return {"found": False, "path": path, "schema": None}
        return {"found": True, "path": path, "schema": node}

    def _chat_history_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        max_chars = _chat_max_chars(params)
        limit = _chat_limit(params)
        resolved = self._chat_resolve_session(params)
        messages = self.store.list_messages(str(resolved["session_id"]), limit=limit)
        return {
            "sessionKey": str(resolved.get("route_key") or resolved["session_id"]),
            "sessionId": str(resolved["session_id"]),
            "messages": [_chat_message_payload(message, max_chars=max_chars) for message in messages],
        }

    def _chat_metadata_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        resolved = self._chat_resolve_session(params)
        session = self.store.get_session(str(resolved["session_id"]))
        return {
            "sessionKey": str(resolved.get("route_key") or session.id),
            "sessionId": session.id,
            "title": session.title,
            "agentId": session.agent or self.config.default_agent_id,
            "provider": session.provider,
            "model": session.model,
            "updatedAt": _iso_to_epoch_ms(session.updated_at),
            "createdAt": _iso_to_epoch_ms(session.created_at),
            "tokens": {
                "input": session.tokens.input,
                "output": session.tokens.output,
                "reasoning": session.tokens.reasoning,
                "cacheRead": session.tokens.cache_read,
                "cacheWrite": session.tokens.cache_write,
            },
            "costUsd": session.cost_usd,
        }

    def _chat_message_get_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        message_id = _chat_message_id(params)
        max_chars = _chat_max_chars(params)
        resolved = self._chat_resolve_session(params)
        messages = self.store.list_messages(str(resolved["session_id"]), limit=None)
        for message in messages:
            if str(message.id) == message_id or str(message.seq) == message_id:
                return {
                    "sessionKey": str(resolved.get("route_key") or resolved["session_id"]),
                    "sessionId": str(resolved["session_id"]),
                    "message": _chat_message_payload(message, max_chars=max_chars),
                }
        raise GatewayError(f"message not found: {message_id}")

    def _chat_resolve_session(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = (
            _optional_text(params, "sessionKey")
            or _optional_text(params, "session_key")
            or _optional_text(params, "key")
            or _optional_text(params, "sessionId")
            or _optional_text(params, "session_id")
        )
        if not key:
            raise GatewayError("chat RPC requires sessionKey or sessionId")
        resolved = self._sessions_resolve_method({
            "session_key": key,
            "agent_id": _optional_text(params, "agentId"),
        })
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {key}")
        return resolved

    def _usage_status_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        agent_id = _usage_agent_filter(params, self.config)
        sessions = self.store.list_sessions(limit=None, agent_id=agent_id)
        return _usage_summary_payload(
            sessions,
            agent_id=agent_id,
            quota_available=False,
        )

    def _usage_cost_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        agent_id = _usage_agent_filter(params, self.config)
        sessions = self.store.list_sessions(limit=None, agent_id=agent_id)
        payload = _usage_summary_payload(
            sessions,
            agent_id=agent_id,
            quota_available=False,
        )
        payload["currency"] = "USD"
        return payload

    def _models_list_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        view = _models_list_view(params)
        rows = _models_catalog_rows(view)
        provider_rows: list[dict[str, Any]] = []
        seen_providers: set[str] = set()
        for row in rows:
            provider = str(row["provider"])
            if provider in seen_providers:
                continue
            seen_providers.add(provider)
            provider_rows.append({
                "id": provider,
                "label": row["providerLabel"],
                "local": row["local"],
                "auth": row["auth"],
            })
        return {
            "view": view,
            "models": rows,
            "providers": provider_rows,
        }

    def _models_auth_status_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        provider_filter = _models_auth_status_provider(params)
        providers = _models_auth_status_rows(provider_filter)
        return {
            "cached": False,
            "cacheTtlSeconds": 0,
            "generatedAt": _now_epoch_ms(),
            "providers": providers,
        }

    def _commands_list_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        agent_id = _commands_list_agent_id(params, self.config)
        scope = _commands_list_scope(params)
        include_args = _commands_list_include_args(params)
        provider = _commands_list_provider(params)
        commands = [
            _command_inventory_row(command, scope=scope, include_args=include_args)
            for command in _agent_command_inventory()
        ]
        return {
            "agentId": agent_id,
            "scope": scope,
            "provider": provider,
            "commands": commands,
        }

    def _tools_catalog_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        agent_id = _tools_catalog_agent_id(params, self.config)
        tools = _tools_catalog_rows(self.config, agent_id)
        return {
            "agentId": agent_id,
            "tools": tools,
            "groups": _tools_catalog_groups(tools),
        }

    def _tools_effective_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        session_key = _tools_effective_session_key(params)
        resolved = self._sessions_resolve_method({
            "session_key": session_key,
            "agent_id": _optional_text(params, "agentId"),
        })
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {session_key}")
        agent_id = str(resolved["agent_id"])
        tools = [
            tool
            for tool in _tools_catalog_rows(self.config, agent_id)
            if tool["enabled"] is True
        ]
        return {
            "sessionKey": str(resolved.get("route_key") or resolved["session_id"]),
            "sessionId": str(resolved["session_id"]),
            "agentId": agent_id,
            "tools": tools,
            "notices": [],
        }

    def _agents_list_method(self, _params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "defaultAgentId": self.config.default_agent_id,
            "agents": [
                {
                    "id": agent_id,
                    "isDefault": agent_id == self.config.default_agent_id,
                    "skills": (
                        list(self.config.skill_allowlist(agent_id))
                        if self.config.skill_allowlist(agent_id) is not None
                        else None
                    ),
                    "tools": (
                        list(self.config.tool_allowlist(agent_id))
                        if self.config.tool_allowlist(agent_id) is not None
                        else None
                    ),
                    "model": None,
                    "runtime": {
                        "orchestrator": "python",
                        "executor": "rust",
                    },
                }
                for agent_id in sorted(self.config.agents)
            ],
        }

    def _skills_status_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        agent_id = _skills_status_agent_id(params, self.config)
        workspace_root = _skills_status_workspace_root(self.store)
        from .skills import discover_skill_catalog

        catalog = discover_skill_catalog(
            workspace_root,
            self.config,
            agent_id=agent_id,
            tool_allowlist=self.config.tool_allowlist(agent_id),
        )
        return {
            "agentId": agent_id,
            "available": len(catalog.skills),
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "source": skill.source,
                    "requiredTools": list(skill.required_tools),
                    "requiredBins": list(skill.required_bins),
                    "eligible": True,
                }
                for skill in catalog.skills.values()
            ],
            "skipped": [
                {
                    "name": item.name,
                    "source": item.path.parent.name if item.name else item.path.name,
                    "reason": item.reason,
                    "eligible": False,
                }
                for item in catalog.skipped
            ],
            "roots": [
                {
                    "source": source,
                    "configured": root.exists(),
                }
                for source, root in catalog.roots
            ],
            "safety": (
                "Skills provide instructions only; tool permissions and approvals remain "
                "enforced by Agent."
            ),
        }

    def _tasks_list_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        _tasks_list_params(params, self.config)
        return {
            "tasks": [],
        }

    def _tasks_get_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        task_id = _task_id_param(params, method="tasks.get")
        raise GatewayError(f"task not found: {task_id}")

    def _tasks_cancel_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        task_id = _task_id_param(params, method="tasks.cancel")
        reason = _tasks_cancel_reason(params)
        result = {
            "found": False,
            "cancelled": False,
            "taskId": task_id,
        }
        if reason is not None:
            result["reason"] = reason
        return result

    def _status_sessions_summary(self) -> dict[str, Any]:
        return {
            "paths": [],
            "count": self.store.count_sessions(),
            "defaults": {
                "model": None,
                "contextTokens": None,
            },
            "recent": [],
            "byAgent": [
                {
                    "agentId": agent_id,
                    "path": "[redacted]",
                    "count": self.store.count_sessions(agent_id=agent_id),
                    "recent": [],
                }
                for agent_id in sorted(self.config.agents)
            ],
        }

    def _health_sessions_summary(self, agent_id: str | None = None) -> dict[str, Any]:
        sessions = self.store.list_sessions(limit=5, agent_id=agent_id)
        return {
            "path": str(self.store.db_path),
            "count": self.store.count_sessions(agent_id=agent_id),
            "recent": [
                {
                    "key": _session_health_key(self.store, session),
                    "updatedAt": _iso_to_epoch_ms(session.updated_at),
                    "age": _age_ms(session.updated_at),
                }
                for session in sessions
            ],
        }

    def _health_plugins_summary(self) -> dict[str, Any]:
        registry = self.runtime.plugin_registry
        if registry is None:
            return {"loaded": [], "errors": []}
        return {
            "loaded": sorted(registry.manifests),
            "errors": [],
        }

    def _channels_status_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        probe = _channels_status_probe(params)
        _channels_status_timeout_ms(params, probe=probe)
        available_channels = tuple(sorted({
            *self.channels.ids(),
            *(
                self.runtime.plugin_registry.channel_ids()
                if self.runtime.plugin_registry is not None
                else ()
            ),
        }))
        requested_channel = _channels_status_requested_channel(params, available_channels)
        selected_channels = (requested_channel,) if requested_channel else available_channels
        snapshots = self.runtime.channels.snapshots()
        snapshots_by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for snapshot in snapshots:
            channel = str(snapshot.get("channel") or "").casefold()
            if channel:
                snapshots_by_channel[channel].append(snapshot)

        channel_order = list(selected_channels)
        labels = {channel: _channel_label(channel) for channel in channel_order}
        detail_labels = {channel: _channel_detail_label(channel) for channel in channel_order}
        channels: dict[str, dict[str, Any]] = {}
        channel_accounts: dict[str, list[dict[str, Any]]] = {}
        default_account_ids: dict[str, str] = {}

        for channel in channel_order:
            account_snapshots = [
                self._channel_status_account_snapshot(snapshot)
                for snapshot in sorted(
                    snapshots_by_channel.get(channel, []),
                    key=lambda item: str(item.get("account_id") or DEFAULT_ACCOUNT_ID),
                )
            ]
            default_account_id = (
                next(
                    (
                        account["accountId"]
                        for account in account_snapshots
                        if account["accountId"] == DEFAULT_ACCOUNT_ID
                    ),
                    account_snapshots[0]["accountId"] if account_snapshots else DEFAULT_ACCOUNT_ID,
                )
            )
            default_account = (
                next((account for account in account_snapshots if account["accountId"] == default_account_id), None)
                or (account_snapshots[0] if account_snapshots else None)
            )
            channel_accounts[channel] = account_snapshots
            default_account_ids[channel] = default_account_id
            channels[channel] = _channel_status_summary(default_account, default_account_id)

        channel_meta = [
            {"id": channel, "label": labels[channel], "detailLabel": detail_labels[channel]}
            for channel in channel_order
        ]
        return {
            "ts": _now_epoch_ms(),
            "channelOrder": channel_order,
            "channelLabels": labels,
            "channelDetailLabels": detail_labels,
            "channelSystemImages": {},
            "channelMeta": channel_meta,
            "channels": channels,
            "channelAccounts": channel_accounts,
            "channelDefaultAccountId": default_account_ids,
        }

    def _channel_status_account_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        state = str(snapshot.get("state") or "registered")
        running = state == "running"
        account = {
            "accountId": str(snapshot.get("account_id") or DEFAULT_ACCOUNT_ID),
            "configured": True,
            "enabled": True,
            "running": running,
            "connected": running,
            "state": state,
            "terminalDisconnect": False,
            "restartPending": snapshot.get("restart_pending") is True,
            "reconnectAttempts": int(snapshot.get("consecutive_failures") or 0),
            "lastStartAt": snapshot.get("last_start_at_ms"),
            "lastTransportActivityAt": snapshot.get("last_transport_activity_at_ms"),
            "lastRunActivityAt": snapshot.get("last_run_activity_at_ms"),
            "activeRuns": 0,
            "busy": False,
            "startPending": snapshot.get("start_pending") is True,
            "taskActive": snapshot.get("task_active") is True,
            "abortRequested": snapshot.get("abort_requested") is True,
            "manuallyStopped": snapshot.get("manually_stopped") is True,
            "healthMonitorEnabled": snapshot.get("health_monitor_enabled") is True,
        }
        if snapshot.get("last_error"):
            account["lastError"] = snapshot["last_error"]
        if snapshot.get("last_heartbeat"):
            account["lastHeartbeat"] = snapshot["last_heartbeat"]
        if snapshot.get("retry_at"):
            account["retryAt"] = snapshot["retry_at"]
        health = evaluate_channel_health(
            account,
            ChannelHealthPolicy(
                channel_id=str(snapshot.get("channel") or ""),
                now=_now_epoch_ms(),
            ),
        )
        if not health.healthy:
            account["healthState"] = health.reason
        return account

    def _channels_stop_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        available_channels = tuple(sorted({
            *self.channels.ids(),
            *(
                self.runtime.plugin_registry.channel_ids()
                if self.runtime.plugin_registry is not None
                else ()
            ),
        }))
        channel_id, account_id = _channel_operation_params(
            params,
            method="channels.stop",
            available_channels=available_channels,
        )
        self.runtime.channels.stop_channel(channel_id, account_id, manual=True)
        runtime = self.runtime.channels.get_runtime_snapshot()
        account_snapshot = runtime["channelAccounts"].get(channel_id, {}).get(account_id)
        stopped = not (isinstance(account_snapshot, Mapping) and account_snapshot.get("running") is True)
        return {"channel": channel_id, "accountId": account_id, "stopped": stopped}

    def _channels_start_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        available_channels = tuple(sorted({
            *self.channels.ids(),
            *(
                self.runtime.plugin_registry.channel_ids()
                if self.runtime.plugin_registry is not None
                else ()
            ),
        }))
        channel_id, account_id = _channel_operation_params(
            params,
            method="channels.start",
            available_channels=available_channels,
        )
        self.runtime.channels.start_channel(channel_id, account_id)
        runtime = self.runtime.channels.get_runtime_snapshot()
        account_snapshot = runtime["channelAccounts"].get(channel_id, {}).get(account_id)
        started = isinstance(account_snapshot, Mapping) and account_snapshot.get("running") is True
        return {"channel": channel_id, "accountId": account_id, "started": started}

    def _channels_logout_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        available_channels = tuple(sorted({
            *self.channels.ids(),
            *(
                self.runtime.plugin_registry.channel_ids()
                if self.runtime.plugin_registry is not None
                else ()
            ),
        }))
        channel_id, _account_id = _channel_operation_params(
            params,
            method="channels.logout",
            available_channels=available_channels,
        )
        raise GatewayError(f"channel {channel_id} does not support logout")

    def _system_event_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        text_value = params.get("text")
        text = text_value.strip() if isinstance(text_value, str) else ""
        if not text:
            raise GatewayError("text required")
        session_key = resolve_main_system_event_session_key(self.config)
        presence_update = update_system_presence({
            "text": text,
            "deviceId": _system_event_optional_text(params, "deviceId"),
            "instanceId": _system_event_optional_text(params, "instanceId"),
            "host": _system_event_optional_text(params, "host"),
            "ip": _system_event_optional_text(params, "ip"),
            "mode": _system_event_optional_text(params, "mode"),
            "version": _system_event_optional_text(params, "version"),
            "platform": _system_event_optional_text(params, "platform"),
            "deviceFamily": _system_event_optional_text(params, "deviceFamily"),
            "modelIdentifier": _system_event_optional_text(params, "modelIdentifier"),
            "lastInputSeconds": _system_event_optional_finite_number(params, "lastInputSeconds"),
            "reason": _system_event_optional_text(params, "reason"),
            "roles": _system_event_optional_text_list(params, "roles"),
            "scopes": _system_event_optional_text_list(params, "scopes"),
            "tags": _system_event_optional_text_list(params, "tags"),
        })
        if text.startswith("Node:"):
            self._enqueue_node_presence_event(session_key, presence_update)
        else:
            enqueue_system_event(text, session_key=session_key)
        self.hooks.emit("presence_snapshot", {"presence": list_system_presence()})
        return {"ok": True}

    def _enqueue_node_presence_event(self, session_key: str, presence_update: Mapping[str, Any]) -> None:
        next_presence = presence_update.get("next")
        if not isinstance(next_presence, Mapping):
            return
        changed = set(presence_update.get("changedKeys") or [])
        reason_value = next_presence.get("reason")
        reason_text = reason_value if isinstance(reason_value, str) else ""
        normalized_reason = reason_text.lower()
        ignore_reason = normalized_reason.startswith("periodic") or normalized_reason == "heartbeat"
        host_changed = "host" in changed
        ip_changed = "ip" in changed
        version_changed = "version" in changed
        mode_changed = "mode" in changed
        reason_changed = "reason" in changed and not ignore_reason
        if not (host_changed or ip_changed or version_changed or mode_changed or reason_changed):
            return
        context_key = str(presence_update.get("key") or "")
        context_changed = is_system_event_context_changed(session_key, context_key)
        parts: list[str] = []
        if context_changed or host_changed or ip_changed:
            host_label = str(next_presence.get("host") or "").strip() or "Unknown"
            ip_label = str(next_presence.get("ip") or "").strip()
            parts.append(f"Node: {host_label}{f' ({ip_label})' if ip_label else ''}")
        if version_changed:
            parts.append(f"app {str(next_presence.get('version') or '').strip() or 'unknown'}")
        if mode_changed:
            parts.append(f"mode {str(next_presence.get('mode') or '').strip() or 'unknown'}")
        if reason_changed:
            parts.append(f"reason {reason_text.strip() or 'event'}")
        delta_text = " · ".join(parts)
        if delta_text:
            enqueue_system_event(delta_text, session_key=session_key, context_key=context_key)

    def _routes_method(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        limit = _bounded_limit(params.get("limit"), default=100)
        return [asdict(route) for route in self.store.list_routes(limit=limit)]

    def _sessions_method(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        limit = _bounded_limit(params.get("limit"), default=50)
        return [
            {
                "id": session.id,
                "title": session.title,
                "workspace_root": session.workspace_root,
                "agent_id": session.agent or self.config.default_agent_id,
                "provider": session.provider,
                "model": session.model,
                "updated_at": session.updated_at,
                "last_prompt": session.last_prompt,
                "routes": len(self.store.list_routes_for_session(session.id)),
            }
            for session in self.store.list_sessions(limit=limit)
        ]

    def _sessions_list_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        limit, limit_applied = _session_list_limit(params.get("limit"))
        active_minutes = (
            _optional_positive_int(params, "activeMinutes")
            or _optional_positive_int(params, "active_minutes")
            or _optional_positive_int(params, "active")
        )
        updated_after = (
            (datetime.now(timezone.utc) - timedelta(minutes=active_minutes)).isoformat()
            if active_minutes is not None
            else None
        )
        agent_id = _optional_text(params, "agentId") or _optional_text(params, "agent_id")
        all_agents = _optional_bool(params, "allAgents") or _optional_bool(params, "all_agents") or False
        configured_only = (
            _optional_bool(params, "configuredAgentsOnly")
            or _optional_bool(params, "configured_agents_only")
            or False
        )

        if configured_only and agent_id is not None and agent_id not in self.config.agents:
            sessions: list[SessionInfo] = []
            total_count = 0
        else:
            sessions = self.store.list_sessions(
                limit=limit,
                agent_id=agent_id,
                updated_after=updated_after,
            )
            total_count = self.store.count_sessions(
                agent_id=agent_id,
                updated_after=updated_after,
            )

        return {
            "ok": True,
            "path": None,
            "stores": [{
                "agentId": agent_id or self.config.default_agent_id,
                "path": str(self.store.db_path),
            }],
            "allAgents": all_agents,
            "count": len(sessions),
            "totalCount": total_count,
            "limitApplied": limit_applied,
            "hasMore": False if limit is None else total_count > len(sessions),
            "activeMinutes": active_minutes,
            "sessions": [
                self._session_list_entry(session)
                for session in sessions
            ],
        }

    def _session_list_entry(self, session: SessionInfo) -> dict[str, Any]:
        routes = self.store.list_routes_for_session(session.id)
        route_key = routes[0].route_key if routes else session.id
        agent_id = session.agent or self.config.default_agent_id
        model = _openclaw_model_name(session.provider, session.model)
        checkpoint_count = self.store.count_session_compaction_checkpoints(session.id)
        latest_checkpoint = None
        if checkpoint_count:
            checkpoints = self.store.list_session_compaction_checkpoints(
                session.id,
                session_key=route_key,
                limit=1,
            )
            latest_checkpoint = checkpoints[0] if checkpoints else None
        return {
            "id": session.id,
            "key": route_key,
            "sessionId": session.id,
            "session_id": session.id,
            "agentId": agent_id,
            "agent_id": agent_id,
            "label": session.title,
            "title": session.title,
            "workspaceRoot": session.workspace_root,
            "workspace_root": session.workspace_root,
            "createdAt": session.created_at,
            "created_at": session.created_at,
            "updatedAt": session.updated_at,
            "updated_at": session.updated_at,
            "lastPrompt": session.last_prompt,
            "last_prompt": session.last_prompt,
            "provider": session.provider,
            "model": model,
            "modelName": session.model,
            "agentRuntime": None,
            "routes": [asdict(route) for route in routes],
            "tokens": {
                "input": session.tokens.input,
                "output": session.tokens.output,
                "reasoning": session.tokens.reasoning,
                "cacheRead": session.tokens.cache_read,
                "cacheWrite": session.tokens.cache_write,
            },
            "costUsd": session.cost_usd,
            "summary": session.summary,
            "compactionCheckpointCount": checkpoint_count,
            "latestCompactionCheckpoint": latest_checkpoint,
        }

    def _sessions_preview_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        limit = min(_bounded_limit(params.get("limit"), default=20), 200)
        route_key = _optional_text(params, "session_key") or _optional_text(params, "route_key")
        session_id = _optional_text(params, "session_id")
        if route_key:
            route = self.store.get_route(route_key)
            session_id = route.session_id
        elif not session_id:
            key = _optional_text(params, "key")
            if key:
                try:
                    route = self.store.get_route(key)
                except KeyError:
                    session_id = self.store.resolve_session_id(key)
                else:
                    route_key = route.route_key
                    session_id = route.session_id
        if not session_id:
            raise GatewayError("sessions.preview requires session_key, route_key, key, or session_id")
        session = self.store.get_session(session_id)
        messages = self.store.list_messages(session.id, limit=limit)
        return {
            "session_id": session.id,
            "route_key": route_key,
            "limit": limit,
            "messages": [
                {
                    "id": message.id,
                    "seq": message.seq,
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at,
                }
                for message in messages
            ],
        }

    def _sessions_compaction_list_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = (
            _optional_text(params, "key")
            or _optional_text(params, "session_key")
            or _optional_text(params, "sessionKey")
            or _optional_text(params, "route_key")
            or _optional_text(params, "session_id")
            or _optional_text(params, "sessionId")
        )
        if not key:
            raise GatewayError("sessions.compaction.list requires key, session_key, route_key, or sessionId")
        limit = min(_bounded_limit(params.get("limit"), default=100), 500)
        resolved = self._sessions_resolve_method(params)
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {resolved.get('key')}")
        session_id = str(resolved["session_id"])
        canonical_key = str(resolved.get("route_key") or session_id)
        return {
            "ok": True,
            "key": canonical_key,
            "session_id": session_id,
            "sessionId": session_id,
            "checkpoints": self.store.list_session_compaction_checkpoints(
                session_id,
                session_key=canonical_key,
                limit=limit,
            ),
        }

    def _sessions_compaction_get_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = (
            _optional_text(params, "key")
            or _optional_text(params, "session_key")
            or _optional_text(params, "sessionKey")
            or _optional_text(params, "route_key")
            or _optional_text(params, "session_id")
            or _optional_text(params, "sessionId")
        )
        if not key:
            raise GatewayError("sessions.compaction.get requires key, session_key, route_key, or sessionId")
        checkpoint_id = _optional_text(params, "checkpointId") or _optional_text(params, "checkpoint_id")
        if not checkpoint_id:
            raise GatewayError("sessions.compaction.get checkpointId required")

        resolved = self._sessions_resolve_method(params)
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {resolved.get('key')}")
        session_id = str(resolved["session_id"])
        canonical_key = str(resolved.get("route_key") or session_id)
        checkpoint = self.store.get_session_compaction_checkpoint(
            session_id,
            session_key=canonical_key,
            checkpoint_id=checkpoint_id,
        )
        return {
            "ok": True,
            "key": canonical_key,
            "session_id": session_id,
            "sessionId": session_id,
            "checkpoint": checkpoint,
        }

    def _sessions_compaction_branch_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = (
            _optional_text(params, "key")
            or _optional_text(params, "session_key")
            or _optional_text(params, "sessionKey")
            or _optional_text(params, "route_key")
        )
        if not key:
            raise GatewayError("sessions.compaction.branch requires key, session_key, or route_key")
        checkpoint_id = _optional_text(params, "checkpointId") or _optional_text(params, "checkpoint_id")
        if not checkpoint_id:
            raise GatewayError("sessions.compaction.branch checkpointId required")

        resolved = self._sessions_resolve_method(params)
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {resolved.get('key')}")
        source_key = resolved.get("route_key")
        if not isinstance(source_key, str) or not source_key:
            raise GatewayError("sessions.compaction.branch requires a routed session key")

        branched = self.store.branch_routed_session_from_compaction_checkpoint(
            source_key,
            checkpoint_id=checkpoint_id,
        )
        entry = self._sessions_describe_method({
            "session_key": branched["key"],
            "session_id": branched["session_id"],
        })
        entry["parentSessionKey"] = source_key
        entry["parent_session_key"] = source_key
        return {
            "ok": True,
            "sourceKey": source_key,
            "source_key": source_key,
            "key": branched["key"],
            "sessionId": branched["session_id"],
            "session_id": branched["session_id"],
            "checkpoint": branched["checkpoint"],
            "entry": entry,
        }

    def _sessions_compaction_restore_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = (
            _optional_text(params, "key")
            or _optional_text(params, "session_key")
            or _optional_text(params, "sessionKey")
            or _optional_text(params, "route_key")
        )
        if not key:
            raise GatewayError("sessions.compaction.restore requires key, session_key, or route_key")
        checkpoint_id = _optional_text(params, "checkpointId") or _optional_text(params, "checkpoint_id")
        if not checkpoint_id:
            raise GatewayError("sessions.compaction.restore checkpointId required")

        resolved = self._sessions_resolve_method(params)
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {resolved.get('key')}")
        source_key = resolved.get("route_key")
        if not isinstance(source_key, str) or not source_key:
            raise GatewayError("sessions.compaction.restore requires a routed session key")
        current_session_id = str(resolved["session_id"])
        if self._session_has_active_lease(current_session_id):
            raise GatewayError(f"Cannot restore active session: {source_key}")

        restored = self.store.restore_routed_session_from_compaction_checkpoint(
            source_key,
            checkpoint_id=checkpoint_id,
        )
        entry = self._sessions_describe_method({
            "session_key": source_key,
            "session_id": restored["session_id"],
        })
        entry["sessionId"] = restored["session_id"]
        return {
            "ok": True,
            "key": source_key,
            "sessionId": restored["session_id"],
            "session_id": restored["session_id"],
            "previousSessionId": restored["previous_session_id"],
            "previous_session_id": restored["previous_session_id"],
            "checkpoint": restored["checkpoint"],
            "entry": entry,
        }

    def _sessions_describe_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        route_key = _optional_text(params, "session_key") or _optional_text(params, "route_key")
        session_id = _optional_text(params, "session_id")
        key = _optional_text(params, "key")
        if route_key:
            route = self.store.get_route(route_key)
            session_id = route.session_id
        elif key:
            try:
                route = self.store.get_route(key)
            except KeyError:
                session_id = key
            else:
                route_key = route.route_key
                session_id = route.session_id
        if not session_id:
            raise GatewayError("sessions.describe requires session_key, route_key, key, or session_id")
        session = self.store.get_session(session_id)
        routes = self.store.list_routes_for_session(session.id)
        return {
            "id": session.id,
            "session_id": session.id,
            "session_key": route_key,
            "route_key": route_key,
            "title": session.title,
            "workspace_root": session.workspace_root,
            "cwd": session.cwd,
            "agent_id": session.agent or self.config.default_agent_id,
            "provider": session.provider,
            "model": session.model,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "last_prompt": session.last_prompt,
            "summary": session.summary,
            "active_root": session.active_root,
            "focus_path": session.focus_path,
            "cost_usd": session.cost_usd,
            "tokens": {
                "input": session.tokens.input,
                "output": session.tokens.output,
                "reasoning": session.tokens.reasoning,
                "cache_read": session.tokens.cache_read,
                "cache_write": session.tokens.cache_write,
            },
            "routes": [asdict(route) for route in routes],
        }

    def _sessions_resolve_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        route_key = _optional_text(params, "session_key") or _optional_text(params, "route_key")
        key = _optional_text(params, "key")
        session_id = _optional_text(params, "session_id") or _optional_text(params, "sessionId")
        label = _optional_text(params, "label")
        agent_filter = _optional_text(params, "agent_id") or _optional_text(params, "agentId")
        include_unknown = (
            _optional_bool(params, "include_unknown")
            or _optional_bool(params, "includeUnknown")
            or False
        )
        if not route_key and key:
            route_key = key
        if not route_key and not session_id and not label:
            raise GatewayError("sessions.resolve requires key, session_key, route_key, sessionId, or label")

        route = None
        source = "session_id"
        try:
            if route_key:
                try:
                    route = self.store.get_route(route_key)
                except KeyError:
                    session_id = self.store.resolve_session_id(route_key)
                    route_key = None
                    source = "session_id"
                else:
                    session_id = route.session_id
                    route_key = route.route_key
                    source = "route_key"
            elif session_id:
                session_id = self.store.resolve_session_id(session_id)
                source = "session_id"
            elif label:
                session = self._resolve_session_label(label, agent_filter=agent_filter)
                session_id = session.id
                source = "label"
            if not session_id:
                raise KeyError("No session target resolved.")
            session = self.store.get_session(session_id)
        except KeyError:
            if include_unknown:
                unresolved = route_key or session_id or label or key
                return {
                    "found": False,
                    "key": unresolved,
                    "session_id": None,
                    "session_key": route_key,
                    "route_key": route_key,
                    "agent_id": agent_filter or self.config.default_agent_id,
                    "source": "unknown",
                }
            raise

        agent_id = session.agent or self.config.default_agent_id
        if agent_filter and agent_filter != agent_id:
            raise KeyError(f"Session target belongs to agent {agent_id}, not {agent_filter}")
        routes = self.store.list_routes_for_session(session.id)
        if route is None and routes:
            route = routes[0]
            route_key = route.route_key
        return {
            "found": True,
            "key": route_key or session.id,
            "session_id": session.id,
            "session_key": route_key,
            "route_key": route_key,
            "agent_id": agent_id,
            "title": session.title,
            "workspace_root": session.workspace_root,
            "source": source,
            "route": asdict(route) if route is not None else None,
        }

    def _resolve_session_label(
        self,
        label: str,
        *,
        agent_filter: str | None = None,
    ) -> SessionInfo:
        matches = [
            session
            for session in self.store.list_sessions(limit=500)
            if session.title == label
            and (agent_filter is None or (session.agent or self.config.default_agent_id) == agent_filter)
        ]
        if not matches:
            raise KeyError(f"No session label matches: {label}")
        if len(matches) > 1:
            raise GatewayError(f"Session label is ambiguous: {label}")
        return matches[0]

    def _sessions_create_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = _optional_text(params, "key") or _optional_text(params, "session_key") or _optional_text(params, "sessionKey")
        agent_id = _optional_text(params, "agent_id") or _optional_text(params, "agentId") or self.config.default_agent_id
        if agent_id not in self.config.agents:
            raise GatewayError(f"Unknown agent profile: {agent_id}")
        label = _optional_text(params, "label") or _optional_text(params, "title")
        provider = _optional_text(params, "provider")
        model = _optional_text(params, "model")
        workspace_root = _optional_path(params, "workspace_root") or _optional_path(params, "workspaceRoot") or _optional_path(params, "cwd") or Path.cwd()
        message = _optional_text(params, "message")
        task = _optional_text(params, "task")
        route_key = None

        if key:
            try:
                self.store.get_route(key)
            except KeyError:
                pass
            else:
                raise GatewayError(f"Session key already exists: {key}")
            scope = _optional_text(params, "scope") or "global"
            if scope not in SESSION_SCOPES:
                raise GatewayError(f"session scope must be one of: {', '.join(sorted(SESSION_SCOPES))}")
            session, created = self.store.get_or_create_routed_session(
                route_key=key,
                workspace_root=workspace_root,
                agent_id=agent_id,
                scope=scope,
                channel=_optional_text(params, "channel") or "gateway",
                account_id=_optional_text(params, "account_id") or _optional_text(params, "accountId") or DEFAULT_ACCOUNT_ID,
                provider=provider,
                model=model,
                title=label,
            )
            if not created:
                raise GatewayError(f"Session key already exists: {key}")
            route_key = key
        else:
            session = self.store.create_session(
                workspace_root=workspace_root,
                provider=provider,
                model=model,
                title=label,
                agent_id=agent_id,
            )

        result = self._sessions_describe_method({
            "session_key": route_key,
            "session_id": session.id,
        })
        result["created"] = True
        result["run_started"] = False
        result["runStarted"] = False
        if message or task:
            result["run_error"] = (
                "Initial nested send is not implemented in Agent yet; "
                "call sessions.send with the returned session key or session id."
            )
            result["runError"] = result["run_error"]
        return result

    def _sessions_send_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        message = _optional_text(params, "message")
        if not message:
            raise GatewayError("sessions.send requires a non-empty message")
        attachments = _gateway_attachment_inputs(params.get("attachments"))
        idempotency_key = _optional_text(params, "idempotency_key") or _optional_text(params, "idempotencyKey")
        if idempotency_key:
            cache_key = f"sessions.send:{idempotency_key}"
            with self._send_idempotency_lock:
                cached = self._send_idempotency.get(cache_key)
                if cached is not None:
                    return dict(cached)

        resolved = self._sessions_resolve_method(params)
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {resolved.get('key')}")
        session_id = str(resolved["session_id"])
        route_key = resolved.get("route_key")
        if attachments:
            messages = [self.store.add_message_with_attachments(
                session_id, "user", message, attachments, last_prompt=message,
                expected_route_key=route_key if isinstance(route_key, str) else None,
            )]
        else:
            messages = self.store.add_messages(
                session_id, [("user", message)], last_prompt=message,
                expected_route_key=route_key if isinstance(route_key, str) else None,
            )
        appended = messages[-1]
        run_id = idempotency_key or f"session-send-{uuid.uuid4().hex}"
        payload = {
            "ok": True,
            "status": "accepted",
            "run_id": run_id,
            "runId": run_id,
            "run_started": False,
            "runStarted": False,
            "run_error": (
                "No Agent sessions.send runner is wired yet; "
                "the user message was persisted to the session transcript."
            ),
            "runError": (
                "No Agent sessions.send runner is wired yet; "
                "the user message was persisted to the session transcript."
            ),
            "session_id": session_id,
            "sessionId": session_id,
            "session_key": route_key,
            "sessionKey": route_key,
            "message_seq": appended.seq,
            "messageSeq": appended.seq,
            "message_id": str(appended.id),
            "messageId": str(appended.id),
        }
        self._emit_transcript_update(
            session_id=session_id,
            route_key=route_key if isinstance(route_key, str) else None,
            agent_id=str(resolved.get("agent_id") or self.config.default_agent_id),
            message=appended,
        )
        if idempotency_key:
            with self._send_idempotency_lock:
                self._send_idempotency[f"sessions.send:{idempotency_key}"] = dict(payload)
        return payload

    def _emit_transcript_update(
        self,
        *,
        session_id: str,
        route_key: str | None,
        agent_id: str,
        message: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "route_key": route_key,
            "agent_id": agent_id,
            "message": {
                "role": message.role,
                "content": message.content,
            },
            "message_id": str(message.id),
            "message_seq": message.seq,
        }
        if route_key:
            payload["target"] = {
                "agent_id": agent_id,
                "session_id": session_id,
                "session_key": route_key,
            }
        self.hooks.emit("session_transcript_updated", payload)

    def _sessions_steer_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        message = _optional_text(params, "message")
        if not message:
            raise GatewayError("sessions.steer requires a non-empty message")
        if params.get("attachments") is not None:
            raise GatewayError("sessions.steer attachments are not implemented in Agent yet")
        resolved = self._sessions_resolve_method(params)
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {resolved.get('key')}")
        session_id = str(resolved["session_id"])
        active = self._session_has_active_lease(session_id)
        reason = "active-run-steering-not-wired" if active else "no-active-run"
        return {
            "ok": True,
            "accepted": False,
            "status": "unsupported" if active else "idle",
            "reason": reason,
            "session_id": session_id,
            "sessionId": session_id,
            "session_key": resolved.get("route_key"),
            "sessionKey": resolved.get("route_key"),
            "message": message,
            "run_started": False,
            "runStarted": False,
        }

    def _session_has_active_lease(self, session_id: str) -> bool:
        with self._lease_guard:
            lease = self._session_leases.get(session_id)
            return bool(lease and lease.locked())

    def _sessions_abort_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        run_id = _optional_text(params, "run_id") or _optional_text(params, "runId")
        has_target = any(
            _optional_text(params, key)
            for key in ("key", "session_key", "sessionKey", "route_key", "session_id", "sessionId")
        )
        if not has_target:
            if run_id:
                return {
                    "ok": True,
                    "aborted": False,
                    "status": "unsupported",
                    "reason": "run-id-resolution-not-wired",
                    "run_id": run_id,
                    "runId": run_id,
                    "session_id": None,
                    "sessionId": None,
                    "session_key": None,
                    "sessionKey": None,
                }
            raise GatewayError("sessions.abort requires key, session_key, route_key, sessionId, or runId")

        resolved = self._sessions_resolve_method(params)
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {resolved.get('key')}")
        session_id = str(resolved["session_id"])
        active = self._session_has_active_lease(session_id)
        status = "unsupported" if active else "idle"
        reason = "active-run-abort-not-wired" if active else "no-active-run"
        return {
            "ok": True,
            "aborted": False,
            "status": status,
            "reason": reason,
            "run_id": run_id,
            "runId": run_id,
            "session_id": session_id,
            "sessionId": session_id,
            "session_key": resolved.get("route_key"),
            "sessionKey": resolved.get("route_key"),
        }

    def _sessions_patch_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        for field_name in sorted(SESSION_PATCH_UNSUPPORTED_FIELDS):
            if field_name in params:
                raise GatewayError(
                    f"sessions.patch field '{field_name}' is not implemented in Agent yet"
                )

        has_target = any(
            _optional_text(params, key)
            for key in ("key", "session_key", "sessionKey", "route_key", "session_id", "sessionId")
        )
        if not has_target:
            raise GatewayError("sessions.patch requires key, session_key, route_key, or sessionId")

        target_params = dict(params)
        if "sessionKey" in target_params and "session_key" not in target_params:
            target_params["session_key"] = target_params["sessionKey"]
        target_params.pop("label", None)
        target_params.pop("title", None)
        resolved = self._sessions_resolve_method(target_params)
        if not resolved.get("found"):
            raise KeyError(f"Session not found: {resolved.get('key')}")

        title = _optional_text(params, "label") or _optional_text(params, "title")
        provider = (
            _optional_text(params, "provider")
            or _optional_text(params, "providerOverride")
        )
        model = _optional_text(params, "model") or _optional_text(params, "modelOverride")
        thinking = _session_patch_thinking_level(params)

        patched_fields: list[str] = []
        if title is not None:
            patched_fields.append("label")
        if provider is not None:
            patched_fields.append("provider")
        if model is not None:
            patched_fields.append("model")
        state_patch = {"reasoning_effort": thinking} if thinking is not None else None
        if state_patch is not None:
            patched_fields.append("thinkingLevel")

        if patched_fields:
            session = self.store.patch_session_metadata(
                str(resolved["session_id"]),
                title=title,
                provider=provider,
                model=model,
                state_patch=state_patch,
            )
        else:
            session = self.store.get_session(str(resolved["session_id"]))

        result = self._sessions_describe_method({
            "session_key": resolved.get("route_key"),
            "session_id": session.id,
        })
        result["ok"] = True
        result["patched"] = bool(patched_fields)
        result["patched_fields"] = patched_fields
        result["patchedFields"] = patched_fields
        result["resolved_model"] = {
            "provider": session.provider,
            "model": session.model,
        }
        result["resolvedModel"] = result["resolved_model"]
        result["agent_runtime"] = None
        result["agentRuntime"] = None
        return result

    def _sessions_reset_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = (
            _optional_text(params, "key")
            or _optional_text(params, "session_key")
            or _optional_text(params, "sessionKey")
            or _optional_text(params, "route_key")
        )
        if not key:
            raise GatewayError("sessions.reset requires key, session_key, or route_key")
        reason = _optional_text(params, "reason") or "reset"
        if reason not in {"new", "reset"}:
            raise GatewayError("sessions.reset reason must be 'new' or 'reset'")

        old_session, new_session = self.store.reset_routed_session(key, reason=reason)
        result = self._sessions_describe_method({
            "session_key": key,
            "session_id": new_session.id,
        })
        result["ok"] = True
        result["reset"] = True
        result["key"] = key
        result["reason"] = reason
        result["previous_session_id"] = old_session.id
        result["previousSessionId"] = old_session.id
        result["session_id"] = new_session.id
        result["sessionId"] = new_session.id
        result["entry"] = {
            key: value
            for key, value in result.items()
            if key not in {"ok", "reset", "reason", "previous_session_id", "previousSessionId"}
        }
        return result

    def _sessions_delete_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = (
            _optional_text(params, "key")
            or _optional_text(params, "session_key")
            or _optional_text(params, "sessionKey")
            or _optional_text(params, "route_key")
        )
        if not key:
            raise GatewayError("sessions.delete requires key, session_key, or route_key")
        delete_transcript = _optional_bool(params, "deleteTranscript")
        if delete_transcript is False:
            raise GatewayError(
                "sessions.delete cannot keep transcripts in Agent's SQLite session store yet"
            )

        route = self.store.get_route(key)
        if self._session_has_active_lease(route.session_id):
            raise GatewayError(f"Cannot delete active session: {key}")

        deleted = self.store.delete_routed_session(key)
        return {
            "ok": True,
            "deleted": True,
            "key": key,
            "session_id": deleted["session_id"],
            "sessionId": deleted["session_id"],
            "delete_transcript": True,
            "deleteTranscript": True,
            "messages_deleted": deleted["messages_deleted"],
            "messagesDeleted": deleted["messages_deleted"],
            "events_deleted": deleted["events_deleted"],
            "eventsDeleted": deleted["events_deleted"],
            "routes_deleted": deleted["routes_deleted"],
            "routesDeleted": deleted["routes_deleted"],
        }

    def _sessions_compact_method(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = (
            _optional_text(params, "key")
            or _optional_text(params, "session_key")
            or _optional_text(params, "sessionKey")
            or _optional_text(params, "route_key")
        )
        if not key:
            raise GatewayError("sessions.compact requires key, session_key, or route_key")
        max_lines = _optional_positive_int(params, "maxLines") or _optional_positive_int(params, "max_lines")
        if max_lines is None:
            raise GatewayError(
                "sessions.compact LLM summarization is not implemented in Agent yet; "
                "pass maxLines to run deterministic transcript truncation"
            )

        route = self.store.get_route(key)
        if self._session_has_active_lease(route.session_id):
            raise GatewayError(f"Cannot compact active session: {key}")

        compacted = self.store.compact_routed_session(key, max_messages=max_lines)
        archived_event_id = compacted["archived_event_id"]
        archived = f"sqlite:event:{archived_event_id}" if archived_event_id is not None else None
        return {
            "ok": True,
            "key": key,
            "session_id": compacted["session_id"],
            "sessionId": compacted["session_id"],
            "compacted": compacted["compacted"],
            "mode": "maxLines",
            "kept": compacted["kept"],
            "pruned": compacted["pruned"],
            "archived": archived,
            "archive_event_id": archived_event_id,
            "archiveEventId": archived_event_id,
            "result": {
                "linesBefore": compacted["lines_before"],
                "linesAfter": compacted["lines_after"],
            },
        }

    def decide_route(
        self,
        address: InboundAddress,
        *,
        workspace_root: Path | None = None,
    ) -> RouteDecision:
        ranked: list[tuple[int, int, RouteBindingConfig]] = []
        for index, binding in enumerate(self.config.bindings):
            specificity = _binding_specificity(binding, address)
            if specificity is not None:
                ranked.append((specificity, -index, binding))
        if ranked:
            _specificity, negative_index, binding = max(ranked, key=lambda item: (item[0], item[1]))
            agent_id = binding.agent_id
            scope = binding.scope or self.config.session.default_scope
            matched_binding = -negative_index
        else:
            agent_id = self.config.default_agent_id
            scope = self.config.session.default_scope
            matched_binding = None
        route_key = canonical_route_key(
            address,
            agent_id=agent_id,
            scope=scope,
            workspace_root=workspace_root,
        )
        return RouteDecision(
            agent_id=agent_id,
            scope=scope,
            route_key=route_key,
            matched_binding=matched_binding,
        )

    def open_session(
        self,
        address: InboundAddress,
        *,
        workspace_root: Path,
        provider: str | None = None,
        model: str | None = None,
    ) -> RoutedSession:
        workspace_root = workspace_root.expanduser().resolve()
        decision = self.decide_route(address, workspace_root=workspace_root)
        self.hooks.emit("route_resolved", {
            "channel": address.channel,
            "account_id": address.account_id,
            "agent_id": decision.agent_id,
            "scope": decision.scope,
            "route_key": decision.route_key,
        })
        session, created = self.store.get_or_create_routed_session(
            route_key=decision.route_key,
            workspace_root=workspace_root,
            agent_id=decision.agent_id,
            scope=decision.scope,
            channel=address.channel,
            account_id=address.account_id,
            peer_kind=address.peer_kind,
            peer_id=address.peer_id,
            sender_id=address.sender_id,
            guild_id=address.guild_id,
            team_id=address.team_id,
            provider=provider,
            model=model,
        )
        if created:
            self.store.add_event(
                session.id,
                event_type="session_route_bound",
                summary=(
                    f"Bound {address.channel} route to {decision.agent_id} "
                    f"with {decision.scope} scope"
                ),
                data={
                    "route_key": decision.route_key,
                    "agent_id": decision.agent_id,
                    "scope": decision.scope,
                    "channel": address.channel,
                    "account_id": address.account_id,
                    "peer_kind": address.peer_kind,
                    "peer_id": address.peer_id,
                },
            )
        self.hooks.emit("session_opened", {
            "session_id": session.id,
            "route_key": decision.route_key,
            "created": created,
            "agent_id": decision.agent_id,
        })
        return RoutedSession(decision=decision, session=session, created=created)

    def ingest(
        self,
        channel_id: str,
        payload: Mapping[str, Any],
        *,
        workspace_root: Path,
        provider: str | None = None,
        model: str | None = None,
    ) -> tuple[InboundMessage, RoutedSession]:
        message = self.channels.normalize(channel_id, payload)
        self.hooks.emit("message_ingress", {
            "channel": message.address.channel,
            "account_id": message.address.account_id,
            "metadata": dict(message.metadata),
        })
        routed = self.open_session(
            message.address,
            workspace_root=workspace_root,
            provider=provider,
            model=model,
        )
        return message, routed

    @contextmanager
    def session_lease(self, session_id: str) -> Iterator[None]:
        """Reject overlapping in-process parent turns for one durable session."""
        with self._lease_guard:
            lease = self._session_leases.setdefault(session_id, threading.Lock())
        if not lease.acquire(blocking=False):
            raise GatewayError(
                f"A parent turn is already active for session {session_id}. "
                "Queue the message or wait for that turn to complete."
            )
        try:
            yield
        finally:
            lease.release()

    def run_routed_turn(
        self,
        message: InboundMessage,
        routed: RoutedSession,
        handler: Callable[[SessionInfo, InboundMessage], TurnResult],
    ) -> TurnResult:
        """Run one routed parent turn synchronously; background execution is not supported."""
        with self.session_lease(routed.session.id):
            self.hooks.emit("turn_dispatch_started", {
                "session_id": routed.session.id,
                "route_key": routed.decision.route_key,
                "agent_id": routed.decision.agent_id,
                "channel": message.address.channel,
            })
            try:
                result = handler(routed.session, message)
                if inspect.isawaitable(result):
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
                    raise GatewayError(
                        "Gateway turn handlers must run synchronously to completion; "
                        "background or async parent turns are not supported."
                    )
            except Exception as exc:
                self.hooks.emit("turn_dispatch_failed", {
                    "session_id": routed.session.id,
                    "error": str(exc),
                })
                raise
            self.hooks.emit("turn_dispatch_completed", {
                "session_id": routed.session.id,
                "route_key": routed.decision.route_key,
            })
            return result

    def status(self) -> dict[str, Any]:
        runtime = self.runtime.snapshot()
        return {
            "control_plane": "local-python",
            "state": runtime["state"],
            "started_at": runtime["started_at"],
            "session_store": str(self.store.db_path),
            "default_agent": self.config.default_agent_id,
            "default_scope": self.config.session.default_scope,
            "bindings": len(self.config.bindings),
            "channels": list(self.channels.ids()),
            "method_count": runtime["methods"],
            "lazy_services": runtime["lazy_services"],
            "config_sources": [str(path) for path in self.config.source_paths],
            "execution_model": (
                "one active parent turn; parallel-only independent subagent batches "
                "with non-overlapping scoped writes"
            ),
        }


def canonical_route_key(
    address: InboundAddress,
    *,
    agent_id: str,
    scope: str,
    workspace_root: Path | str | None = None,
) -> str:
    if scope not in SESSION_SCOPES:
        raise GatewayError(f"unsupported session scope: {scope}")
    identity: dict[str, Any] = {
        "v": 1,
        "agent": agent_id,
        "scope": scope,
        "workspace": str(Path(workspace_root).expanduser().resolve()) if workspace_root else None,
    }
    if scope != "global":
        identity.update({
            "channel": address.channel.casefold(),
            "account": address.account_id,
            "peer_kind": address.peer_kind,
            "peer_id": address.peer_id,
            "guild_id": address.guild_id,
            "team_id": address.team_id,
        })
    if scope == "per-sender":
        sender_id = address.sender_id
        if not sender_id and address.peer_kind == "direct":
            sender_id = address.peer_id
        if not sender_id:
            raise GatewayError("per-sender routing requires sender_id or a direct peer id")
        identity["sender_id"] = sender_id
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
    return f"agent-route-v1:{digest}"


def _gateway_attachment_inputs(value: Any) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise GatewayError("attachments must be a non-empty array of local file paths.")
    if len(value) > 16:
        raise GatewayError("attachments may contain at most 16 files.")
    inputs: list[dict[str, object]] = []
    for item in value:
        path = item.get("path") if isinstance(item, Mapping) else item
        if not isinstance(path, str) or not path.strip():
            raise GatewayError("attachments must each be a non-empty path or {path} object.")
        try:
            inputs.append(import_attachment(path, source="gateway_upload").to_store_input())
        except ValueError as exc:
            raise GatewayError(str(exc)) from exc
    return inputs


UNSUPPORTED_READ_GATEWAY_METHODS = (
    "doctor.memory.status",
    "doctor.memory.dreamDiary",
    "doctor.memory.remHarness",
    "tts.status",
    "tts.providers",
    "tts.personas",
    "talk.catalog",
    "talk.config",
    "cron.get",
    "cron.list",
    "cron.status",
    "cron.runs",
    "plugins.list",
    "plugins.search",
    "node.list",
    "node.describe",
    "worktrees.list",
    "artifacts.list",
    "artifacts.get",
    "artifacts.download",
    "environments.list",
    "environments.status",
    "update.status",
    "voicewake.get",
    "voicewake.routing.get",
    "sessions.files.list",
    "sessions.files.get",
    "agents.files.list",
    "agents.files.get",
    "sessions.groups.list",
    "controlUi.sessionPullRequests",
    "system.info",
    "agents.workspace.list",
    "agents.workspace.get",
)

UNSUPPORTED_WRITE_GATEWAY_METHODS = (
    "tts.enable",
    "tts.disable",
    "tts.convert",
    "tts.setProvider",
    "tts.setPersona",
    "tts.speak",
    "talk.client.create",
    "talk.client.toolCall",
    "talk.client.steer",
    "talk.session.create",
    "talk.session.join",
    "talk.session.appendAudio",
    "talk.session.startTurn",
    "talk.session.endTurn",
    "talk.session.cancelTurn",
    "talk.session.cancelOutput",
    "talk.session.acknowledgeMark",
    "talk.session.submitToolResult",
    "talk.session.steer",
    "talk.session.close",
    "talk.speak",
    "talk.mode",
    "cron.run",
    "voicewake.set",
    "voicewake.routing.set",
    "sessions.groups.put",
    "sessions.groups.rename",
    "sessions.groups.delete",
    "message.action",
    "send",
    "agent",
    "chat.abort",
    "chat.send",
)

UNSUPPORTED_ADMIN_GATEWAY_METHODS = (
    "doctor.memory.backfillDreamDiary",
    "doctor.memory.resetDreamDiary",
    "doctor.memory.resetGroundedShortTerm",
    "doctor.memory.repairDreamingArtifacts",
    "doctor.memory.dedupeDreamDiary",
    "config.set",
    "config.apply",
    "config.patch",
    "config.openFile",
    "exec.approvals.get",
    "exec.approvals.set",
    "exec.approvals.node.get",
    "exec.approvals.node.set",
    "exec.approval.get",
    "exec.approval.list",
    "exec.approval.request",
    "exec.approval.waitDecision",
    "exec.approval.resolve",
    "plugin.approval.list",
    "plugin.approval.request",
    "plugin.approval.waitDecision",
    "plugin.approval.resolve",
    "plugins.install",
    "plugins.setEnabled",
    "plugins.uninstall",
    "agents.create",
    "agents.update",
    "agents.delete",
    "agents.files.set",
    "sessions.files.set",
    "worktrees.branches",
    "fs.listDir",
    "worktrees.create",
    "worktrees.remove",
    "worktrees.restore",
    "worktrees.gc",
    "terminal.open",
    "terminal.input",
    "terminal.resize",
    "terminal.close",
    "terminal.attach",
    "terminal.list",
    "terminal.text",
    "terminal.upload",
    "update.run",
    "environments.create",
    "environments.destroy",
    "openclaw.setup.detect",
    "openclaw.setup.activate",
    "openclaw.setup.auth.start",
    "openclaw.setup.verify",
    "wizard.start",
    "wizard.next",
    "wizard.cancel",
    "wizard.status",
    "secrets.reload",
    "secrets.resolve",
    "node.pair.list",
    "node.pair.approve",
    "node.pair.reject",
    "node.pair.remove",
    "node.rename",
    "node.pending.enqueue",
    "node.invoke",
    "device.pair.list",
    "device.pair.approve",
    "device.pair.reject",
    "device.pair.remove",
    "device.pair.rename",
    "device.token.rotate",
    "device.token.revoke",
)


def _unsupported_gateway_methods() -> tuple[GatewayMethod, ...]:
    methods: list[GatewayMethod] = []
    for name in UNSUPPORTED_READ_GATEWAY_METHODS:
        methods.append(GatewayMethod(name=name, handler=_unsupported_gateway_handler(name)))
    for name in UNSUPPORTED_WRITE_GATEWAY_METHODS:
        methods.append(GatewayMethod(
            name=name,
            handler=_unsupported_gateway_handler(name),
            required_scopes=frozenset({"gateway.write"}),
            control_write=True,
        ))
    for name in UNSUPPORTED_ADMIN_GATEWAY_METHODS:
        methods.append(GatewayMethod(
            name=name,
            handler=_unsupported_gateway_handler(name),
            required_scopes=frozenset({"gateway.admin"}),
            control_write=True,
        ))
    return tuple(methods)


def _unsupported_gateway_handler(name: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def handler(_params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "unsupported",
            "method": name,
            "reason": "This OpenClaw Gateway method is advertised for compatibility, but Agent has no implementation for this subsystem yet.",
        }

    return handler


def _binding_specificity(binding: RouteBindingConfig, address: InboundAddress) -> int | None:
    if binding.channel.casefold() != address.channel.casefold():
        return None
    if binding.account_id == "*":
        account_specificity = 2
    elif binding.account_id is None:
        if address.account_id != DEFAULT_ACCOUNT_ID:
            return None
        account_specificity = 1
    elif binding.account_id != address.account_id:
        return None
    else:
        account_specificity = 3
    if binding.peer_id is not None:
        if binding.peer_id != address.peer_id or binding.peer_kind != address.peer_kind:
            return None
        return 60 + account_specificity
    if binding.guild_id is not None:
        if binding.guild_id != address.guild_id:
            return None
        return 50 + account_specificity
    if binding.team_id is not None:
        if binding.team_id != address.team_id:
            return None
        return 40 + account_specificity
    return account_specificity * 10


def _validate_identity(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GatewayError(f"{field_name} cannot be empty")
    if len(value) > 256:
        raise GatewayError(f"{field_name} exceeds 256 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise GatewayError(f"{field_name} contains control characters")


def _payload_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise GatewayError(f"tui channel payload field '{key}' must be a string")
    return value or None


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise GatewayError(f"gateway payload field '{key}' must be a string")
    value = value.strip()
    return value or None


def _optional_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise GatewayError(f"gateway payload field '{key}' must be a boolean")
    return value


def _system_event_optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _system_event_optional_text_list(payload: Mapping[str, Any], key: str) -> list[str] | None:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return value


def _system_event_optional_finite_number(payload: Mapping[str, Any], key: str) -> int | float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _health_probe(payload: Mapping[str, Any]) -> bool:
    value = payload.get("probe")
    if value is None:
        return False
    if not isinstance(value, bool):
        raise GatewayError("invalid health params: probe must be a boolean")
    return value


def _health_timeout_ms(payload: Mapping[str, Any]) -> int:
    value = payload.get("timeoutMs")
    if value is None:
        return 10_000
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GatewayError("invalid health params: timeoutMs must be a finite number")
    return min(max(50, int(value)), CHANNEL_STATUS_MAX_TIMEOUT_MS)


def _status_include_channel_summary(payload: Mapping[str, Any]) -> bool:
    value = payload.get("includeChannelSummary")
    if value is None:
        return True
    if not isinstance(value, bool):
        raise GatewayError("invalid status params: includeChannelSummary must be a boolean")
    return value


def _diagnostics_stability_params(payload: Mapping[str, Any]) -> None:
    for key in ("probe", "deep"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            raise GatewayError(f"invalid diagnostics.stability params: {key} must be a boolean")


def _logs_tail_params(payload: Mapping[str, Any]) -> None:
    limit = payload.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 1_000):
        raise GatewayError("invalid logs.tail params: limit must be an integer from 1 to 1000")
    max_bytes = payload.get("maxBytes")
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1 or max_bytes > 1_000_000
    ):
        raise GatewayError("invalid logs.tail params: maxBytes must be an integer from 1 to 1000000")
    cursor = payload.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise GatewayError("invalid logs.tail params: cursor must be a string")


def _suspend_timeout_ms(payload: Mapping[str, Any]) -> int:
    value = payload.get("timeoutMs")
    if value is None:
        return 30_000
    if isinstance(value, bool) or not isinstance(value, int) or value < 1_000 or value > 300_000:
        raise GatewayError("invalid gateway.suspend.prepare params: timeoutMs must be an integer from 1000 to 300000")
    return value


def _config_payload(config: AgentConfig) -> dict[str, Any]:
    return {
        "agents": {
            "default": config.default_agent_id,
            "defaults": {
                "skills": list(config.default_skills) if config.default_skills is not None else None,
                "tools": list(config.default_tools) if config.default_tools is not None else None,
            },
            "list": [
                {
                    "id": agent.id,
                    "skills": list(agent.skills) if agent.skills is not None else None,
                    "tools": list(agent.tools) if agent.tools is not None else None,
                }
                for agent in config.agents.values()
            ],
        },
        "bindings": [asdict(binding) for binding in config.bindings],
        "session": {"default_scope": config.session.default_scope},
        "skills": {
            "extra_dirs": [str(path) for path in config.skills.extra_dirs],
            "max_loaded": config.skills.max_loaded,
            "max_instruction_chars": config.skills.max_instruction_chars,
        },
    }


def _config_schema_payload() -> dict[str, Any]:
    return {
        "schema": {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "object",
                    "properties": {
                        "default": {"type": "string"},
                        "defaults": {
                            "type": "object",
                            "properties": {
                                "skills": {"type": ["array", "null"], "items": {"type": "string"}},
                                "tools": {"type": ["array", "null"], "items": {"type": "string"}},
                            },
                        },
                        "list": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "skills": {"type": ["array", "null"], "items": {"type": "string"}},
                                    "tools": {"type": ["array", "null"], "items": {"type": "string"}},
                                },
                                "required": ["id"],
                            },
                        },
                    },
                },
                "bindings": {"type": "array"},
                "session": {
                    "type": "object",
                    "properties": {
                        "default_scope": {"type": "string", "enum": sorted(SESSION_SCOPES)},
                    },
                },
                "skills": {
                    "type": "object",
                    "properties": {
                        "extra_dirs": {"type": "array", "items": {"type": "string"}},
                        "max_loaded": {"type": "integer"},
                        "max_instruction_chars": {"type": "integer"},
                    },
                },
            },
        }
    }


def _chat_limit(payload: Mapping[str, Any]) -> int:
    return _bounded_limit(payload.get("limit"), default=100)


def _chat_max_chars(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("maxChars")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GatewayError("invalid chat params: maxChars must be a positive integer")
    return min(value, 1_000_000)


def _chat_message_id(payload: Mapping[str, Any]) -> str:
    value = (
        payload.get("messageId")
        if "messageId" in payload
        else payload.get("message_id")
        if "message_id" in payload
        else payload.get("id")
        if "id" in payload
        else payload.get("seq")
    )
    if isinstance(value, bool) or value is None:
        raise GatewayError("chat.message.get requires messageId, id, or seq")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise GatewayError("chat.message.get requires messageId, id, or seq")


def _chat_message_payload(message: Any, *, max_chars: int | None) -> dict[str, Any]:
    content = str(message.content)
    truncated = False
    if max_chars is not None and len(content) > max_chars:
        content = content[:max_chars]
        truncated = True
    return {
        "id": str(message.id),
        "seq": message.seq,
        "role": message.role,
        "content": content,
        "truncated": truncated,
        "createdAt": _iso_to_epoch_ms(message.created_at),
    }


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _models_list_view(payload: Mapping[str, Any]) -> str:
    value = payload.get("view")
    if value is None:
        return "default"
    if not isinstance(value, str):
        raise GatewayError("invalid models.list params: view must be a string")
    normalized = value.strip().casefold()
    if normalized not in {"default", "configured", "provider-config", "all"}:
        raise GatewayError("invalid models.list params: view must be default, configured, provider-config, or all")
    return normalized


def _usage_agent_filter(payload: Mapping[str, Any], config: AgentConfig) -> str | None:
    scope = payload.get("agentScope")
    if scope is not None:
        if not isinstance(scope, str):
            raise GatewayError("invalid usage params: agentScope must be a string")
        if scope.strip().casefold() != "all":
            raise GatewayError("invalid usage params: agentScope must be all")
        return None

    value = payload.get("agentId")
    if value is None:
        return config.default_agent_id
    if not isinstance(value, str) or not value.strip():
        raise GatewayError("invalid usage params: agentId must be a string")
    agent_id = value.strip()
    if agent_id not in config.agents:
        raise GatewayError(f"unknown agent: {agent_id}")
    return agent_id


def _usage_summary_payload(
    sessions: list[SessionInfo],
    *,
    agent_id: str | None,
    quota_available: bool,
) -> dict[str, Any]:
    totals = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "total": 0,
    }
    cost_usd = 0.0
    by_provider: dict[str, dict[str, Any]] = {}
    for session in sessions:
        tokens = session.tokens
        session_total = (
            max(0, tokens.input)
            + max(0, tokens.output)
            + max(0, tokens.reasoning)
            + max(0, tokens.cache_read)
            + max(0, tokens.cache_write)
        )
        totals["input"] += max(0, tokens.input)
        totals["output"] += max(0, tokens.output)
        totals["reasoning"] += max(0, tokens.reasoning)
        totals["cacheRead"] += max(0, tokens.cache_read)
        totals["cacheWrite"] += max(0, tokens.cache_write)
        totals["total"] += session_total
        cost_usd += max(0.0, float(session.cost_usd))
        provider = session.provider or "unknown"
        bucket = by_provider.setdefault(
            provider,
            {
                "provider": provider,
                "sessions": 0,
                "costUsd": 0.0,
                "tokens": {
                    "input": 0,
                    "output": 0,
                    "reasoning": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "total": 0,
                },
            },
        )
        bucket["sessions"] += 1
        bucket["costUsd"] += max(0.0, float(session.cost_usd))
        bucket_tokens = bucket["tokens"]
        bucket_tokens["input"] += max(0, tokens.input)
        bucket_tokens["output"] += max(0, tokens.output)
        bucket_tokens["reasoning"] += max(0, tokens.reasoning)
        bucket_tokens["cacheRead"] += max(0, tokens.cache_read)
        bucket_tokens["cacheWrite"] += max(0, tokens.cache_write)
        bucket_tokens["total"] += session_total

    return {
        "agentId": agent_id,
        "agentScope": "all" if agent_id is None else None,
        "quotaAvailable": quota_available,
        "sessions": len(sessions),
        "tokens": totals,
        "costUsd": cost_usd,
        "byProvider": [
            by_provider[provider]
            for provider in sorted(by_provider)
        ],
    }


def _models_catalog_rows(view: str) -> list[dict[str, Any]]:
    from . import main as main_module
    from .llm import _model_supports_reasoning

    rows: list[dict[str, Any]] = []
    install_by_ref = {
        (entry["provider"], entry["model"]): entry
        for entry in main_module.LOCAL_INSTALL_CATALOG
    }
    for provider in sorted(
        main_module.PROVIDER_MODEL_HINTS,
        key=lambda item: main_module.MODEL_PICKER_SORT_ORDER.get(item, 99),
    ):
        for model in main_module.PROVIDER_MODEL_HINTS.get(provider, ()):
            install_entry = install_by_ref.get((provider, model))
            row: dict[str, Any] = {
                "id": f"{provider}/{model}",
                "provider": provider,
                "providerLabel": main_module.PROVIDER_DISPLAY_NAMES.get(provider, provider),
                "model": model,
                "label": model,
                "local": provider in main_module.LOCAL_PROVIDERS,
                "openSource": provider in main_module.LOCAL_PROVIDERS,
                "status": _models_row_status(main_module, provider),
                "auth": _models_row_auth(main_module, provider),
                "contextWindow": main_module._context_window_for_model(model),
                "contextTokens": main_module._context_window_for_model(model),
                "input": ["text"],
                "capabilities": (
                    ["text", "reasoning"]
                    if _model_supports_reasoning(provider, model)
                    else ["text"]
                ),
            }
            if install_entry is not None:
                row["install"] = {
                    "provider": install_entry["provider"],
                    "id": install_entry["install_id"],
                    "parameters": install_entry["parameters"],
                    "size": install_entry["size"],
                    "memory": install_entry["memory"],
                    "context": install_entry["context"],
                    "quantization": install_entry["quantization"],
                    "local": True,
                }
            rows.append(row)
    return rows


def _models_row_status(main_module: Any, provider: str) -> str:
    if provider in main_module.LOCAL_PROVIDERS:
        return "local_runtime_not_checked"
    if provider in main_module.UNIMPLEMENTED_PROVIDER_TRANSPORTS:
        return "unavailable"
    env_name = main_module.PROVIDER_API_KEY_ENVS.get(provider)
    if env_name and not os.environ.get(env_name):
        return "auth_required"
    if provider == "openai-compatible" and not os.environ.get("AGENT_OPENAI_COMPAT_BASE_URL"):
        return "endpoint_required"
    return "available"


def _models_row_auth(main_module: Any, provider: str) -> dict[str, Any]:
    if provider in main_module.LOCAL_PROVIDERS:
        return {"required": False, "kind": "none"}
    env_name = main_module.PROVIDER_API_KEY_ENVS.get(provider)
    if env_name is not None:
        return {
            "required": True,
            "kind": "api_key",
            "env": env_name,
            "configured": bool(os.environ.get(env_name)),
            "setupUrl": main_module.PROVIDER_LOGIN_URLS.get(provider),
        }
    if provider in {"bedrock", "vertexai"}:
        return {
            "required": True,
            "kind": "cloud_credentials",
            "configured": False,
            "setupUrl": main_module.PROVIDER_LOGIN_URLS.get(provider),
        }
    if provider == "copilot":
        return {
            "required": True,
            "kind": "browser_login",
            "configured": False,
            "setupUrl": main_module.PROVIDER_LOGIN_URLS.get(provider),
        }
    return {"required": True, "kind": "setup", "configured": False}


def _models_auth_status_provider(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("provider")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise GatewayError("invalid models.authStatus params: provider must be a string")
    from .llm import _normalize_provider

    try:
        return _normalize_provider(value)
    except ValueError as exc:
        raise GatewayError(str(exc)) from exc


def _models_auth_status_rows(provider_filter: str | None) -> list[dict[str, Any]]:
    from . import main as main_module

    providers = [
        provider
        for provider in sorted(
            main_module.PROVIDER_MODEL_HINTS,
            key=lambda item: main_module.PROVIDER_SORT_ORDER.get(item, 99),
        )
        if provider_filter is None or provider == provider_filter
    ]
    return [
        _models_auth_status_row(main_module, provider)
        for provider in providers
    ]


def _models_auth_status_row(main_module: Any, provider: str) -> dict[str, Any]:
    auth = _models_row_auth(main_module, provider)
    required = auth.get("required") is True
    configured = bool(auth.get("configured")) if required else True
    if not required:
        status = "not_required"
    elif configured:
        status = "configured"
    elif provider in main_module.UNIMPLEMENTED_PROVIDER_TRANSPORTS:
        status = "unavailable"
    else:
        status = "missing"
    row = {
        "provider": provider,
        "label": main_module.PROVIDER_DISPLAY_NAMES.get(provider, provider),
        "status": status,
        "configured": configured,
        "required": required,
        "kind": auth.get("kind", "setup"),
        "attention": required and not configured,
    }
    env_name = auth.get("env")
    if isinstance(env_name, str):
        row["env"] = env_name
    setup_url = auth.get("setupUrl")
    if isinstance(setup_url, str):
        row["setupUrl"] = setup_url
    return row


def _commands_list_agent_id(payload: Mapping[str, Any], config: AgentConfig) -> str:
    value = payload.get("agentId")
    if value is None:
        return config.default_agent_id
    if not isinstance(value, str) or not value.strip():
        raise GatewayError("invalid commands.list params: agentId must be a string")
    agent_id = value.strip()
    if agent_id not in config.agents:
        raise GatewayError(f"unknown agent: {agent_id}")
    return agent_id


def _commands_list_scope(payload: Mapping[str, Any]) -> str:
    value = payload.get("scope")
    if value is None:
        return "both"
    if not isinstance(value, str):
        raise GatewayError("invalid commands.list params: scope must be a string")
    scope = value.strip().casefold()
    if scope not in {"text", "native", "both"}:
        raise GatewayError("invalid commands.list params: scope must be text, native, or both")
    return scope


def _commands_list_include_args(payload: Mapping[str, Any]) -> bool:
    value = payload.get("includeArgs")
    if value is None:
        return True
    if not isinstance(value, bool):
        raise GatewayError("invalid commands.list params: includeArgs must be a boolean")
    return value


def _commands_list_provider(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("provider")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise GatewayError("invalid commands.list params: provider must be a string")
    from .llm import _normalize_provider

    try:
        return _normalize_provider(value)
    except ValueError as exc:
        raise GatewayError(str(exc)) from exc


def _agent_command_inventory() -> list[dict[str, Any]]:
    from . import main as main_module

    descriptions = {name: description for name, description in main_module.LOCAL_COMMANDS}
    return [
        {
            "name": "model",
            "description": descriptions.get("/model", "Choose a model or local runtime"),
            "textAliases": ["/model", "/models"],
            "args": [
                {"name": "provider", "type": "string", "required": False},
                {"name": "model", "type": "string", "required": False},
            ],
        },
        {
            "name": "install",
            "description": descriptions.get("/install", "Install an open-source/open-weight model locally"),
            "textAliases": ["/install"],
            "args": [
                {"name": "provider", "type": "string", "required": True},
                {"name": "model", "type": "string", "required": True},
                {"name": "yes", "type": "boolean", "required": False},
            ],
        },
        {
            "name": "reasoning",
            "description": descriptions.get("/reasoning", "Set reasoning effort for supported models"),
            "textAliases": ["/reasoning"],
            "args": [
                {
                    "name": "effort",
                    "type": "string",
                    "required": False,
                    "enum": ["minimal", "low", "medium", "high"],
                },
            ],
        },
        {
            "name": "skills",
            "description": descriptions.get("/skills", "Show layered workspace and personal skills"),
            "textAliases": ["/skills"],
            "args": [],
        },
        {
            "name": "gateway",
            "description": descriptions.get("/gateway", "Show control-plane routing and session status"),
            "textAliases": ["/gateway"],
            "args": [],
        },
        {
            "name": "status",
            "description": descriptions.get("/status", "Show model, context, and session usage"),
            "textAliases": ["/status"],
            "args": [],
        },
        {
            "name": "setup",
            "description": descriptions.get("/setup", "Set up local runtimes or hosted providers"),
            "textAliases": ["/setup"],
            "args": [],
        },
        {
            "name": "apikey",
            "description": "Load a provider API key into this Agent process",
            "textAliases": ["/apikey", "/key"],
            "args": [
                {"name": "provider", "type": "string", "required": True},
                {"name": "apiKey", "type": "string", "required": False, "secret": True},
            ],
        },
        {
            "name": "login",
            "description": "Open a hosted provider account or API-key page",
            "textAliases": ["/login", "/auth"],
            "args": [{"name": "provider", "type": "string", "required": False}],
        },
        {
            "name": "help",
            "description": descriptions.get("/help", "Show commands and keyboard shortcuts"),
            "textAliases": ["/help"],
            "args": [],
        },
        {
            "name": "exit",
            "description": descriptions.get("/exit", "Close Agent"),
            "textAliases": ["/exit", "/quit", "/q"],
            "args": [],
        },
    ]


def _command_inventory_row(
    command: Mapping[str, Any],
    *,
    scope: str,
    include_args: bool,
) -> dict[str, Any]:
    text_name = str(command["name"])
    native_name = command.get("nativeName")
    selected_name = (
        str(native_name)
        if scope in {"native", "both"} and isinstance(native_name, str) and native_name
        else text_name
    )
    row = {
        "name": selected_name,
        "description": str(command.get("description") or ""),
        "source": "core",
        "textAliases": list(command.get("textAliases") or []),
    }
    if isinstance(native_name, str) and native_name:
        row["nativeName"] = native_name
    if include_args:
        row["args"] = list(command.get("args") or [])
    return row


def _tools_catalog_agent_id(payload: Mapping[str, Any], config: AgentConfig) -> str:
    value = payload.get("agentId")
    if value is None:
        return config.default_agent_id
    if not isinstance(value, str) or not value.strip():
        raise GatewayError("invalid tools.catalog params: agentId must be a string")
    agent_id = value.strip()
    if agent_id not in config.agents:
        raise GatewayError(f"unknown agent: {agent_id}")
    return agent_id


def _tools_catalog_rows(config: AgentConfig, agent_id: str) -> list[dict[str, Any]]:
    from .rust_tools import RustTools
    from .tools import ToolContext, build_tool_registry

    workspace_root = Path.cwd().resolve()
    ctx = ToolContext(
        rust=RustTools(rust_bin=Path("agent-rust")),
        workspace_root=workspace_root,
        search_roots=[workspace_root],
    )
    schemas = build_tool_registry(ctx).schemas()
    allowlist = config.tool_allowlist(agent_id)
    allowed = set(allowlist) if allowlist is not None else None
    rows: list[dict[str, Any]] = []
    for schema in schemas:
        name = str(schema.get("name") or "")
        if not name:
            continue
        group = tool_group_for(name)
        rows.append({
            "name": name,
            "description": str(schema.get("description") or ""),
            "source": "core",
            "group": group.id if group is not None else "other",
            "optional": False,
            "enabled": allowed is None or name in allowed,
            "schema": dict(schema),
        })
    return rows


def _tools_catalog_groups(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = {str(tool.get("name")) for tool in tools if tool.get("name")}
    groups: list[dict[str, Any]] = []
    for group in TOOL_GROUPS:
        names = [
            str(tool.get("name"))
            for tool in tools
            if tool.get("name") in group.tools
        ]
        if not names:
            continue
        remaining.difference_update(names)
        groups.append({
            "id": group.id,
            "label": group.label,
            "source": "core",
            "description": group.summary,
            "tools": names,
        })
    if remaining:
        groups.append({
            "id": "other",
            "label": "Other",
            "source": "core",
            "description": "Tools that are not assigned to a product area yet.",
            "tools": sorted(remaining),
        })
    return groups


def _tools_effective_session_key(payload: Mapping[str, Any]) -> str:
    value = payload.get("sessionKey") or payload.get("session_key") or payload.get("key")
    if not isinstance(value, str) or not value.strip():
        raise GatewayError("invalid tools.effective params: sessionKey must be a non-empty string")
    return value.strip()


def _skills_status_agent_id(payload: Mapping[str, Any], config: AgentConfig) -> str:
    value = payload.get("agentId")
    if value is None:
        return config.default_agent_id
    if not isinstance(value, str) or not value.strip():
        raise GatewayError("invalid skills.status params: agentId must be a string")
    agent_id = value.strip()
    if agent_id not in config.agents:
        raise GatewayError(f"unknown agent: {agent_id}")
    return agent_id


def _skills_status_workspace_root(store: SessionStore) -> Path:
    sessions = store.list_sessions(limit=1)
    if sessions:
        return Path(sessions[0].workspace_root).expanduser().resolve()
    return Path.cwd().resolve()


TASK_STATUSES = frozenset({"queued", "running", "completed", "failed", "cancelled", "timed_out"})


def _tasks_list_params(payload: Mapping[str, Any], config: AgentConfig) -> None:
    status = payload.get("status")
    if status is not None:
        if isinstance(status, str):
            statuses = [status]
        elif isinstance(status, list) and all(isinstance(item, str) for item in status):
            statuses = status
        else:
            raise GatewayError("invalid tasks.list params: status must be a string or string array")
        unknown = sorted({item for item in statuses if item not in TASK_STATUSES})
        if unknown:
            raise GatewayError(f"invalid tasks.list params: unknown status {', '.join(unknown)}")

    agent_id = payload.get("agentId")
    if agent_id is not None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise GatewayError("invalid tasks.list params: agentId must be a string")
        if agent_id.strip() not in config.agents:
            raise GatewayError(f"unknown agent: {agent_id.strip()}")

    session_key = payload.get("sessionKey")
    if session_key is not None and (not isinstance(session_key, str) or not session_key.strip()):
        raise GatewayError("invalid tasks.list params: sessionKey must be a string")

    cursor = payload.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise GatewayError("invalid tasks.list params: cursor must be a string")

    limit = payload.get("limit")
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 500:
        raise GatewayError("invalid tasks.list params: limit must be an integer from 1 to 500")


def _task_id_param(payload: Mapping[str, Any], *, method: str) -> str:
    value = payload.get("taskId")
    if not isinstance(value, str) or not value.strip():
        raise GatewayError(f"invalid {method} params: taskId must be a non-empty string")
    return value.strip()


def _tasks_cancel_reason(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("reason")
    if value is None:
        return None
    if not isinstance(value, str):
        raise GatewayError("invalid tasks.cancel params: reason must be a string")
    return value.strip() or None


def _agent_runtime_version() -> str | None:
    try:
        return importlib.metadata.version("agent")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _empty_task_registry_summary() -> dict[str, Any]:
    return {
        "total": 0,
        "active": 0,
        "terminal": 0,
        "failures": 0,
        "byStatus": {
            "queued": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "timed_out": 0,
            "cancelled": 0,
            "lost": 0,
        },
        "byRuntime": {
            "subagent": 0,
            "acp": 0,
            "cli": 0,
            "cron": 0,
        },
    }


def _empty_task_audit_summary() -> dict[str, Any]:
    return {
        "total": 0,
        "errors": 0,
        "warnings": 0,
        "byCode": {
            "stale_queued": 0,
            "stale_running": 0,
            "lost": 0,
            "delivery_failed": 0,
            "missing_cleanup": 0,
            "inconsistent_timestamps": 0,
        },
    }


def _health_channels_from_status(status: Mapping[str, Any]) -> dict[str, Any]:
    accounts_by_channel = status.get("channelAccounts")
    default_accounts = status.get("channelDefaultAccountId")
    if not isinstance(accounts_by_channel, Mapping):
        return {}
    out: dict[str, Any] = {}
    for channel, accounts_value in accounts_by_channel.items():
        if not isinstance(channel, str) or not isinstance(accounts_value, list):
            continue
        account_map: dict[str, Any] = {}
        default_account_id = (
            default_accounts.get(channel)
            if isinstance(default_accounts, Mapping)
            and isinstance(default_accounts.get(channel), str)
            else DEFAULT_ACCOUNT_ID
        )
        default_snapshot: Mapping[str, Any] | None = None
        for account in accounts_value:
            if not isinstance(account, Mapping):
                continue
            account_id = str(account.get("accountId") or DEFAULT_ACCOUNT_ID)
            account_map[account_id] = dict(account)
            if account_id == default_account_id:
                default_snapshot = account
        if default_snapshot is None and account_map:
            default_snapshot = next(iter(account_map.values()))
        if default_snapshot is None:
            continue
        summary = dict(default_snapshot)
        summary["accounts"] = account_map
        out[channel] = summary
    return out


def _status_channel_summary(status: Mapping[str, Any]) -> list[str]:
    channel_order = status.get("channelOrder")
    channel_accounts = status.get("channelAccounts")
    default_accounts = status.get("channelDefaultAccountId")
    labels = status.get("channelLabels")
    if (
        not isinstance(channel_order, list)
        or not isinstance(channel_accounts, Mapping)
        or not isinstance(default_accounts, Mapping)
    ):
        return []
    out: list[str] = []
    for channel in channel_order:
        if not isinstance(channel, str):
            continue
        accounts = channel_accounts.get(channel)
        if not isinstance(accounts, list):
            continue
        default_account_id = default_accounts.get(channel)
        if not isinstance(default_account_id, str):
            default_account_id = DEFAULT_ACCOUNT_ID
        account = next(
            (
                item
                for item in accounts
                if isinstance(item, Mapping)
                and str(item.get("accountId") or DEFAULT_ACCOUNT_ID) == default_account_id
            ),
            None,
        )
        if not isinstance(account, Mapping) and accounts:
            first = accounts[0]
            account = first if isinstance(first, Mapping) else None
        label = labels.get(channel) if isinstance(labels, Mapping) else None
        label_text = label if isinstance(label, str) and label else _channel_label(channel)
        if not isinstance(account, Mapping):
            out.append(f"{label_text}: not configured")
            continue
        state = str(account.get("state") or "registered")
        account_id = str(account.get("accountId") or DEFAULT_ACCOUNT_ID)
        running = account.get("running") is True
        suffix = "running" if running else state
        if account_id == DEFAULT_ACCOUNT_ID:
            out.append(f"{label_text}: {suffix} (default)")
        else:
            out.append(f"{label_text}: {suffix} ({account_id})")
    return out


def _channels_status_probe(payload: Mapping[str, Any]) -> bool:
    value = payload.get("probe")
    if value is None:
        return False
    if not isinstance(value, bool):
        raise GatewayError("invalid channels.status params: probe must be a boolean")
    return value


def _channels_status_timeout_ms(payload: Mapping[str, Any], *, probe: bool) -> int:
    value = payload.get("timeoutMs")
    fallback = CHANNEL_STATUS_MAX_TIMEOUT_MS if probe else 10_000
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GatewayError("invalid channels.status params: timeoutMs must be a finite number")
    return min(max(1_000, int(value)), CHANNEL_STATUS_MAX_TIMEOUT_MS)


def _channels_status_requested_channel(
    payload: Mapping[str, Any],
    available_channels: tuple[str, ...],
) -> str | None:
    value = payload.get("channel")
    if value is None:
        return None
    if not isinstance(value, str):
        raise GatewayError(f"unknown channel: {value!r}")
    channel = value.strip().casefold()
    if not channel or channel not in available_channels:
            raise GatewayError(f"unknown channel: {value}")
    return channel


def _channel_operation_params(
    payload: Mapping[str, Any],
    *,
    method: str,
    available_channels: tuple[str, ...],
) -> tuple[str, str]:
    raw_channel = payload.get("channel")
    if not isinstance(raw_channel, str) or not raw_channel.strip():
        raise GatewayError(f"invalid {method} channel")
    channel = raw_channel.strip().casefold()
    if channel not in available_channels:
        raise GatewayError(f"unknown channel: {raw_channel}")
    raw_account = payload.get("accountId", payload.get("account_id"))
    if raw_account is None:
        return channel, DEFAULT_ACCOUNT_ID
    if not isinstance(raw_account, str):
        raise GatewayError(f"invalid {method} accountId")
    account = raw_account.strip() or DEFAULT_ACCOUNT_ID
    return channel, account


def _channel_label(channel: str) -> str:
    return "TUI" if channel == "tui" else channel.replace("_", " ").replace("-", " ").title()


def _channel_detail_label(channel: str) -> str:
    return "Local terminal UI" if channel == "tui" else _channel_label(channel)


def _channel_status_summary(
    default_account: Mapping[str, Any] | None,
    default_account_id: str,
) -> dict[str, Any]:
    if default_account is None:
        return {"configured": False, "accountId": default_account_id}
    summary = {
        "configured": default_account.get("configured") is True,
        "enabled": default_account.get("enabled") is not False,
        "running": default_account.get("running") is True,
        "connected": default_account.get("connected") is True,
        "accountId": default_account_id,
    }
    for key in ("healthState", "lastError", "lastHeartbeat"):
        if default_account.get(key):
            summary[key] = default_account[key]
    return summary


def _now_epoch_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _session_health_key(store: SessionStore, session: SessionInfo) -> str:
    routes = store.list_routes_for_session(session.id)
    return routes[0].route_key if routes else session.id


def _iso_to_epoch_ms(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _age_ms(value: str | None) -> int | None:
    updated_at = _iso_to_epoch_ms(value)
    if updated_at is None:
        return None
    return max(0, _now_epoch_ms() - updated_at)


def _optional_positive_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GatewayError(f"gateway payload field '{key}' must be a positive integer")
    return value


def _session_patch_thinking_level(payload: Mapping[str, Any]) -> str | None:
    thinking_level = _optional_text(payload, "thinkingLevel")
    if thinking_level is None:
        return None

    normalized = thinking_level.casefold()
    if normalized not in SESSION_PATCH_THINKING_LEVELS:
        allowed = ", ".join(sorted(SESSION_PATCH_THINKING_LEVELS))
        raise GatewayError(f"sessions.patch thinking level must be one of: {allowed}")
    return normalized


def _optional_path(payload: Mapping[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise GatewayError(f"gateway payload field '{key}' must be a non-empty path")
    return Path(value).expanduser().resolve()


def _bounded_limit(value: Any, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise GatewayError("Gateway list limit must be an integer.")
    return max(1, min(value, 500))


def _session_list_limit(value: Any) -> tuple[int | None, int | None]:
    if value is None:
        return 100, 100
    if isinstance(value, str) and value.strip().casefold() == "all":
        return None, None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GatewayError("sessions.list limit must be a positive integer or 'all'.")
    bounded = min(value, 500)
    return bounded, bounded


def _openclaw_model_name(provider: str | None, model: str | None) -> str | None:
    if not provider:
        return model
    if not model:
        return provider
    return f"{provider}/{model}"
