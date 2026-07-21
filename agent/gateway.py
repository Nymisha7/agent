from __future__ import annotations

import importlib
import os
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import AgentConfig
    from .gateway_impl import ChannelRegistry, InboundAddress, LifecycleHooks, AgentGateway
    from .session_store import SessionStore


_EXPORTED_IMPL_NAMES = {
    "ChannelAdapter",
    "ChannelRegistry",
    "DEFAULT_ACCOUNT_ID",
    "GatewayError",
    "HookResult",
    "InboundAddress",
    "InboundMessage",
    "LifecycleHooks",
    "LocalTuiChannel",
    "AgentGateway",
    "RouteDecision",
    "RoutedSession",
    "canonical_route_key",
}


def _emit_startup_trace(name: str, duration_ms: float, total_ms: float) -> None:
    if not os.environ.get("AGENT_GATEWAY_STARTUP_TRACE"):
        return
    sys.stderr.write(
        f"[gateway] startup trace: {name} {duration_ms:.1f}ms total={total_ms:.1f}ms\n"
    )


def _load_gateway_impl() -> Any:
    startup_started_at = time.perf_counter()
    before = time.perf_counter()
    try:
        return importlib.import_module(".gateway_impl", __package__)
    finally:
        now = time.perf_counter()
        _emit_startup_trace(
            "gateway.impl-import",
            (now - before) * 1000,
            (now - startup_started_at) * 1000,
        )


def start_gateway(
    *,
    config: "AgentConfig",
    store: "SessionStore",
    channels: "ChannelRegistry | None" = None,
    hooks: "LifecycleHooks | None" = None,
) -> "AgentGateway":
    impl = _load_gateway_impl()
    return impl.AgentGateway(config=config, store=store, channels=channels, hooks=hooks)


def create_inbound_address(*args: Any, **kwargs: Any) -> "InboundAddress":
    impl = _load_gateway_impl()
    return impl.InboundAddress(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTED_IMPL_NAMES:
        raise AttributeError(name)
    return getattr(_load_gateway_impl(), name)


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTED_IMPL_NAMES))
