from __future__ import annotations

from ..plugin_sdk import PluginManifest, PluginRegistry


def builtin_plugin_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_manifest(
        PluginManifest(
            plugin_id="agent.builtin.channels",
            name="Agent built-in channel plugins",
            channel_plugins=("tui",),
            capabilities=("gateway.channel", "local.tui"),
        )
    )
    registry.register_channel("tui", _create_tui_adapter)
    return registry


def create_builtin_channel_adapters() -> list[object]:
    registry = builtin_plugin_registry()
    return [registry.create_channel(channel_id) for channel_id in registry.channel_ids()]


def _create_tui_adapter() -> object:
    from ..gateway_impl import LocalTuiChannel

    return LocalTuiChannel()
