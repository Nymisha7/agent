from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class ChannelAdapter(Protocol):
    channel_id: str

    def normalize(self, payload: Mapping[str, Any]) -> Any:
        ...


ChannelFactory = Callable[[], ChannelAdapter]
_ACTIVE_PLUGIN_REGISTRY: PluginRegistry | None = None


@dataclass(frozen=True)
class PluginManifest:
    plugin_id: str
    name: str
    version: str = "0.1.0"
    bundled: bool = True
    channel_plugins: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()


@dataclass
class PluginRegistry:
    manifests: dict[str, PluginManifest] = field(default_factory=dict)
    channel_factories: dict[str, ChannelFactory] = field(default_factory=dict)

    def register_manifest(self, manifest: PluginManifest) -> None:
        if not manifest.plugin_id:
            raise ValueError("plugin_id cannot be empty")
        if manifest.plugin_id in self.manifests:
            raise ValueError(f"duplicate plugin manifest: {manifest.plugin_id}")
        self.manifests[manifest.plugin_id] = manifest

    def register_channel(self, channel_id: str, factory: ChannelFactory) -> None:
        normalized = channel_id.strip().casefold()
        if not normalized:
            raise ValueError("channel_id cannot be empty")
        if normalized in self.channel_factories:
            raise ValueError(f"duplicate channel plugin: {normalized}")
        self.channel_factories[normalized] = factory

    def channel_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.channel_factories))

    def create_channel(self, channel_id: str) -> ChannelAdapter:
        normalized = channel_id.strip().casefold()
        try:
            return self.channel_factories[normalized]()
        except KeyError as exc:
            raise KeyError(f"unknown channel plugin: {normalized}") from exc

    def describe(self) -> dict[str, Any]:
        return {
            "plugins": [
                {
                    "id": manifest.plugin_id,
                    "name": manifest.name,
                    "version": manifest.version,
                    "bundled": manifest.bundled,
                    "channel_plugins": list(manifest.channel_plugins),
                    "capabilities": list(manifest.capabilities),
                }
                for manifest in sorted(self.manifests.values(), key=lambda item: item.plugin_id)
            ],
            "channels": list(self.channel_ids()),
        }


def pin_active_plugin_registry(registry: PluginRegistry) -> None:
    global _ACTIVE_PLUGIN_REGISTRY
    _ACTIVE_PLUGIN_REGISTRY = registry


def release_pinned_plugin_registry() -> None:
    global _ACTIVE_PLUGIN_REGISTRY
    _ACTIVE_PLUGIN_REGISTRY = None


def active_plugin_registry() -> PluginRegistry | None:
    return _ACTIVE_PLUGIN_REGISTRY
