from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from agent.bundle import bundle_manifest
from agent.channel_plugins import builtin_plugin_registry, create_builtin_channel_adapters
from agent.gateway import ChannelRegistry
from agent.main import resolve_rust_bin


class BundleDistributionTests(unittest.TestCase):
    def test_bundle_manifest_names_every_shipped_runtime_layer(self) -> None:
        manifest = bundle_manifest()
        components = manifest["components"]

        self.assertEqual(components["gateway"], "agent.gateway")
        self.assertEqual(components["agent_core"], "agent.planner")
        self.assertEqual(components["plugin_sdk"], "agent.plugin_sdk")
        self.assertEqual(components["llm_layer"], "agent.llm")
        self.assertEqual(components["tools"], "agent.tools")
        self.assertEqual(components["skills"], "agent.skills")
        self.assertIn("tui", manifest["plugins"]["channels"])

    def test_builtin_channel_plugins_are_registerable_gateway_adapters(self) -> None:
        registry = builtin_plugin_registry()
        channel_registry = ChannelRegistry()

        for adapter in create_builtin_channel_adapters():
            channel_registry.register(adapter)

        self.assertEqual(registry.channel_ids(), ("tui",))
        self.assertEqual(channel_registry.ids(), ("tui",))

    def test_resolve_rust_bin_prefers_packaged_backend_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_bin = Path(tmp) / "agent-rust"
            package_bin.write_text("")

            with patch("agent.main.bundled_rust_binary", return_value=package_bin):
                resolved = resolve_rust_bin(
                    Namespace(rust_bin=None),
                    Path(tmp) / "workspace",
                    repo_root=Path(tmp) / "repo",
                )

        self.assertEqual(resolved, package_bin.resolve())

    def test_pyproject_includes_bundled_assets_and_backend_binary(self) -> None:
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"prompts/*.txt"', pyproject)
        self.assertIn('"skills/*/SKILL.md"', pyproject)
        self.assertIn('"bin/*"', pyproject)
        self.assertIn('"wheel"', pyproject)

    def test_wheel_is_marked_platform_specific(self) -> None:
        setup_py = Path("setup.py").read_text(encoding="utf-8")

        self.assertIn("self.root_is_pure = False", setup_py)
