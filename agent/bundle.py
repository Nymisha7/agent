from __future__ import annotations

import os
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from .channel_plugins import builtin_plugin_registry


def bundled_rust_binary() -> Path | None:
    executable = "agent-rust.exe" if sys.platform.startswith("win") else "agent-rust"
    try:
        candidate = resources.files("agent").joinpath("bin", executable)
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    try:
        path = Path(os.fspath(candidate))
    except TypeError:
        return None
    return path if path.is_file() else None


def bundle_manifest() -> dict[str, Any]:
    registry = builtin_plugin_registry()
    return {
        "package": "agent",
        "components": {
            "gateway": "agent.gateway",
            "agent_core": "agent.planner",
            "plugin_sdk": "agent.plugin_sdk",
            "llm_layer": "agent.llm",
            "tools": "agent.tools",
            "rust_backend": str(bundled_rust_binary() or ""),
            "skills": "agent.skills",
        },
        "plugins": registry.describe(),
    }
