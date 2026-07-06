from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tools import ToolContext


ToolHandler = Callable[[dict[str, Any], "ToolContext"], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Duplicate tool registered: {spec.name}")
        self._tools[spec.name] = spec

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        try:
            return tool.handler(args, ctx)
        except TimeoutError as exc:
            return {
                "ok": False,
                "tool": name,
                "args": args,
                "blocked": True,
                "recoverable": True,
                "reason": "tool_timeout",
                "error": str(exc),
                "guidance": (
                    "The tool call exceeded its runtime budget. Retry with a narrower path, "
                    "more specific pattern, or smaller limit before expanding the search."
                ),
            }
