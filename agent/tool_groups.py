from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolGroup:
    id: str
    label: str
    summary: str
    tools: tuple[str, ...]


TOOL_GROUPS = (
    ToolGroup("workspace", "Workspace", "Resolve, search, inspect, read, write, edit, and delete project files.", ("inspect_target", "glob", "grep", "list_path", "path_status", "inspect_tree", "read_path", "write_file", "edit_file", "delete_path")),
    ToolGroup("code", "Code Intelligence", "Use configured language servers for symbols, definitions, and references.", ("language_server",)),
    ToolGroup("host_inspection", "Host Inspection", "Read current-machine facts and resolve applications or windows before desktop control.", ("system_info", "connected_devices", "process_list", "desktop_capabilities", "desktop_observe", "desktop_resolve")),
    ToolGroup("desktop_control", "Desktop Control", "Launch apps and close observed windows directly; other desktop changes require approval.", ("desktop_action", "desktop_send_message", "desktop_clipboard_files")),
    ToolGroup("system_control", "System Control", "Run narrow allowlisted host commands; service mutations require approval.", ("run_system_command",)),
    ToolGroup("safety", "Safety", "Inspect secrets and credentials without returning secret values.", ("secret_scan",)),
    ToolGroup("skills", "Skills And Subagents", "Load task-specific instructions or fan out independent agents with non-overlapping ownership.", ("load_skill", "parallel_subagents")),
)


def tool_group_for(name: str) -> ToolGroup | None:
    return next((group for group in TOOL_GROUPS if name in group.tools), None)


def grouped_tool_names(names: set[str]) -> list[tuple[ToolGroup, list[str]]]:
    grouped = [(group, [name for name in group.tools if name in names]) for group in TOOL_GROUPS]
    grouped = [(group, present) for group, present in grouped if present]
    seen = {name for _, present in grouped for name in present}
    if other := sorted(names - seen):
        grouped.append((ToolGroup("other", "Other", "Other enabled tools.", tuple(other)), other))
    return grouped
