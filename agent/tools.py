from __future__ import annotations

import fnmatch
import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from .language_servers import LanguageServerManager
from .policy import PolicyEngine
from .project_identity import resolve_workspace_alias
from .rust_tools import RustTools
from .tool_registry import ToolRegistry, ToolSpec

if TYPE_CHECKING:
    from .skills import SkillCatalog


@dataclass
class ToolContext:
    rust: RustTools
    workspace_root: Path
    search_roots: list[Path]
    owned_write_roots: list[Path] | None = None
    approved_external_read_roots: list[Path] = field(default_factory=list)
    approved_external_write_roots: list[Path] = field(default_factory=list)
    approved_external_delete_roots: list[Path] = field(default_factory=list)
    approved_system_commands: list[str] = field(default_factory=list)
    language_servers: LanguageServerManager | None = None
    parallel_runner: Callable[..., Any] | None = None
    skill_catalog: SkillCatalog | None = None


@dataclass(frozen=True)
class IgnoreRule:
    base: Path
    pattern: str
    directory_only: bool = False


@dataclass
class WalkPolicy:
    root: Path
    boundary: Path
    ignore_rules: list[IgnoreRule]
    visited_dirs: set[Path] = field(default_factory=set)


SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".packages",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}

DEEP_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
}

HIDDEN_DIR_ALLOWLIST = {
    ".circleci",
    ".config",
    ".github",
    ".gitlab",
    ".idea",
    ".vscode",
}

SKIP_FILE_NAMES = {
    ".DS_Store",
}

DESKTOP_ACTIONS = frozenset({
    "set_volume", "set_mute", "set_brightness",
    "bluetooth_connect", "bluetooth_disconnect",
    "network_connect", "network_disconnect", "eject_storage",
    "terminate_process", "launch_application", "open_path",
    "open_path_in_application", "open_url",
    "focus_window", "minimize_window", "maximize_window", "restore_window",
    "close_window", "clipboard_write", "send_key", "type_text", "mouse_click",
    "scroll", "focus_element", "invoke_element", "set_field_text",
})
DESKTOP_TARGET_REQUIRED_ACTIONS = frozenset({
    "bluetooth_connect", "bluetooth_disconnect", "network_connect",
    "network_disconnect", "eject_storage", "terminate_process",
    "launch_application", "open_path", "open_path_in_application", "open_url",
    "focus_window",
    "minimize_window", "maximize_window", "restore_window", "close_window",
    "mouse_click", "focus_element", "invoke_element", "set_field_text",
})
DESKTOP_VALUE_REQUIRED_ACTIONS = frozenset({
    "set_volume", "set_brightness", "clipboard_write", "send_key", "type_text",
    "scroll", "invoke_element", "set_field_text",
    "open_path_in_application",
})
DESKTOP_ACTIONS_WITHOUT_APPROVAL = frozenset({
    "launch_application", "open_path", "open_path_in_application", "focus_window",
    "close_window",
})


def desktop_action_requires_approval(action: str) -> bool:
    return action not in DESKTOP_ACTIONS_WITHOUT_APPROVAL


TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_CONTEXT_FILE_NAMES = {
    ".goosehints",
    "AGENTS.md",
}


def build_tool_registry(ctx: ToolContext) -> ToolRegistry:
    registry = ToolRegistry()
    register_rust_file_tools(registry, ctx)
    if ctx.skill_catalog is not None and ctx.skill_catalog.skills:
        registry.register(
            ToolSpec(
                name="load_skill",
                handler=_load_skill,
                schema=_function_schema(
                    name="load_skill",
                    description=(
                        "Load the full instructions for one available Agent skill when its "
                        "catalog description matches the current task. Skills cannot grant "
                        "tools, bypass approvals, or override core safety policy."
                    ),
                    properties={
                        "name": {
                            "type": "string",
                            "enum": list(ctx.skill_catalog.names()),
                            "description": "Exact skill name from the available-skills catalog.",
                        },
                    },
                    required=["name"],
                ),
            )
        )
    if ctx.parallel_runner is not None:
        registry.register(
            ToolSpec(
                name="parallel_subagents",
                handler=_parallel_subagents,
                schema=_function_schema(
                    name="parallel_subagents",
                    description=(
                        "Proactively delegate one concurrent batch during the normal agent turn when a "
                        "complex request has at least two independent deliverables, repository areas, or "
                        "verbose investigations that can proceed without waiting on one another. "
                        "Each child has a fresh model/tool loop, can read the workspace, and may write or edit "
                        "only inside its declared non-overlapping owns directories. Omit owns for a read-only "
                        "research task. Use delegation to reduce parent-context noise and allocate safe disjoint "
                        "implementation work; do not create filler tasks or dependent phases. Children cannot delete, run arbitrary "
                        "commands, control the desktop, send messages, request approval, or spawn agents. "
                        "The parent integrates cross-cutting files, verifies the batch, and gives the final answer."
                    ),
                    properties={
                        "tasks": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 8,
                            "description": (
                                "Independent bounded tasks that can safely execute at the same time."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "Unique stable label for this workstream.",
                                    },
                                    "task": {
                                        "type": "string",
                                        "description": (
                                            "Specific implementation, investigation, or review task "
                                            "that this child can finish independently."
                                        ),
                                    },
                                    "owns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "maxItems": 8,
                                        "description": (
                                            "Workspace-relative directories this child exclusively owns for "
                                            "write/edit operations. Ownership must not overlap another task. "
                                            "Omit or use an empty array for read-only work."
                                        ),
                                    },
                                },
                                "required": ["id", "task"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    required=["tasks"],
                ),
            )
        )
    return registry


def _load_skill(args: dict[str, Any], ctx: ToolContext) -> Any:
    if ctx.skill_catalog is None:
        return {
            "ok": False,
            "tool": "load_skill",
            "blocked": True,
            "reason": "skill_catalog_unavailable",
        }
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("load_skill requires non-empty string arg 'name'.")
    return ctx.skill_catalog.load(name)


def _parallel_subagents(args: dict[str, Any], ctx: ToolContext) -> Any:
    if ctx.parallel_runner is None:
        return {
            "ok": False,
            "tool": "parallel_subagents",
            "blocked": True,
            "reason": "parallel_subagents_unavailable",
        }
    tasks = args.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("parallel_subagents requires array arg 'tasks'.")
    return ctx.parallel_runner(tasks=tasks)


def register_rust_file_tools(registry: ToolRegistry, _ctx: ToolContext) -> None:
    registry.register(
        ToolSpec(
            name="language_server",
            handler=_language_server,
            default_enabled=False,
            schema=_function_schema(
                name="language_server",
                description=(
                    "Check, start, stop, or query configured language servers for the current workspace. "
                    "Supports workspace symbols, document symbols, definitions, and references."
                ),
                properties={
                    "action": {
                        "type": "string",
                        "enum": [
                            "status",
                            "start",
                            "stop",
                            "initialize",
                            "workspace_symbol",
                            "document_symbol",
                            "definition",
                            "references",
                        ],
                        "description": "Language-server lifecycle or code-intelligence action.",
                        "default": "status",
                    },
                    "server": {
                        "type": "string",
                        "description": (
                            "Optional server, language, or command name such as pyright, Python, "
                            "clangd, tsserver, gopls, rust-analyzer, or eclipse.jdt.ls."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": "Source file path for document_symbol, definition, or references.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Workspace symbol search query.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based source line for definition or references.",
                    },
                    "character": {
                        "type": "integer",
                        "description": "1-based source character for definition or references.",
                    },
                    "include_declaration": {
                        "type": "boolean",
                        "description": "Include the declaration location in references results.",
                        "default": True,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum returned symbols or locations.",
                        "default": 50,
                    },
                },
                required=["action"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="glob",
            handler=_glob,
            schema=_function_schema(
                name="glob",
                description=(
                    "Discover files and directories by glob pattern. Use *, **, and ? "
                    "for exact, case-sensitive path matching; use **/ for recursive "
                    "matching. If no path is given, the workspace root is used as the "
                    "search root."
                ),
                properties={
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match against paths.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional search root. Relative paths are resolved from the "
                            "workspace root. Defaults to the workspace root."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 20,
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["any", "file", "directory"],
                        "description": "Restrict results by resource type.",
                        "default": "any",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files and directories.",
                        "default": False,
                    },
                    "include_generated": {
                        "type": "boolean",
                        "description": (
                            "Include generated/dependency directories such as node_modules, "
                            ".venv, target, build, and dist. Defaults to false."
                        ),
                        "default": False,
                    },
                },
                required=["pattern"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="grep",
            handler=_grep,
            schema=_function_schema(
                name="grep",
                description=(
                    "Search file contents. Use this for text or code search, not for path discovery."
                ),
                properties={
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for in file contents.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional search root. Relative paths are resolved from the "
                            "workspace root. Defaults to the workspace root."
                        ),
                    },
                    "include": {
                        "type": "string",
                        "description": "Optional glob filter for limiting searched files.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return.",
                        "default": 20,
                    },
                    "literal_text": {
                        "type": "boolean",
                        "description": "Treat the search pattern as literal text.",
                        "default": False,
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files and directories.",
                        "default": False,
                    },
                },
                required=["pattern"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="list_path",
            handler=_list_path,
            schema=_function_schema(
                name="list_path",
                description=(
                    "List a directory or inspect a file path explicitly. Use this when the "
                    "user already gave a path or when you need a direct listing."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": "Absolute path, explicit relative path, or a path chosen from glob results.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based starting line for text files.",
                        "default": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of text lines or directory entries. Use 300-500 for source files.",
                        "default": 400,
                    },
                },
                required=["path"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="path_status",
            handler=_path_status,
            schema=_function_schema(
                name="path_status",
                description=(
                    "Verify whether a file or directory exists without changing it. For a "
                    "directory, report whether it is empty. When the target is inside a Git "
                    "working tree, also report tracked files Git sees as deleted. Use this for "
                    "questions such as 'has this project been deleted?' or 'does this still exist?'."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path whose current state should be verified.",
                    },
                },
                required=["path"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="inspect_target",
            handler=_inspect_target,
            schema=_function_schema(
                name="inspect_target",
                description=(
                    "Resolve an exact file or directory path or return ranked candidates. "
                    "Use this when the user already gave a concrete path or target name "
                    "and you need path resolution before reading."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute path, explicit relative path, or a path returned by glob."
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["any", "file", "directory"],
                        "description": "Expected resource type.",
                        "default": "any",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based starting line for text files.",
                        "default": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of text lines or directory entries. Use 300-500 for source files.",
                        "default": 400,
                    },
                },
                required=["path"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="inspect_tree",
            handler=_inspect_tree,
            schema=_function_schema(
                name="inspect_tree",
                description=(
                    "Recursively inventory a directory or project and read bounded text "
                    "files that actually exist. Use this for requests to understand, "
                    "summarize, explain, or inspect all files in a directory/project. "
                    "Resolve a named target with inspect_target first; do not scan a "
                    "workspace-container root unless the user explicitly requested it."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to inspect.",
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Maximum number of readable text files to include.",
                        "default": 25,
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum number of file/directory metadata entries to inventory.",
                        "default": 1000,
                    },
                    "max_bytes_per_file": {
                        "type": "integer",
                        "description": "Maximum bytes to read from each text file.",
                        "default": 12000,
                    },
                    "max_total_bytes": {
                        "type": "integer",
                        "description": "Maximum combined bytes of file content to include.",
                        "default": 80000,
                    },
                    "allow_broad_workspace": {
                        "type": "boolean",
                        "description": (
                            "Set true only when the user explicitly asked to recursively inspect "
                            "the entire workspace-container root."
                        ),
                        "default": False,
                    },
                },
                required=["path"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="read_path",
            handler=_read_path,
            schema=_function_schema(
                name="read_path",
                description=(
                    "Read a file by explicit path. Use this after path resolution for exact "
                    "file paths and for paths returned by glob, list_path, or inspect_target."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": "Absolute path, explicit relative path, or a path returned by glob.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based starting line for text files.",
                        "default": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of text lines. Use 300-500 for source files, paginate with offset if needed.",
                        "default": 400,
                    },
                },
                required=["path"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="secret_scan",
            handler=_secret_scan,
            schema=_function_schema(
                name="secret_scan",
                description=(
                    "Scan a workspace or directory for secrets, credentials, and token-like values. "
                    "Returns redacted findings only."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional directory to scan. Relative paths are resolved from the workspace root. "
                            "Defaults to the workspace root."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of files to scan before stopping.",
                        "default": 150,
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files and directories in the scan.",
                        "default": True,
                    },
                    "include_generated": {
                        "type": "boolean",
                        "description": "Include generated/dependency directories such as node_modules, .venv, target, build, and dist.",
                        "default": False,
                    },
                },
                required=[],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="system_info",
            handler=_system_info,
            schema=_function_schema(
                name="system_info",
                description=(
                    "Inspect the current runtime host and return structured operating-system "
                    "details such as hostname, uptime, CPU count, memory, disk summary, and "
                    "whether the agent is running under WSL."
                ),
                properties={},
                required=[],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="connected_devices",
            handler=_connected_devices,
            schema=_function_schema(
                name="connected_devices",
                description=(
                    "Inspect live devices visible to the current runtime. Returns status-rich "
                    "records and per-category availability for USB, storage, network, input, "
                    "Bluetooth, audio, displays, cameras, printers, and power devices. Use this "
                    "for natural questions about what is connected to this computer."
                ),
                properties={
                    "scope": {
                        "type": "string",
                        "description": (
                            "Optional discovered category to focus on, such as usb, bluetooth, "
                            "network, or other. Omit it to return every runtime category."
                        ),
                        "default": "all",
                    },
                },
                required=[],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="desktop_capabilities",
            handler=_desktop_capabilities,
            schema=_function_schema(
                name="desktop_capabilities",
                description=(
                    "Report which desktop actions are supported by the current runtime and "
                    "which local backends are available. Use this before attempting an "
                    "uncertain desktop action, or when the user asks what desktop operations "
                    "Agent can perform on this machine."
                ),
                properties={},
                required=[],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="desktop_observe",
            handler=_desktop_observe,
            schema=_function_schema(
                name="desktop_observe",
                description=(
                    "Read the current desktop state visible to this runtime: installed "
                    "applications, visible windows, active window, connected displays, and audio state when supported. "
                    "This is read-only and reports backend limitations instead of changing state."
                ),
                properties={
                    "scope": {
                        "type": "string",
                        "enum": ["all", "applications", "windows", "active_window", "clipboard", "ui_tree", "displays", "audio", "dialogs", "downloads"],
                        "description": "Desktop observation scope to return.",
                        "default": "all",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of applications or windows to return.",
                        "default": 50,
                    },
                },
                required=[],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="desktop_resolve",
            handler=_desktop_resolve,
            schema=_function_schema(
                name="desktop_resolve",
                description=(
                    "Resolve a natural desktop target name to concrete observed applications "
                    "or windows for read-only discovery, ambiguity handling, or an exact window "
                    "target. Application launches resolve names within desktop_action and do not "
                    "need this separate call."
                ),
                properties={
                    "query": {
                        "type": "string",
                        "description": "User-provided app/window name or title fragment to resolve.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["any", "application", "window"],
                        "description": "Target kind to search.",
                        "default": "any",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of candidates to return.",
                        "default": 10,
                    },
                },
                required=["query"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="process_list",
            handler=_process_list,
            schema=_function_schema(
                name="process_list",
                description=(
                    "List running processes visible to the current runtime, sorted by CPU or "
                    "memory usage."
                ),
                properties={
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of processes to return.",
                        "default": 20,
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["cpu", "memory"],
                        "description": "Sort order for the process list.",
                        "default": "cpu",
                    },
                },
                required=[],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="run_system_command",
            handler=_run_system_command,
            schema=_function_schema(
                name="run_system_command",
                description=(
                    "Run a narrow allowlisted system command. Read-only commands run "
                    "immediately. Commands that start, stop, or restart services require "
                    "explicit approval."
                ),
                properties={
                    "command": {
                        "type": "string",
                        "enum": [
                            "list_block_devices",
                            "list_network_interfaces",
                            "list_listening_ports",
                            "service_status",
                            "start_service",
                            "stop_service",
                            "restart_service",
                        ],
                        "description": "Allowlisted system command identifier.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional target such as a service name.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows/lines to return for list commands.",
                        "default": 50,
                    },
                },
                required=["command"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="desktop_action",
            handler=_desktop_action,
            schema=_function_schema(
                name="desktop_action",
                description=(
                    "Perform one narrow desktop or device action. Launching an application, opening a "
                    "local path, focusing an observed window, and closing an observed window run "
                    "immediately; other actions require explicit user approval. "
                    "Never use this for read-only questions; use connected_devices, system_info, "
                    "or process_list instead. Use open_path for a local path's default handler, "
                    "open_path_in_application for a local path in the executable named by value, "
                    "and open_url only for HTTP(S). open_path_in_application launches the selected "
                    "application when needed; do not call launch_application first. Results include "
                    "execution verification."
                ),
                properties={
                    "action": {
                        "type": "string",
                        "enum": sorted(DESKTOP_ACTIONS),
                        "description": "Typed desktop operation to perform.",
                    },
                    "target": {
                        "type": "string",
                        "description": (
                            "Concrete target when required: device address, interface, /dev path, "
                            "PID, application name or ID, local path, HTTP(S) URL, window id from "
                            "desktop_observe, or x,y screen coordinates for mouse_click. Application "
                            "names are resolved to installed targets before Rust execution."
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "Action value, such as an application executable for "
                            "open_path_in_application, volume/brightness percent, "
                            "true/false/toggle, clipboard text, key spec, or text to type."
                        ),
                    },
                },
                required=["action"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="desktop_send_message",
            handler=_desktop_send_message,
            schema=_function_schema(
                name="desktop_send_message",
                description=(
                    "Focus an observed desktop window, type an exact user-provided message, "
                    "and optionally submit it with Enter or Ctrl+Enter after explicit approval. "
                    "Use this only when the user asked to send text through an already open app."
                ),
                properties={
                    "target": {
                        "type": "string",
                        "description": "Observed window id from desktop_observe or desktop_resolve.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Exact message text to type. The receipt redacts content.",
                    },
                    "submit": {
                        "type": "string",
                        "enum": ["enter", "ctrl+enter", "none"],
                        "default": "enter",
                        "description": "Whether to press a submit key after typing.",
                    },
                },
                required=["target", "message"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="desktop_clipboard_files",
            handler=_desktop_clipboard_files,
            schema=_function_schema(
                name="desktop_clipboard_files",
                description="Place existing local files or folders on the desktop clipboard without reading their contents.",
                properties={
                    "paths": {
                        "type": "array", "items": {"type": "string"},
                        "minItems": 1, "maxItems": 32,
                    },
                    "operation": {
                        "type": "string", "enum": ["copy", "cut"], "default": "copy",
                    },
                },
                required=["paths"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="write_file",
            handler=_write_file,
            schema=_function_schema(
                name="write_file",
                description=(
                    "Create or overwrite a text file inside the workspace. Use this when "
                    "the user asks to save, create, make, build, generate, or update a "
                    "file, script, app, project, or software."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path for the file to create or overwrite.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full file contents to write.",
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if they do not exist.",
                        "default": True,
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite an existing file if it already exists.",
                        "default": True,
                    },
                    "preserve_line_endings": {
                        "type": "boolean",
                        "description": "Preserve existing line endings when overwriting.",
                        "default": True,
                    },
                    "expected_sha256": {
                        "type": "string",
                        "description": "Optional expected sha256 hash to guard against concurrent edits.",
                    },
                },
                required=["path", "content"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="edit_file",
            handler=_edit_file,
            schema=_function_schema(
                name="edit_file",
                description=(
                    "Edit an existing text file inside the workspace by replacing exact text. "
                    "By default the text must appear exactly once."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path for the file to edit.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact existing text to replace.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_text.",
                        "default": False,
                    },
                    "expected_sha256": {
                        "type": "string",
                        "description": "Optional expected sha256 hash to guard against concurrent edits.",
                    },
                },
                required=["path", "old_text", "new_text"],
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="delete_path",
            handler=_delete_path,
            schema=_function_schema(
                name="delete_path",
                description=(
                    "Delete an existing file or directory inside the workspace. Directories "
                    "require recursive=true when not empty. The runtime requires one structured "
                    "approval for the exact resolved target before execution."
                ),
                properties={
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path for the file or directory to delete.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Recursively delete directories.",
                        "default": False,
                    },
                },
                required=["path"],
            ),
        )
    )

def _function_schema(
    *,
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def _language_server(args: dict[str, Any], ctx: ToolContext) -> Any:
    action = _enum_arg(
        args.get("action"),
        default="status",
        allowed={
            "status",
            "start",
            "stop",
            "initialize",
            "workspace_symbol",
            "document_symbol",
            "definition",
            "references",
        },
    )
    server_value = args.get("server")
    if server_value is not None and (not isinstance(server_value, str) or not server_value.strip()):
        raise ValueError("server must be a non-empty string when provided.")
    server = server_value.strip() if isinstance(server_value, str) else None
    manager = ctx.language_servers or LanguageServerManager()

    if action == "status":
        return manager.status(server)
    if action == "start":
        if not server:
            return _language_server_missing_server(action)
        return manager.start(server, ctx.workspace_root)
    if action == "stop":
        return manager.stop(server)
    if action == "initialize":
        if not server:
            return _language_server_missing_server(action)
        return manager.initialize(server, ctx.workspace_root)
    if action == "workspace_symbol":
        if not server:
            return _language_server_missing_server(action)
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("workspace_symbol requires non-empty string arg 'query'.")
        limit = _int_arg(args.get("limit"), default=50, minimum=1, maximum=500)
        return manager.workspace_symbols(server, query.strip(), ctx.workspace_root, limit=limit)
    if action == "document_symbol":
        if not server:
            return _language_server_missing_server(action)
        path_or_observation = _language_server_source_path(args, ctx)
        if isinstance(path_or_observation, dict):
            return path_or_observation
        limit = _int_arg(args.get("limit"), default=100, minimum=1, maximum=1000)
        return manager.document_symbols(server, path_or_observation, ctx.workspace_root, limit=limit)
    if action == "definition":
        if not server:
            return _language_server_missing_server(action)
        path_or_observation = _language_server_source_path(args, ctx)
        if isinstance(path_or_observation, dict):
            return path_or_observation
        line = _int_arg(args.get("line"), default=1, minimum=1, maximum=1_000_000)
        character = _int_arg(args.get("character"), default=1, minimum=1, maximum=1_000_000)
        limit = _int_arg(args.get("limit"), default=20, minimum=1, maximum=500)
        return manager.definition(
            server,
            path_or_observation,
            ctx.workspace_root,
            line=line,
            character=character,
            limit=limit,
        )
    if action == "references":
        if not server:
            return _language_server_missing_server(action)
        path_or_observation = _language_server_source_path(args, ctx)
        if isinstance(path_or_observation, dict):
            return path_or_observation
        line = _int_arg(args.get("line"), default=1, minimum=1, maximum=1_000_000)
        character = _int_arg(args.get("character"), default=1, minimum=1, maximum=1_000_000)
        limit = _int_arg(args.get("limit"), default=50, minimum=1, maximum=1000)
        include_declaration = _bool_arg(args.get("include_declaration"), default=True)
        return manager.references(
            server,
            path_or_observation,
            ctx.workspace_root,
            line=line,
            character=character,
            include_declaration=include_declaration,
            limit=limit,
        )
    raise ValueError(f"Unsupported language_server action: {action}")


def _language_server_missing_server(action: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": "language_server",
        "blocked": True,
        "recoverable": True,
        "reason": "language_server_required",
        "guidance": f"`language_server {action}` requires a specific server or language name.",
    }


def _language_server_source_path(args: dict[str, Any], ctx: ToolContext) -> Path | dict[str, Any]:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("language_server action requires non-empty string arg 'path'.")
    return _resolve_existing_path_or_target(
        raw_path,
        ctx,
        kind="file",
        tool="language_server",
    )


def _resolve_tool_path(
    raw_path: str,
    ctx: ToolContext,
    *,
    tool: str,
    operation: str,
) -> Path | dict[str, Any]:
    alias_path = _workspace_alias_path(raw_path, ctx.workspace_root)
    if alias_path is not None:
        return _enforce_owned_mutation_scope(
            alias_path,
            ctx,
            tool=tool,
            operation=operation,
            requested_path=raw_path,
        )
    if _looks_like_windows_drive_path(raw_path):
        return _resolve_windows_tool_path(raw_path, ctx, tool=tool, operation=operation)

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        resolved = _ensure_within_workspace(ctx.workspace_root / path, ctx.workspace_root)
        return _enforce_owned_mutation_scope(
            resolved,
            ctx,
            tool=tool,
            operation=operation,
            requested_path=raw_path,
        )

    resolved = _resolve_without_strict(path)
    if _path_is_within_or_equal(resolved, ctx.workspace_root.expanduser().resolve()):
        workspace_path = _ensure_within_workspace(resolved, ctx.workspace_root)
        return _enforce_owned_mutation_scope(
            workspace_path,
            ctx,
            tool=tool,
            operation=operation,
            requested_path=raw_path,
        )

    if _is_broad_external_path(resolved):
        return _external_path_observation(
            tool=tool,
            operation=operation,
            reason="broad_external_path_blocked",
            requested_path=raw_path,
            resolved_path=str(resolved),
            guidance=(
                "This path is outside the workspace and too broad for an agentic file search. "
                "Use a narrower directory and approve that exact external path before retrying."
            ),
        )

    if _external_path_is_approved(resolved, ctx, operation):
        return resolved

    if operation == "delete":
        return _external_path_observation(
            tool=tool,
            operation=operation,
            reason="external_delete_requires_confirmation",
            requested_path=raw_path,
            resolved_path=str(resolved),
            guidance=(
                "Deleting outside the workspace requires an explicit delete approval for "
                "this external path after the exact target has been discovered."
            ),
        )

    return _external_path_observation(
        tool=tool,
        operation=operation,
        reason="external_path_requires_approval",
        requested_path=raw_path,
        resolved_path=str(resolved),
        guidance=(
            "This path is outside the workspace. Ask the user to approve this external "
            "path for the requested operation before retrying."
        ),
    )


def _enforce_owned_mutation_scope(
    path: Path,
    ctx: ToolContext,
    *,
    tool: str,
    operation: str,
    requested_path: str,
) -> Path | dict[str, Any]:
    if operation not in {"write", "delete"} or ctx.owned_write_roots is None:
        return path
    resolved = _resolve_without_strict(path)
    owned_roots = [_resolve_without_strict(root) for root in ctx.owned_write_roots]
    if any(_path_is_within_or_equal(resolved, root) for root in owned_roots):
        return path
    return {
        "ok": False,
        "tool": tool,
        "blocked": True,
        "recoverable": True,
        "reason": "subagent_write_scope_violation",
        "operation": operation,
        "requested_path": requested_path,
        "resolved_path": str(resolved),
        "owned_write_roots": [str(root) for root in owned_roots],
        "guidance": (
            "This independent agent may mutate only its assigned ownership scopes. "
            "Keep the change inside one owned path or return the cross-scope requirement "
            "to the parent coordinator."
        ),
    }


def _resolve_windows_tool_path(
    raw_path: str,
    ctx: ToolContext,
    *,
    tool: str,
    operation: str,
) -> Path | dict[str, Any]:
    translated = _translate_windows_path(raw_path)
    if translated is None:
        return _external_path_observation(
            tool=tool,
            operation=operation,
            reason="windows_path_unavailable_from_current_runtime",
            requested_path=raw_path,
            guidance=(
                "This looks like a Windows path, but Agent is running in a Linux/Ubuntu "
                "runtime that cannot currently access it. Provide a Linux-accessible "
                "mount path, or run Agent from WSL with the Windows drive mounted."
            ),
        )

    translated = _resolve_without_strict(translated)
    if _is_broad_windows_path(raw_path) or _is_broad_external_path(translated):
        return _external_path_observation(
            tool=tool,
            operation=operation,
            reason="external_windows_path_requires_approval",
            requested_path=raw_path,
            translated_path=str(translated),
            broad_path=True,
            guidance=(
                "This Windows path maps outside the workspace and is too broad. Ask the "
                "user for a narrower path, such as a specific Desktop folder, then request "
                "approval before searching it."
            ),
        )

    if _external_path_is_approved(translated, ctx, operation):
        return translated

    return _external_path_observation(
        tool=tool,
        operation=operation,
        reason="external_windows_path_requires_approval",
        requested_path=raw_path,
        translated_path=str(translated),
        guidance=(
            "This Windows path maps outside the workspace. Ask the user to approve access "
            "to the translated path for this operation before retrying."
        ),
    )


def _external_path_observation(
    *,
    tool: str,
    operation: str,
    reason: str,
    requested_path: str,
    guidance: str,
    resolved_path: str | None = None,
    translated_path: str | None = None,
    broad_path: bool = False,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "ok": False,
        "tool": tool,
        "blocked": True,
        "recoverable": True,
        "reason": reason,
        "operation": operation,
        "requested_path": requested_path,
        "guidance": guidance,
    }
    if resolved_path is not None:
        observation["resolved_path"] = resolved_path
    if translated_path is not None:
        observation["translated_path"] = translated_path
    if broad_path:
        observation["broad_path"] = True
    return observation


def _external_path_is_approved(path: Path, ctx: ToolContext, operation: str) -> bool:
    roots = {
        "read": ctx.approved_external_read_roots,
        "write": ctx.approved_external_write_roots,
        "delete": ctx.approved_external_delete_roots,
    }.get(operation, [])
    return any(_path_is_within_or_equal(path, _resolve_without_strict(root)) for root in roots)


def _workspace_root_for_tool_path(path: Path, ctx: ToolContext, operation: str) -> Path:
    workspace_root = ctx.workspace_root.expanduser().resolve()
    if operation in {"write", "delete"} and ctx.owned_write_roots is not None:
        owned_roots = [_resolve_without_strict(root) for root in ctx.owned_write_roots]
        containing = [root for root in owned_roots if _path_is_within_or_equal(path, root)]
        if containing:
            return max(containing, key=lambda root: len(root.parts))
    if _path_is_within_or_equal(path, workspace_root):
        return workspace_root

    roots = {
        "read": ctx.approved_external_read_roots,
        "write": ctx.approved_external_write_roots,
        "delete": ctx.approved_external_delete_roots,
    }.get(operation, [])
    approved_roots = [_resolve_without_strict(root) for root in roots]
    containing = [root for root in approved_roots if _path_is_within_or_equal(path, root)]
    if not containing:
        return workspace_root
    return max(containing, key=lambda root: len(root.parts))


def _looks_like_windows_drive_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value.strip()))


def _translate_windows_path(value: str) -> Path | None:
    if not _is_wsl_runtime():
        return None
    try:
        completed = subprocess.run(
            ["wslpath", "-u", value],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    translated = completed.stdout.strip()
    if not translated:
        return None
    translated_path = Path(translated)
    drive_mount = Path("/mnt") / value[0].lower()
    if translated_path.is_absolute() and not drive_mount.exists():
        return None
    return translated_path


def _is_wsl_runtime() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(errors="ignore").casefold()
    except OSError:
        return False


def _is_broad_windows_path(value: str) -> bool:
    normalized = value.strip().replace("\\", "/").rstrip("/").casefold()
    if re.fullmatch(r"[a-z]:", normalized):
        return True
    return bool(re.fullmatch(r"[a-z]:/users", normalized))


def _is_broad_external_path(path: Path) -> bool:
    resolved = _resolve_without_strict(path)
    if not resolved.is_absolute():
        return False

    filesystem_root = Path(resolved.anchor).resolve(strict=False)
    if resolved == filesystem_root or resolved.parent == filesystem_root:
        return True

    home = Path.home().expanduser().resolve(strict=False)
    if resolved in {home, home.parent}:
        return True

    # Mount roots and their immediate children represent an entire volume or
    # a broad top-level scope. This covers WSL drives and removable media
    # without naming specific drive letters or directory names.
    if os.path.ismount(resolved) or os.path.ismount(resolved.parent):
        return True

    configured = os.environ.get("AGENT_PROTECTED_PATHS", "")
    return any(
        resolved == _resolve_without_strict(Path(value.strip()))
        for value in configured.split(os.pathsep)
        if value.strip()
    )


def _resolve_without_strict(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _glob(args: dict[str, Any], ctx: ToolContext) -> Any:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("glob requires non-empty string arg 'pattern'.")

    root_value = args.get("path")
    root_or_observation = _resolve_glob_root(root_value, ctx)
    if isinstance(root_or_observation, dict):
        return root_or_observation
    root = root_or_observation
    limit = _int_arg(args.get("limit"), default=20, minimum=1, maximum=200)
    kind = _enum_arg(
        args.get("kind"),
        default="any",
        allowed={"any", "file", "directory"},
    )
    include_hidden = _bool_arg(args.get("include_hidden"), default=False)
    include_generated = _bool_arg(args.get("include_generated"), default=False)
    return ctx.rust.glob_files(
        pattern=pattern,
        root=root,
        limit=limit,
        include_hidden=include_hidden,
        include_generated=include_generated,
        kind=kind,
    )


def _resolve_glob_root(root_value: Any, ctx: ToolContext) -> Path | dict[str, Any]:
    if not isinstance(root_value, str) or not root_value.strip():
        return ctx.workspace_root

    root_or_observation = _resolve_tool_path(
        root_value,
        ctx,
        tool="glob",
        operation="read",
    )
    if isinstance(root_or_observation, dict):
        return root_or_observation
    root = root_or_observation
    if root.exists():
        return root

    if not _path_is_within_or_equal(root, ctx.workspace_root.expanduser().resolve()):
        return root

    inspect_target = getattr(ctx.rust, "inspect_target", None)
    if not callable(inspect_target):
        return root

    resolved = inspect_target(
        path=root_value,
        workspace_root=ctx.workspace_root,
        kind="directory",
        offset=1,
        limit=20,
    )
    if not isinstance(resolved, dict):
        return root

    if resolved.get("status") == "resolved":
        target = resolved.get("target")
        if isinstance(target, dict):
            path = target.get("path")
            if isinstance(path, str) and path:
                return Path(path)

    if resolved.get("status") == "candidates":
        scoped_path = _single_candidate_scope_path(
            resolved.get("candidates"),
            workspace_root=ctx.workspace_root,
        )
        if scoped_path is not None:
            return scoped_path
        return {
            "ok": False,
            "tool": "glob",
            "recoverable": True,
            "reason": "glob_root_ambiguous",
            "query": root_value,
            "candidates": resolved.get("candidates", []),
            "guidance": (
                "The glob root is not an existing path and resolved to multiple candidate "
                "directories. Select exactly one candidate as the search root before globbing."
            ),
        }

    return root


def _resolve_existing_path_or_target(
    raw_path: str,
    ctx: ToolContext,
    *,
    kind: str,
    tool: str,
) -> Path | dict[str, Any]:
    alias_path = _workspace_alias_path(raw_path, ctx.workspace_root)
    if alias_path is not None:
        return alias_path
    path_or_observation = _resolve_tool_path(
        raw_path,
        ctx,
        tool=tool,
        operation="read",
    )
    if isinstance(path_or_observation, dict):
        return path_or_observation
    path = path_or_observation
    if path.exists():
        return path

    if not _path_is_within_or_equal(path, ctx.workspace_root.expanduser().resolve()):
        return path

    inspect_target = getattr(ctx.rust, "inspect_target", None)
    if not callable(inspect_target):
        return path

    resolved = inspect_target(
        path=raw_path,
        workspace_root=ctx.workspace_root,
        kind=kind,
        offset=1,
        limit=20,
    )
    if not isinstance(resolved, dict):
        return path

    if resolved.get("status") == "resolved":
        target = resolved.get("target")
        if isinstance(target, dict):
            target_path = target.get("path")
            if isinstance(target_path, str) and target_path:
                return Path(target_path)

    if resolved.get("status") == "candidates":
        scoped_path = _single_candidate_scope_path(
            resolved.get("candidates"),
            workspace_root=ctx.workspace_root,
        )
        if scoped_path is not None:
            return scoped_path
        return {
            "ok": False,
            "tool": tool,
            "recoverable": True,
            "reason": "target_ambiguous",
            "query": raw_path,
            "candidates": resolved.get("candidates", []),
            "guidance": (
                "The requested path is not an existing exact path, but it matched multiple "
                "candidate targets. Select exactly one candidate before reading or acting."
            ),
        }

    return path


def _grep(args: dict[str, Any], ctx: ToolContext) -> Any:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("grep requires non-empty string arg 'pattern'.")

    root_value = args.get("path")
    root_or_observation = (
        _resolve_existing_path_or_target(
            root_value,
            ctx,
            kind="directory",
            tool="grep",
        )
        if isinstance(root_value, str) and root_value.strip()
        else ctx.workspace_root
    )
    if isinstance(root_or_observation, dict):
        return root_or_observation
    root = root_or_observation
    include = args.get("include")
    if include is not None and (not isinstance(include, str) or not include.strip()):
        raise ValueError("include must be a non-empty string when provided.")

    limit = _int_arg(args.get("limit"), default=20, minimum=1, maximum=500)
    literal_text = _bool_arg(args.get("literal_text"), default=False)
    include_hidden = _bool_arg(args.get("include_hidden"), default=False)

    return ctx.rust.grep_files(
        pattern=pattern,
        root=root,
        include=include,
        limit=limit,
        literal_text=literal_text,
        include_hidden=include_hidden,
    )


def _list_path(args: dict[str, Any], ctx: ToolContext) -> Any:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("list_path requires non-empty string arg 'path'.")

    path_or_observation = _resolve_existing_path_or_target(
        raw_path,
        ctx,
        kind="any",
        tool="list_path",
    )
    if isinstance(path_or_observation, dict):
        return path_or_observation
    path = path_or_observation
    offset = _int_arg(args.get("offset"), default=1, minimum=1, maximum=1_000_000)
    limit = _int_arg(args.get("limit"), default=400, minimum=1, maximum=2_000)

    return ctx.rust.read_path(path=path, offset=offset, limit=limit)


def _path_status(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path_status requires non-empty string arg 'path'.")

    path_or_observation = _resolve_tool_path(
        raw_path,
        ctx,
        tool="path_status",
        operation="read",
    )
    if isinstance(path_or_observation, dict):
        return path_or_observation
    path = path_or_observation

    exists = path.exists()
    is_symlink = path.is_symlink()
    kind = "missing"
    empty: bool | None = None
    entry_count: int | None = None
    if exists and path.is_dir():
        kind = "directory"
        try:
            entry_count = sum(1 for _ in path.iterdir())
            empty = entry_count == 0
        except OSError:
            entry_count = None
            empty = None
    elif exists and path.is_file():
        kind = "file"
    elif is_symlink:
        kind = "broken_symlink"

    result: dict[str, Any] = {
        "ok": True,
        "tool": "path_status",
        "path": str(path),
        "exists": exists,
        "kind": kind,
        "is_symlink": is_symlink,
    }
    if empty is not None:
        result["empty"] = empty
    if entry_count is not None:
        result["entry_count"] = entry_count

    git_root = _git_root_for(path if exists and path.is_dir() else path.parent)
    git_changes: list[dict[str, str]] = []
    if git_root is not None:
        result["git_root"] = str(git_root)
        try:
            relative = path.relative_to(git_root)
        except ValueError:
            relative = Path(".")
        pathspec = str(relative) if str(relative) else "."
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(git_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                pathspec,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                if len(line) < 4:
                    continue
                status = line[:2]
                changed_path = line[3:]
                if " -> " in changed_path:
                    changed_path = changed_path.split(" -> ", 1)[1]
                git_changes.append({"status": status, "path": changed_path})
        else:
            result["git_error"] = (completed.stderr or "git status failed").strip()

    tracked_deletions = [item for item in git_changes if "D" in item["status"]]
    result["git_change_count"] = len(git_changes)
    result["git_changes"] = git_changes[:25]
    result["git_changes_truncated"] = len(git_changes) > 25
    result["tracked_deleted_count"] = len(tracked_deletions)
    result["tracked_deleted_paths"] = [item["path"] for item in tracked_deletions[:25]]

    if not exists:
        state = "missing"
    elif tracked_deletions and kind == "directory" and empty:
        state = "tracked_content_deleted"
    elif kind == "directory" and empty:
        state = "empty_directory"
    elif tracked_deletions:
        state = "partially_deleted"
    else:
        state = "present"
    result["state"] = state
    return result


def _inspect_target(args: dict[str, Any], ctx: ToolContext) -> Any:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("inspect_target requires non-empty string arg 'path'.")

    kind = _enum_arg(
        args.get("kind"),
        default="any",
        allowed={"any", "file", "directory"},
    )
    offset = _int_arg(args.get("offset"), default=1, minimum=1, maximum=1_000_000)
    limit = _int_arg(args.get("limit"), default=400, minimum=1, maximum=2_000)
    path_arg: str | Path
    if _looks_like_windows_drive_path(raw_path) or Path(raw_path).expanduser().is_absolute():
        path_or_observation = _resolve_tool_path(
            raw_path,
            ctx,
            tool="inspect_target",
            operation="read",
        )
        if isinstance(path_or_observation, dict):
            return path_or_observation
        path_arg = path_or_observation
    else:
        path_arg = _inspect_target_arg(raw_path, ctx.workspace_root)

    result = ctx.rust.inspect_target(
        path=path_arg,
        workspace_root=ctx.workspace_root,
        kind=kind,
        offset=offset,
        limit=limit,
    )
    return _bound_target_candidates(result, path_arg if isinstance(path_arg, str) else raw_path)


def _inspect_tree(args: dict[str, Any], ctx: ToolContext) -> Any:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("inspect_tree requires non-empty string arg 'path'.")

    root_or_observation = _resolve_existing_path_or_target(
        raw_path,
        ctx,
        kind="any",
        tool="inspect_tree",
    )
    if isinstance(root_or_observation, dict):
        return root_or_observation
    root = root_or_observation
    max_files = _int_arg(args.get("max_files"), default=25, minimum=1, maximum=300)
    max_entries = _int_arg(args.get("max_entries"), default=1_000, minimum=10, maximum=5_000)
    max_bytes_per_file = _int_arg(
        args.get("max_bytes_per_file"),
        default=12_000,
        minimum=1_000,
        maximum=200_000,
    )
    max_total_bytes = _int_arg(
        args.get("max_total_bytes"),
        default=80_000,
        minimum=10_000,
        maximum=800_000,
    )

    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")
    if not root.is_file() and not root.is_dir():
        raise ValueError(f"Path is not a regular file or directory: {root}")

    if (
        root.is_dir()
        and _same_path(root, ctx.workspace_root)
        and not _looks_like_project_root(root)
        and not _bool_arg(args.get("allow_broad_workspace"), default=False)
    ):
        direct_children = _directory_children(root, ctx.workspace_root)
        return {
            "ok": False,
            "tool": "inspect_tree",
            "blocked": True,
            "recoverable": True,
            "reason": "workspace_container_requires_target",
            "path": str(root),
            "direct_children": direct_children[:100],
            "direct_children_truncated": len(direct_children) > 100,
            "guidance": (
                "This workspace is a container for multiple projects. Resolve the user's "
                "named project with inspect_target, then inspect only the selected project. "
                "Use allow_broad_workspace=true only for an explicit whole-workspace request."
            ),
        }
    return ctx.rust.inspect_tree(
        path=root,
        workspace_root=ctx.workspace_root,
        max_files=max_files,
        max_entries=max_entries,
        max_bytes_per_file=max_bytes_per_file,
        max_total_bytes=max_total_bytes,
    )


def _secret_scan(args: dict[str, Any], ctx: ToolContext) -> Any:
    root_value = args.get("path")
    root_or_observation = (
        _resolve_existing_path_or_target(
            root_value,
            ctx,
            kind="directory",
            tool="secret_scan",
        )
        if isinstance(root_value, str) and root_value.strip()
        else ctx.workspace_root
    )
    if isinstance(root_or_observation, dict):
        return root_or_observation
    root = root_or_observation

    if not root.exists():
        return {
            "ok": False,
            "tool": "secret_scan",
            "blocked": True,
            "recoverable": True,
            "reason": "path_missing",
            "path": str(root),
            "guidance": "The requested scan root does not exist. Resolve the path first and retry.",
        }
    if not root.is_dir():
        return {
            "ok": False,
            "tool": "secret_scan",
            "blocked": True,
            "recoverable": True,
            "reason": "scan_root_not_directory",
            "path": str(root),
            "guidance": "Secret scans operate on directories. Use a directory path.",
        }

    limit = _int_arg(args.get("limit"), default=150, minimum=1, maximum=1_000)
    include_hidden = _bool_arg(args.get("include_hidden"), default=True)
    include_generated = _bool_arg(args.get("include_generated"), default=False)
    policy = PolicyEngine()
    findings: list[dict[str, Any]] = []
    scanned_files = 0

    project_paths, _tree_truncated = _walk_project(root, max_entries=max(limit * 20, 1_000))
    for path in project_paths:
        if scanned_files >= limit:
            break
        if not path.is_file():
            continue
        if not include_hidden and path.name.startswith("."):
            continue
        if not include_generated and _is_generated_path(path, root=root):
            continue
        if not _looks_readable_text_file(path):
            continue

        scanned_files += 1
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        rel_path = _relative_display(path, ctx.workspace_root)
        findings.extend(policy.scan_text(text, path=rel_path))

    return {
        "ok": True,
        "tool": "secret_scan",
        "path": str(root),
        "file_count": scanned_files,
        "findings": findings[:100],
        "finding_count": len(findings),
        "redacted": True,
    }


def _system_info(_args: dict[str, Any], ctx: ToolContext) -> Any:
    return ctx.rust.system_info()


def _connected_devices(args: dict[str, Any], ctx: ToolContext) -> Any:
    raw_scope = args.get("scope", "all")
    if not isinstance(raw_scope, str) or not raw_scope.strip():
        raise ValueError("connected_devices scope must be a non-empty category string.")
    scope = raw_scope.strip().casefold()
    if len(scope) > 64 or not re.fullmatch(r"[a-z0-9_-]+", scope):
        raise ValueError("connected_devices scope may contain only letters, numbers, hyphens, and underscores.")
    return ctx.rust.connected_devices(scope=scope)


def _desktop_capabilities(_args: dict[str, Any], ctx: ToolContext) -> Any:
    return ctx.rust.desktop_capabilities()


def _desktop_observe(args: dict[str, Any], ctx: ToolContext) -> Any:
    scope = _enum_arg(
        args.get("scope"),
        default="all",
        allowed={"all", "applications", "windows", "active_window", "clipboard", "ui_tree", "displays", "audio", "dialogs", "downloads"},
    )
    limit = _int_arg(args.get("limit"), default=50, minimum=1, maximum=200)
    return ctx.rust.desktop_observe(scope=scope, limit=limit)


def _desktop_resolve(args: dict[str, Any], ctx: ToolContext) -> Any:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("desktop_resolve requires a non-empty query.")
    if len(query) > 200:
        raise ValueError("desktop_resolve query must be 200 characters or fewer.")
    kind = _enum_arg(
        args.get("kind"),
        default="any",
        allowed={"any", "application", "window"},
    )
    limit = _int_arg(args.get("limit"), default=10, minimum=1, maximum=50)
    return ctx.rust.desktop_resolve(query=query.strip(), kind=kind, limit=limit)


def _desktop_action(args: dict[str, Any], ctx: ToolContext) -> Any:
    action = _enum_arg(
        args.get("action"),
        default="",
        allowed=DESKTOP_ACTIONS,
    )
    if not action:
        raise ValueError("desktop_action requires a supported action.")
    target = _optional_desktop_arg(args.get("target"), "target")
    value = _optional_desktop_arg(args.get("value"), "value")
    if action in DESKTOP_TARGET_REQUIRED_ACTIONS and target is None:
        return {
            "ok": False,
            "tool": "desktop_action",
            "blocked": True,
            "recoverable": True,
            "reason": "desktop_action_target_required",
            "operation": "desktop",
            "guidance": f"{action} requires a concrete target before it can run.",
        }
    if (
        action == "open_url"
        and target is not None
        and not target.casefold().startswith(("http://", "https://"))
    ):
        local_target = _desktop_local_path(target, ctx.workspace_root)
        if local_target.exists():
            action = "open_path"
            target = str(local_target.resolve())
        else:
            return {
                "ok": False,
                "tool": "desktop_action",
                "blocked": True,
                "recoverable": True,
                "reason": "desktop_url_target_invalid",
                "operation": "desktop",
                "guidance": (
                    "open_url accepts only HTTP(S). Use open_path for a local path, or "
                    "open_path_in_application with an application executable in value."
                ),
            }
    if action in {"open_path", "open_path_in_application"} and target is not None:
        local_target = _desktop_local_path(target, ctx.workspace_root)
        if not local_target.exists():
            return {
                "ok": False,
                "tool": "desktop_action",
                "blocked": True,
                "recoverable": True,
                "reason": "desktop_path_unavailable",
                "operation": "desktop",
                "target": str(local_target),
                "guidance": (
                    "The local path does not exist. Resolve an existing file or directory first."
                ),
            }
        target = str(local_target.resolve())
    if action in DESKTOP_VALUE_REQUIRED_ACTIONS and value is None:
        return {
            "ok": False,
            "tool": "desktop_action",
            "blocked": True,
            "recoverable": True,
            "reason": "desktop_action_value_required",
            "operation": "desktop",
            "guidance": (
                f"{action} requires an application executable in value."
                if action == "open_path_in_application"
                else f"{action} requires clipboard text."
                if action == "clipboard_write"
                else f"{action} requires a value."
                if action in {"send_key", "type_text", "scroll", "invoke_element", "set_field_text"}
                else f"{action} requires a value between 0 and 100."
            ),
        }
    if action in {"set_volume", "set_brightness"}:
        try:
            numeric_value = int(value or "")
        except ValueError as exc:
            raise ValueError(f"{action} value must be an integer between 0 and 100.") from exc
        if not 0 <= numeric_value <= 100:
            raise ValueError(f"{action} value must be between 0 and 100.")
        value = str(numeric_value)
    if action == "set_mute":
        value = value or "toggle"
        if value.casefold() not in {"true", "false", "toggle"}:
            raise ValueError("set_mute value must be true, false, or toggle.")
        value = value.casefold()

    if desktop_action_requires_approval(action):
        approval_key = _desktop_action_approval_key(action, target, value)
        if approval_key not in ctx.approved_system_commands:
            return {
                "ok": False,
                "tool": "desktop_action",
                "blocked": True,
                "recoverable": True,
                "reason": "desktop_action_requires_approval",
                "operation": "desktop",
                "requested_path": approval_key,
                "guidance": (
                    "This desktop action needs approval. Ask the user to approve this exact "
                    "action before retrying."
                ),
            }
    backend_bus = _optional_desktop_backend_arg(args.get("_backend_bus"), "_backend_bus")
    backend_path = _optional_desktop_backend_arg(args.get("_backend_path"), "_backend_path")
    rust_args: dict[str, Any] = {"action": action, "target": target, "value": value}
    if backend_bus is not None:
        rust_args["backend_bus"] = backend_bus
    if backend_path is not None:
        rust_args["backend_path"] = backend_path
    result = ctx.rust.desktop_action(**rust_args)
    if (
        action == "close_window"
        and isinstance(result, dict)
        and result.get("ok") is True
        and result.get("verified") is not True
    ):
        failed = dict(result)
        failed.update({
            "ok": False,
            "verified": False,
            "recoverable": True,
            "reason": "desktop_action_verification_failed",
            "guidance": (
                "The close command was dispatched, but the window was not confirmed closed. "
                "Re-observe it before reporting success."
            ),
        })
        return failed
    return result


def _desktop_local_path(target: str, workspace_root: Path) -> Path:
    path = Path(target).expanduser()
    return path if path.is_absolute() else workspace_root / path


def _desktop_send_message(args: dict[str, Any], ctx: ToolContext) -> Any:
    target = _optional_desktop_arg(args.get("target"), "target")
    message = _desktop_message_arg(args.get("message"))
    if target is None:
        return {
            "ok": False,
            "tool": "desktop_send_message",
            "blocked": True,
            "recoverable": True,
            "reason": "desktop_action_target_required",
            "operation": "desktop",
            "guidance": "desktop_send_message requires an observed window id.",
        }
    if message is None:
        return {
            "ok": False,
            "tool": "desktop_send_message",
            "blocked": True,
            "recoverable": True,
            "reason": "desktop_action_value_required",
            "operation": "desktop",
            "guidance": "desktop_send_message requires the exact message text to type.",
        }
    message_bytes = message.encode("utf-8")
    if not message.strip():
        raise ValueError("desktop_send_message message must not be empty.")
    if len(message_bytes) > 4000:
        raise ValueError("desktop_send_message message must be 4000 UTF-8 bytes or fewer.")

    submit = _enum_arg(
        args.get("submit"),
        default="enter",
        allowed={"enter", "ctrl+enter", "none"},
    )
    capability_block = _desktop_send_message_capability_block(ctx, submit)
    if capability_block is not None:
        return capability_block

    approval_key = _desktop_send_message_approval_key(target, message, submit)
    if approval_key not in ctx.approved_system_commands:
        return {
            "ok": False,
            "tool": "desktop_send_message",
            "blocked": True,
            "recoverable": True,
            "reason": "desktop_action_requires_approval",
            "operation": "desktop",
            "requested_path": approval_key,
            "guidance": (
                "Ask the user to approve sending this exact redacted message to the "
                "observed window before retrying."
            ),
        }

    steps: list[dict[str, Any]] = []
    focus = ctx.rust.desktop_action(action="focus_window", target=target)
    steps.append(_desktop_message_step("focus_window", focus))
    if not _desktop_step_ok(focus):
        return _desktop_message_result(target, message, submit, steps, ok=False)

    typed = ctx.rust.desktop_action(action="type_text", value=message)
    steps.append(_desktop_message_step("type_text", typed))
    if not _desktop_step_ok(typed):
        return _desktop_message_result(target, message, submit, steps, ok=False)

    if submit != "none":
        key = "Return" if submit == "enter" else "ctrl+Return"
        sent = ctx.rust.desktop_action(action="send_key", value=key)
        steps.append(_desktop_message_step("send_key", sent))
        if not _desktop_step_ok(sent):
            return _desktop_message_result(target, message, submit, steps, ok=False)

    return _desktop_message_result(target, message, submit, steps, ok=True)


def _desktop_send_message_capability_block(ctx: ToolContext, submit: str) -> dict[str, Any] | None:
    try:
        capabilities = ctx.rust.desktop_capabilities()
    except Exception as exc:
        return {
            "ok": False,
            "tool": "desktop_send_message",
            "blocked": True,
            "recoverable": True,
            "reason": "desktop_capability_check_failed",
            "operation": "desktop",
            "guidance": f"Could not confirm desktop messaging support before approval: {exc}",
        }
    actions = capabilities.get("actions") if isinstance(capabilities, dict) else None
    if not isinstance(actions, list):
        return {
            "ok": False,
            "tool": "desktop_send_message",
            "blocked": True,
            "recoverable": True,
            "reason": "desktop_capability_check_failed",
            "operation": "desktop",
            "guidance": "Could not read desktop action capabilities before approval.",
        }
    available = {
        item.get("action")
        for item in actions
        if isinstance(item, dict) and item.get("available") is True
    }
    required = ["focus_window", "type_text"]
    if submit != "none":
        required.append("send_key")
    missing = [action for action in required if action not in available]
    if not missing:
        return None
    runtime = capabilities.get("runtime") if isinstance(capabilities, dict) else None
    return {
        "ok": False,
        "tool": "desktop_send_message",
        "blocked": True,
        "recoverable": True,
        "reason": "desktop_action_dependency_unavailable",
        "operation": "desktop",
        "runtime": runtime,
        "unavailable_actions": missing,
        "guidance": (
            "desktop_send_message requires focus_window, type_text, and send_key support "
            "before approval. Use desktop_capabilities to inspect this runtime."
        ),
    }


def _desktop_message_step(action: str, result: Any) -> dict[str, Any]:
    result_dict = result if isinstance(result, dict) else {}
    return {
        "action": action,
        "ok": result_dict.get("ok") is True,
        "verification": result_dict.get("verification") or "unknown",
        "verified": result_dict.get("verified") is True,
        "error": result_dict.get("error"),
    }


def _desktop_step_ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("ok") is True


def _desktop_message_result(
    target: str,
    message: str,
    submit: str,
    steps: list[dict[str, Any]],
    *,
    ok: bool,
) -> dict[str, Any]:
    encoded = message.encode("utf-8")
    return {
        "ok": ok,
        "tool": "desktop_send_message",
        "target": target,
        "submit": submit,
        "message_receipt": {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
            "chars": len(message),
            "content_returned": False,
        },
        "steps": steps,
        "verified": ok and all(step.get("ok") is True for step in steps),
        "verification": "confirmed" if ok else "not_confirmed",
    }


def _desktop_clipboard_files(args: dict[str, Any], ctx: ToolContext) -> Any:
    raw_paths = args.get("paths")
    if not isinstance(raw_paths, list) or not 1 <= len(raw_paths) <= 32:
        raise ValueError("desktop_clipboard_files requires between 1 and 32 paths.")
    operation = _enum_arg(args.get("operation"), default="copy", allowed={"copy", "cut"})
    resolved_paths: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("desktop_clipboard_files paths must be non-empty strings.")
        resolved = _resolve_tool_path(raw_path.strip(), ctx, tool="desktop_clipboard_files", operation="read")
        if isinstance(resolved, dict):
            return resolved
        canonical = resolved.resolve()
        if not canonical.exists():
            raise ValueError(f"desktop clipboard path does not exist: {raw_path}")
        if canonical not in seen:
            seen.add(canonical)
            resolved_paths.append(canonical)

    approval_key = _desktop_clipboard_files_approval_key(operation, resolved_paths)
    args["_approval_key"] = approval_key
    if approval_key not in ctx.approved_system_commands:
        return {
            "ok": False, "tool": "desktop_clipboard_files", "blocked": True,
            "recoverable": True, "reason": "desktop_action_requires_approval",
            "operation": "desktop", "requested_path": approval_key,
            "paths": [str(path) for path in resolved_paths], "item_count": len(resolved_paths),
            "guidance": "Ask the user to approve placing these exact paths on the desktop clipboard.",
        }
    return ctx.rust.desktop_clipboard_files(
        paths=[str(path) for path in resolved_paths], operation=operation
    )


def _optional_desktop_arg(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"desktop_action {name} must be a string when provided.")
    normalized = value.strip()
    return normalized or None


def _desktop_message_arg(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("desktop_send_message message must be a string when provided.")
    return value


def _process_list(args: dict[str, Any], ctx: ToolContext) -> Any:
    limit = _int_arg(args.get("limit"), default=20, minimum=1, maximum=200)
    sort_by = _enum_arg(
        args.get("sort_by"),
        default="cpu",
        allowed={"cpu", "memory"},
    )
    return ctx.rust.process_list(limit=limit, sort_by=sort_by)


def _run_system_command(args: dict[str, Any], ctx: ToolContext) -> Any:
    command = _enum_arg(
        args.get("command"),
        default="",
        allowed={
            "list_block_devices",
            "list_network_interfaces",
            "list_listening_ports",
            "service_status",
            "start_service",
            "stop_service",
            "restart_service",
        },
    )
    if not command:
        raise ValueError("run_system_command requires non-empty string arg 'command'.")

    target = args.get("target")
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError("target must be a non-empty string when provided.")
    target_value = target.strip() if isinstance(target, str) else None
    limit = _int_arg(args.get("limit"), default=50, minimum=1, maximum=200)

    if command.startswith(("start_", "stop_", "restart_")):
        if not target_value:
            return {
                "ok": False,
                "tool": "run_system_command",
                "blocked": True,
                "recoverable": True,
                "reason": "system_command_target_required",
                "operation": "system",
                "guidance": "This system command requires a concrete target such as a service name.",
            }
        approval_key = _system_command_approval_key(command, target_value)
        if approval_key not in ctx.approved_system_commands:
            return {
                "ok": False,
                "tool": "run_system_command",
                "blocked": True,
                "recoverable": True,
                "reason": "system_command_requires_approval",
                "operation": "system",
                "requested_path": approval_key,
                "guidance": (
                    "This system command can change the host machine. Ask the user to approve "
                    "this exact command before retrying."
                ),
            }

    return ctx.rust.run_system_command(
        command=command,
        target=target_value,
        limit=limit,
    )


def _read_path(args: dict[str, Any], ctx: ToolContext) -> Any:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("read_path requires non-empty string arg 'path'.")

    path_or_observation = _resolve_existing_path_or_target(
        raw_path,
        ctx,
        kind="any",
        tool="read_path",
    )
    if isinstance(path_or_observation, dict):
        return path_or_observation
    path = path_or_observation
    offset = _int_arg(args.get("offset"), default=1, minimum=1, maximum=1_000_000)
    limit = _int_arg(args.get("limit"), default=400, minimum=1, maximum=2_000)

    return ctx.rust.read_path(path=path, offset=offset, limit=limit)


def _write_file(args: dict[str, Any], ctx: ToolContext) -> Any:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("write_file requires non-empty string arg 'path'.")

    content = args.get("content")
    if not isinstance(content, str):
        raise ValueError("write_file requires string arg 'content'.")

    create_dirs = _bool_arg(args.get("create_dirs"), default=True)
    overwrite = _bool_arg(args.get("overwrite"), default=True)
    preserve_line_endings = _bool_arg(args.get("preserve_line_endings"), default=True)
    expected_sha256 = args.get("expected_sha256")
    if isinstance(expected_sha256, str) and not expected_sha256.strip():
        expected_sha256 = None
    elif expected_sha256 is not None and not isinstance(expected_sha256, str):
        raise ValueError("expected_sha256 must be a non-empty string when provided.")

    path_or_observation = _resolve_tool_path(
        raw_path,
        ctx,
        tool="write_file",
        operation="write",
    )
    if isinstance(path_or_observation, dict):
        return path_or_observation
    path = path_or_observation
    if expected_sha256 is not None and not path.exists():
        expected_sha256 = None
    workspace_root = _workspace_root_for_tool_path(path, ctx, "write")

    return ctx.rust.write_file(
        path=path,
        workspace_root=workspace_root,
        content=content,
        create_dirs=create_dirs,
        overwrite=overwrite,
        preserve_line_endings=preserve_line_endings,
        expected_sha256=expected_sha256,
    )


def _edit_file(args: dict[str, Any], ctx: ToolContext) -> Any:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("edit_file requires non-empty string arg 'path'.")

    old_text = args.get("old_text")
    if not isinstance(old_text, str) or old_text == "":
        raise ValueError("edit_file requires non-empty string arg 'old_text'.")

    new_text = args.get("new_text")
    if not isinstance(new_text, str):
        raise ValueError("edit_file requires string arg 'new_text'.")

    replace_all = _bool_arg(args.get("replace_all"), default=False)
    expected_sha256 = args.get("expected_sha256")
    if isinstance(expected_sha256, str) and not expected_sha256.strip():
        expected_sha256 = None
    elif expected_sha256 is not None and not isinstance(expected_sha256, str):
        raise ValueError("expected_sha256 must be a non-empty string when provided.")

    path_or_observation = _resolve_tool_path(
        raw_path,
        ctx,
        tool="edit_file",
        operation="write",
    )
    if isinstance(path_or_observation, dict):
        return path_or_observation
    path = path_or_observation
    workspace_root = _workspace_root_for_tool_path(path, ctx, "write")

    return ctx.rust.edit_file(
        path=path,
        workspace_root=workspace_root,
        old_text=old_text,
        new_text=new_text,
        replace_all=replace_all,
        expected_sha256=expected_sha256,
    )


def _delete_path(args: dict[str, Any], ctx: ToolContext) -> Any:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("delete_path requires non-empty string arg 'path'.")

    recursive = _bool_arg(args.get("recursive"), default=False)
    path_or_observation = _resolve_tool_path(
        raw_path,
        ctx,
        tool="delete_path",
        operation="delete",
    )
    if isinstance(path_or_observation, dict):
        return path_or_observation
    path = path_or_observation
    workspace_root = _workspace_root_for_tool_path(path, ctx, "delete")

    return ctx.rust.delete_path(
        path=path,
        workspace_root=workspace_root,
        recursive=recursive,
    )


def verify_mutation_observation(
    tool: str,
    args: dict[str, Any],
    observation: Any,
    *,
    workspace_root: Path,
    allowed_write_roots: list[Path] | None = None,
) -> Any:
    """Verify a successful mutation report against the filesystem."""
    if tool not in {"write_file", "edit_file", "delete_path"}:
        return observation
    if not isinstance(observation, dict) or observation.get("ok") is False:
        return observation

    raw_path = observation.get("path") or args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return _mutation_verification_failure(
            observation,
            tool,
            "The tool reported success without an affected path.",
        )
    target = Path(raw_path).expanduser()
    if not target.is_absolute():
        target = workspace_root / target
    target = target.resolve(strict=False)
    if allowed_write_roots is not None and not any(
        _path_is_within_or_equal(target, root.resolve(strict=False))
        for root in allowed_write_roots
    ):
        return _mutation_verification_failure(
            observation,
            tool,
            "The mutation result escaped the agent's assigned ownership.",
            target,
        )

    try:
        if tool == "delete_path":
            if observation.get("deleted") is not True:
                return _mutation_verification_failure(
                    observation,
                    tool,
                    "The delete result did not confirm deletion.",
                    target,
                )
            if target.exists() or target.is_symlink():
                return _mutation_verification_failure(
                    observation,
                    tool,
                    "The target still exists after deletion was reported.",
                    target,
                )
            postcondition, digest = "path_absent", None
        else:
            if target.is_symlink() or not target.is_file():
                return _mutation_verification_failure(
                    observation,
                    tool,
                    "The reported output path is not a regular file.",
                    target,
                )
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if observation.get("after_sha256") != digest:
                return _mutation_verification_failure(
                    observation,
                    tool,
                    "The mutation output hash does not match the file on disk.",
                    target,
                )
            postcondition = "file_hash_matches"
    except OSError as exc:
        return _mutation_verification_failure(
            observation,
            tool,
            f"Could not inspect the mutation result: {exc}",
            target,
        )

    verified = dict(observation)
    verified["verified"] = True
    verified["verification"] = {
        "postcondition": postcondition,
        "path": str(target),
        **({"sha256": digest} if digest else {}),
    }
    return verified


def _mutation_verification_failure(
    observation: dict[str, Any],
    tool: str,
    detail: str,
    target: Path | None = None,
) -> dict[str, Any]:
    failed = dict(observation)
    failed.update({
        "ok": False,
        "verified": False,
        "reason": "mutation_verification_failed",
        "error": detail,
        "recoverable": True,
        "guidance": "Inspect the target and retry; do not claim success before verification.",
        "tool": tool,
    })
    if target is not None:
        failed["path"] = str(target)
    return failed


def _walk_project(root: Path, *, max_entries: int) -> tuple[list[Path], bool]:
    policy = _walk_policy(root)
    paths: list[Path] = []
    truncated = _collect_project_paths(
        root,
        paths,
        policy,
        depth=0,
        max_entries=max_entries,
    )

    return paths, truncated


def _directory_children(root: Path, workspace_root: Path) -> list[dict[str, Any]]:
    policy = _walk_policy(root)
    try:
        entries = sorted(root.iterdir(), key=_entry_sort_key)
    except OSError:
        return []

    children: list[dict[str, Any]] = []
    for entry in entries:
        if _should_skip_path(entry, policy=policy, depth=0):
            continue
        item: dict[str, Any] = {
            "path": _relative_display(entry, workspace_root),
            "kind": "directory" if entry.is_dir() else "file",
        }
        if entry.is_file():
            try:
                item["bytes"] = entry.stat().st_size
            except OSError:
                pass
        children.append(item)
    return children


def _collect_project_paths(
    directory: Path,
    paths: list[Path],
    policy: WalkPolicy,
    *,
    depth: int,
    max_entries: int,
) -> bool:
    if len(paths) >= max_entries:
        return True
    try:
        canonical = directory.resolve()
    except OSError:
        return False
    if canonical in policy.visited_dirs:
        return False
    policy.visited_dirs.add(canonical)

    try:
        entries = sorted(directory.iterdir(), key=_entry_sort_key)
    except OSError:
        return False

    files = [
        entry
        for entry in entries
        if entry.is_file() and not _should_skip_path(entry, policy=policy, depth=depth)
    ]
    dirs = [
        entry
        for entry in entries
        if entry.is_dir() and not _should_skip_path(entry, policy=policy, depth=depth)
    ]

    remaining = max_entries - len(paths)
    paths.extend(files[:remaining])
    if len(files) > remaining:
        return True
    for entry in dirs:
        if len(paths) >= max_entries:
            return True
        paths.append(entry)
        if _collect_project_paths(
            entry,
            paths,
            policy,
            depth=depth + 1,
            max_entries=max_entries,
        ):
            return True
    return False


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def _looks_like_project_root(path: Path) -> bool:
    markers = (
        ".git",
        "Cargo.toml",
        "pyproject.toml",
        "package.json",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "Makefile",
    )
    return any((path / marker).exists() for marker in markers)


def _entry_sort_key(path: Path) -> tuple[int, str]:
    priority = 0 if path.is_file() else 1
    return (priority, path.name.lower())


def _should_skip_path(path: Path, *, policy: WalkPolicy | None = None, depth: int = 0) -> bool:
    if path.is_symlink() and not path.is_dir():
        return True
    name = path.name
    if name in SKIP_FILE_NAMES:
        return True
    if path.is_dir():
        path_depth = depth + 1
        if path.is_symlink():
            try:
                canonical = path.resolve()
            except OSError:
                return True
            if policy is None or canonical in policy.visited_dirs:
                return True
        if _should_skip_directory_name(name, path_depth):
            return True
    if policy is not None and _ignored_by_policy(path, policy):
        return True
    return False


def _should_skip_directory_name(name: str, depth: int) -> bool:
    if name.startswith(".") and name not in HIDDEN_DIR_ALLOWLIST:
        return True
    skip_names = DEEP_SKIP_DIR_NAMES if depth > 2 else SKIP_DIR_NAMES
    return name in skip_names or name.endswith(".egg-info")


def _walk_policy(root: Path) -> WalkPolicy:
    resolved_root = root.resolve(strict=False)
    boundary = _git_root_for(resolved_root) or resolved_root
    return WalkPolicy(
        root=resolved_root,
        boundary=boundary,
        ignore_rules=_load_hierarchical_ignore_rules(boundary, resolved_root),
    )


def _git_root_for(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _load_hierarchical_ignore_rules(boundary: Path, root: Path) -> list[IgnoreRule]:
    directories = _directory_chain(boundary, root)
    rules: list[IgnoreRule] = []
    for directory in directories:
        rules.extend(_read_gitignore_rules(directory / ".gitignore", directory))
    return rules


def _directory_chain(boundary: Path, root: Path) -> list[Path]:
    try:
        relative = root.resolve(strict=False).relative_to(boundary.resolve(strict=False))
    except ValueError:
        return [root]

    directories = [boundary]
    current = boundary
    for part in relative.parts:
        current = current / part
        if current.is_dir():
            directories.append(current)
    return directories


def _read_gitignore_rules(path: Path, base: Path) -> list[IgnoreRule]:
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return []

    rules: list[IgnoreRule] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        directory_only = line.endswith("/")
        pattern = line.rstrip("/")
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")
        rules.append(IgnoreRule(base=base, pattern=pattern, directory_only=directory_only))
    return rules


def _ignored_by_policy(path: Path, policy: WalkPolicy) -> bool:
    return any(_matches_ignore_rule(path, rule) for rule in policy.ignore_rules)


def _matches_ignore_rule(path: Path, rule: IgnoreRule) -> bool:
    try:
        relative = path.resolve(strict=False).relative_to(rule.base.resolve(strict=False)).as_posix()
    except ValueError:
        return False
    if not relative:
        return False
    if rule.directory_only and not path.is_dir():
        return False

    basename = Path(relative).name
    pattern = rule.pattern
    if "/" in pattern:
        return fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(relative, f"{pattern}/**")
    return any(fnmatch.fnmatch(part, pattern) for part in relative.split("/")) or fnmatch.fnmatch(basename, pattern)


def _context_file_names() -> set[str]:
    raw_value = os.environ.get("CONTEXT_FILE_NAMES")
    if not raw_value:
        return set(DEFAULT_CONTEXT_FILE_NAMES)
    names = {item.strip() for item in raw_value.split(os.pathsep) if item.strip()}
    return names or set(DEFAULT_CONTEXT_FILE_NAMES)


def _context_file_paths(start: Path) -> list[Path]:
    current = start if start.is_dir() else start.parent
    git_root = _git_root_for(current)
    if git_root is None:
        boundary = current
        directories = [current]
    else:
        boundary = git_root
        directories = _directory_chain(boundary, current)
    names = _context_file_names()
    paths: list[Path] = []
    for directory in directories:
        for name in names:
            path = directory / name
            if path.is_file():
                paths.append(path)
    return paths


def _looks_readable_text_file(path: Path) -> bool:
    if path.name in SKIP_FILE_NAMES:
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if not path.suffix and path.name.lower() in {"dockerfile", "makefile", "procfile"}:
        return True
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    if not sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _relative_display(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.relative_to(workspace_root))
    except ValueError:
        return str(path)


def _single_candidate_scope_path(candidates: Any, *, workspace_root: Path) -> Path | None:
    if not isinstance(candidates, list):
        return None

    candidate_paths: list[Path] = []
    directory_paths: list[Path] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            root_value = item.get("root")
            root = Path(root_value) if isinstance(root_value, str) and root_value else workspace_root
            path = root / path
        path = path.resolve(strict=False)
        candidate_paths.append(path)
        kind = item.get("kind") or item.get("match_type")
        if kind == "directory":
            directory_paths.append(path)

    if not candidate_paths or not directory_paths:
        return None

    for directory in directory_paths:
        if all(_path_is_within_or_equal(path, directory) for path in candidate_paths):
            return directory
    return None


def _path_is_within_or_equal(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _resolve_path(value: str, workspace_root: Path) -> Path:
    alias_path = _workspace_alias_path(value, workspace_root)
    if alias_path is not None:
        return alias_path
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return _ensure_within_workspace(path, workspace_root)


def _inspect_target_arg(value: str, workspace_root: Path) -> str:
    alias_path = _workspace_alias_path(value, workspace_root)
    if alias_path is not None:
        return str(alias_path)
    path = Path(value).expanduser()
    if path.is_absolute() or _looks_like_explicit_path(value):
        return str(_resolve_path(value, workspace_root))
    return value


def _bound_target_candidates(result: Any, query: str, *, limit: int = 20) -> Any:
    if not isinstance(result, dict) or result.get("status") != "candidates":
        return result
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return result

    normalized_query = query.strip().casefold().replace(" ", "_").replace("-", "_")
    top_level_prefixes: list[Any] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        path = candidate.get("path")
        if not isinstance(path, str) or "/" in path.strip("/") or "\\" in path.strip("\\"):
            continue
        normalized_name = path.strip("/\\").casefold().replace("-", "_")
        if normalized_query and normalized_name.startswith(normalized_query):
            top_level_prefixes.append(candidate)

    selected = top_level_prefixes if top_level_prefixes else candidates
    bounded = dict(result)
    bounded["candidates"] = selected[:limit]
    bounded["candidate_count"] = len(candidates)
    bounded["candidates_truncated"] = len(selected) > limit or len(selected) < len(candidates)
    if top_level_prefixes:
        bounded["candidate_filter"] = "top_level_prefix"
    return bounded


def _workspace_alias_path(value: str, workspace_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    alias = value.strip()
    if "/" in alias or "\\" in alias:
        return None
    resolved = resolve_workspace_alias(alias, workspace_root)
    return resolved


def _ensure_within_workspace(path: Path, workspace_root: Path) -> Path:
    resolved_root = workspace_root.expanduser().resolve()
    expanded_path = path.expanduser()
    if expanded_path.exists() or expanded_path.is_symlink():
        resolved_path = expanded_path.parent.resolve() / expanded_path.name
    else:
        resolved_path = expanded_path.resolve(strict=False)

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Path '{path}' is outside the workspace root '{resolved_root}'."
        ) from exc

    return resolved_path


def _looks_like_explicit_path(value: str) -> bool:
    return value.startswith(("~", ".", "..")) or "/" in value


def _int_arg(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        result = default
    else:
        result = int(value)

    return max(minimum, min(maximum, result))


def _enum_arg(value: Any, *, default: str, allowed: set[str]) -> str:
    if value is None:
        return default

    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"Expected one of {sorted(allowed)}.")

    return value


def _bool_arg(value: Any, *, default: bool) -> bool:
    if value is None:
        return default

    if not isinstance(value, bool):
        raise ValueError("Expected a boolean value.")

    return value


def _system_command_approval_key(command: str, target: str | None) -> str:
    normalized_target = (target or "").strip()
    return f"{command} {normalized_target}".strip()


def _desktop_action_approval_key(
    action: str,
    target: str | None,
    value: str | None,
) -> str:
    parts = ["desktop", action]
    if target:
        parts.append(target.strip())
    if value and _desktop_action_value_affects_approval(action):
        if action in {"clipboard_write", "type_text", "set_field_text"}:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            parts.append(f"sha256:{digest}")
            parts.append(f"bytes:{len(value.encode('utf-8'))}")
        else:
            parts.append(value.strip())
    return " ".join(parts)


def _desktop_action_value_affects_approval(action: str) -> bool:
    return action in {
        "set_volume",
        "set_mute",
        "set_brightness",
        "clipboard_write",
        "send_key",
        "type_text",
        "scroll",
        "invoke_element",
        "set_field_text",
    }


def _desktop_send_message_approval_key(target: str, message: str, submit: str) -> str:
    encoded = message.encode("utf-8")
    return (
        f"desktop send_message {target.strip()} "
        f"sha256:{hashlib.sha256(encoded).hexdigest()} "
        f"bytes:{len(encoded)} submit:{submit}"
    )


def _desktop_clipboard_files_approval_key(operation: str, paths: list[Path]) -> str:
    encoded = b"\0".join(os.fsencode(path) for path in paths)
    return f"desktop clipboard_files {operation} sha256:{hashlib.sha256(encoded).hexdigest()} items:{len(paths)}"


def _optional_desktop_backend_arg(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError(f"{name} is invalid.")
    normalized = value.strip()
    if name == "_backend_bus" and not re.fullmatch(r"(?::[0-9]+\.[0-9]+|[A-Za-z_][A-Za-z0-9_.-]*)", normalized):
        raise ValueError("_backend_bus is invalid.")
    if name == "_backend_path" and not re.fullmatch(r"/(?:[A-Za-z0-9_]+/?)*", normalized):
        raise ValueError("_backend_path is invalid.")
    return normalized


def _is_generated_path(path: Path, *, root: Path | None = None) -> bool:
    try:
        candidate = path.resolve(strict=False)
        if root is not None:
            candidate = candidate.relative_to(root.resolve(strict=False))
    except ValueError:
        candidate = path

    return any(
        component in SKIP_DIR_NAMES or component.endswith(".egg-info")
        for component in candidate.parts
    )
