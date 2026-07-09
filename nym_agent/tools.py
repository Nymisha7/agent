from __future__ import annotations

import fnmatch
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import re
from collections.abc import Iterable
from typing import Any

from .language_servers import LanguageServerManager
from .policy import PolicyEngine
from .project_identity import resolve_workspace_alias
from .rust_tools import RustTools
from .tool_registry import ToolRegistry, ToolSpec


@dataclass
class ToolContext:
    rust: RustTools
    workspace_root: Path
    search_roots: list[Path]
    approved_external_read_roots: list[Path] = field(default_factory=list)
    approved_external_write_roots: list[Path] = field(default_factory=list)
    approved_external_delete_roots: list[Path] = field(default_factory=list)
    approved_system_commands: list[str] = field(default_factory=list)
    language_servers: LanguageServerManager | None = None


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
    return registry


def register_rust_file_tools(registry: ToolRegistry, _ctx: ToolContext) -> None:
    registry.register(
        ToolSpec(
            name="language_server",
            handler=_language_server,
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
                    "for explicit path matching. If no path is given, the workspace root "
                    "is used as the search root."
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
                    "summarize, explain, or inspect all files in a directory/project."
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
                    "Count and list devices visible to the current runtime. Returns category "
                    "counts for USB, storage, network, input, and Bluetooth devices."
                ),
                properties={
                    "scope": {
                        "type": "string",
                        "enum": ["all", "usb", "storage", "network", "input", "bluetooth"],
                        "description": "Optional device category to focus on.",
                        "default": "all",
                    },
                },
                required=[],
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
            name="write_file",
            handler=_write_file,
            schema=_function_schema(
                name="write_file",
                description=(
                    "Create or overwrite a text file inside the workspace. Use this when "
                    "the user asks to save, create, or update a file."
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
                    "require recursive=true when not empty."
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
        return alias_path
    if _looks_like_windows_drive_path(raw_path):
        return _resolve_windows_tool_path(raw_path, ctx, tool=tool, operation=operation)

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return _ensure_within_workspace(ctx.workspace_root / path, ctx.workspace_root)

    resolved = _resolve_without_strict(path)
    if _path_is_within_or_equal(resolved, ctx.workspace_root.expanduser().resolve()):
        return _ensure_within_workspace(resolved, ctx.workspace_root)

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
                "This looks like a Windows path, but Nym is running in a Linux/Ubuntu "
                "runtime that cannot currently access it. Provide a Linux-accessible "
                "mount path, or run Nym from WSL with the Windows drive mounted."
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
    text = resolved.as_posix().rstrip("/") or "/"
    broad = {
        "/",
        "/home",
        "/mnt",
        "/mnt/c",
        "/mnt/c/Users",
        "/mnt/d",
        "/mnt/d/Users",
    }
    return text in broad


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
    search_limit = limit if include_generated else max(limit, 200)

    result = _filter_glob_result(
        ctx.rust.glob_files(
            pattern=pattern,
            root=root,
            limit=search_limit,
            include_hidden=include_hidden,
            kind=kind,
        ),
        root=root,
        pattern=pattern,
        limit=limit,
        include_generated=include_generated,
    )

    if _glob_has_matches(result):
        return result

    for fallback_pattern in _glob_fallback_patterns(pattern, kind=kind):
        fallback_result = _filter_glob_result(
            ctx.rust.glob_files(
                pattern=fallback_pattern,
                root=root,
                limit=search_limit,
                include_hidden=include_hidden,
                kind=kind,
            ),
            root=root,
            pattern=fallback_pattern,
            limit=limit,
            include_generated=include_generated,
        )
        if _glob_has_matches(fallback_result):
            if isinstance(fallback_result, dict):
                fallback_result = dict(fallback_result)
                fallback_result["fallback_pattern"] = fallback_pattern
            return fallback_result

    return result


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

    return ctx.rust.inspect_target(
        path=path_arg,
        workspace_root=ctx.workspace_root,
        kind=kind,
        offset=offset,
        limit=limit,
    )


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
    max_files = _int_arg(args.get("max_files"), default=80, minimum=1, maximum=300)
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

    if root.is_file():
        return _inspect_single_file(
            root,
            workspace_root=ctx.workspace_root,
            max_bytes=max_bytes_per_file,
        )

    if not root.is_dir():
        raise ValueError(f"Path is not a regular file or directory: {root}")

    direct_children = _directory_children(root, ctx.workspace_root)
    tree: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    total_bytes = 0
    truncated = False

    for path in _walk_project(root):
        rel = _relative_display(path, ctx.workspace_root)

        if path.is_dir():
            tree.append({"path": rel, "kind": "directory"})
            continue

        if not path.is_file():
            skipped.append({"path": rel, "reason": "not a regular file"})
            continue

        tree.append({"path": rel, "kind": "file", "bytes": path.stat().st_size})

        if len(files) >= max_files:
            skipped.append({"path": rel, "reason": "max file count reached"})
            truncated = True
            continue

        if not _looks_readable_text_file(path):
            skipped.append({"path": rel, "reason": "binary or unsupported file type"})
            continue

        remaining = max_total_bytes - total_bytes
        if remaining <= 0:
            skipped.append({"path": rel, "reason": "max total content bytes reached"})
            truncated = True
            continue

        content_limit = min(max_bytes_per_file, remaining)
        inspected = _read_text_file(path, content_limit)
        total_bytes += inspected["bytes_read"]
        files.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "truncated": inspected["truncated"],
                "content": inspected["content"],
            }
        )
        if inspected["truncated"]:
            truncated = True

    return {
        "path": str(root),
        "kind": "directory",
        "direct_children": direct_children,
        "tree": tree,
        "files": files,
        "skipped": skipped,
        "file_count": len([item for item in tree if item["kind"] == "file"]),
        "read_file_count": len(files),
        "bytes_read": total_bytes,
        "truncated": truncated,
    }


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

    for path in _walk_project(root):
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
    scope = _enum_arg(
        args.get("scope"),
        default="all",
        allowed={"all", "usb", "storage", "network", "input", "bluetooth"},
    )
    return ctx.rust.connected_devices(scope=scope)


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


def _inspect_single_file(
    path: Path,
    *,
    workspace_root: Path,
    max_bytes: int,
) -> dict[str, Any]:
    rel = _relative_display(path, workspace_root)
    if not _looks_readable_text_file(path):
        return {
            "path": str(path),
            "kind": "file",
            "files": [],
            "skipped": [{"path": rel, "reason": "binary or unsupported file type"}],
            "file_count": 1,
            "read_file_count": 0,
            "bytes_read": 0,
            "truncated": False,
        }

    inspected = _read_text_file(path, max_bytes)
    return {
        "path": str(path),
        "kind": "file",
        "files": [
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "truncated": inspected["truncated"],
                "content": inspected["content"],
            }
        ],
        "skipped": [],
        "file_count": 1,
        "read_file_count": 1,
        "bytes_read": inspected["bytes_read"],
        "truncated": inspected["truncated"],
    }


def _walk_project(root: Path) -> list[Path]:
    policy = _walk_policy(root)
    paths: list[Path] = []
    _collect_project_paths(root, paths, policy, depth=0)

    return paths


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
) -> None:
    try:
        canonical = directory.resolve()
    except OSError:
        return
    if canonical in policy.visited_dirs:
        return
    policy.visited_dirs.add(canonical)

    try:
        entries = sorted(directory.iterdir(), key=_entry_sort_key)
    except OSError:
        return

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

    paths.extend(files)
    for entry in dirs:
        paths.append(entry)
        _collect_project_paths(entry, paths, policy, depth=depth + 1)


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


def _read_text_file(path: Path, max_bytes: int) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {
            "content": f"[Could not read file: {exc}]",
            "bytes_read": 0,
            "truncated": False,
        }

    truncated = len(data) > max_bytes
    data = data[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    return {
        "content": text,
        "bytes_read": len(data),
        "truncated": truncated,
    }


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


def _glob_has_matches(result: Any) -> bool:
    return isinstance(result, dict) and isinstance(result.get("matches"), list) and bool(result["matches"])


def _filter_glob_result(
    result: Any,
    *,
    root: Path,
    pattern: str,
    limit: int,
    include_generated: bool,
) -> Any:
    if not isinstance(result, dict) or not isinstance(result.get("matches"), list):
        return result
    if include_generated or _pattern_mentions_generated_path(pattern) or _is_generated_path(root):
        return _limit_glob_result(result, limit)

    filtered: list[Any] = []
    omitted_generated = 0
    for item in result["matches"]:
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        raw_path = item.get("path")
        if isinstance(raw_path, str) and _is_generated_path(Path(raw_path), root=root):
            omitted_generated += 1
            continue
        filtered.append(item)

    limited = dict(result)
    limited["matches"] = filtered
    if omitted_generated:
        limited["omitted_generated"] = omitted_generated + int(limited.get("omitted_generated", 0) or 0)
    return _limit_glob_result(limited, limit)


def _limit_glob_result(result: dict[str, Any], limit: int) -> dict[str, Any]:
    matches = result.get("matches")
    if not isinstance(matches, list):
        return result
    limited = dict(result)
    limited["matches"] = matches[:limit]
    limited["truncated"] = bool(result.get("truncated")) or len(matches) > limit
    return limited


def _pattern_mentions_generated_path(pattern: str) -> bool:
    normalized = pattern.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "*", "**"}]
    return any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in parts)


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


def _glob_fallback_patterns(pattern: str, *, kind: str = "any") -> list[str]:
    variants: list[str] = []
    if kind == "file" and "/" not in pattern and not pattern.startswith("**/"):
        variants.append(f"**/{pattern}")

    for candidate in _case_fallback_patterns(pattern):
        if candidate != pattern and candidate not in variants:
            variants.append(candidate)

    for match in re.finditer(r"[A-Za-z]{3,}", pattern):
        token = match.group(0)
        singular = _singularize_token(token)
        if singular == token:
            continue
        candidate = f"{pattern[:match.start()]}{singular}{pattern[match.end():]}"
        if candidate != pattern and candidate not in variants:
            variants.append(candidate)
    return variants


def _case_fallback_patterns(pattern: str) -> list[str]:
    segments = pattern.replace("\\", "/").split("/")
    variants: list[str] = []
    for index, segment in enumerate(segments):
        literal_prefix = _literal_case_prefix(segment)
        if not literal_prefix or not any(char.isalpha() for char in literal_prefix):
            continue
        suffix = segment[len(literal_prefix):]
        for replacement in _case_variants(literal_prefix):
            replacement_segment = f"{replacement}{suffix}"
            if replacement_segment == segment:
                continue
            candidate_segments = list(segments)
            candidate_segments[index] = replacement_segment
            candidate = "/".join(candidate_segments)
            if candidate != pattern and candidate not in variants:
                variants.append(candidate)
    return variants


def _literal_case_prefix(segment: str) -> str:
    if segment in {"", "*", "**"} or "?" in segment or "[" in segment or "]" in segment:
        return ""
    if "*" not in segment:
        return segment
    if segment.endswith("*") and segment.count("*") == 1:
        return segment[:-1]
    return ""


def _case_variants(segment: str) -> list[str]:
    variants = [
        segment.upper(),
        segment.lower(),
    ]
    if segment:
        variants.append(segment[:1].upper() + segment[1:].lower())
    result: list[str] = []
    for variant in variants:
        if variant not in result:
            result.append(variant)
    return result


def _singularize_token(token: str) -> str:
    lower = token.lower()
    replacement = token
    if lower.endswith("ies") and len(token) > 3:
        replacement = token[:-3] + "y"
    elif lower.endswith("s") and not lower.endswith("ss"):
        replacement = token[:-1]
    return _match_token_case(token, replacement)


def _match_token_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement
