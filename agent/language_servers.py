from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import select
from shutil import which
import subprocess
import time
from typing import Any


@dataclass(frozen=True)
class LanguageServerSpec:
    language: str
    server: str
    command: str
    args: tuple[str, ...]
    purpose: str


DEFAULT_LANGUAGE_SERVERS: tuple[LanguageServerSpec, ...] = (
    LanguageServerSpec(
        language="Python",
        server="pyright",
        command="pyright-langserver",
        args=("--stdio",),
        purpose="Fast, type-aware analysis for Python projects.",
    ),
    LanguageServerSpec(
        language="C/C++",
        server="clangd",
        command="clangd",
        args=(),
        purpose="LLVM-based indexing and diagnostics for C and C++.",
    ),
    LanguageServerSpec(
        language="Java",
        server="eclipse.jdt.ls",
        command="jdtls",
        args=(),
        purpose="Official Eclipse language server for Java.",
    ),
    LanguageServerSpec(
        language="JavaScript/TypeScript",
        server="tsserver",
        command="typescript-language-server",
        args=("--stdio",),
        purpose="TypeScript and JavaScript language intelligence.",
    ),
    LanguageServerSpec(
        language="Go",
        server="gopls",
        command="gopls",
        args=("serve",),
        purpose="Official Go language server.",
    ),
    LanguageServerSpec(
        language="Rust",
        server="rust-analyzer",
        command="rust-analyzer",
        args=(),
        purpose="Rust analysis, completions, and diagnostics.",
    ),
)


@dataclass
class LanguageServerProcess:
    spec: LanguageServerSpec
    process: subprocess.Popen[bytes]
    workspace_root: Path
    started_at: float
    initialized: bool = False
    next_id: int = 1


class LanguageServerManager:
    def __init__(self, specs: tuple[LanguageServerSpec, ...] | None = None) -> None:
        self.specs = specs or DEFAULT_LANGUAGE_SERVERS
        self._processes: dict[str, LanguageServerProcess] = {}

    def status(self, server: str | None = None) -> dict[str, Any]:
        servers = [self._spec_for(server)] if server else list(self.specs)
        result: list[dict[str, Any]] = []
        for spec in servers:
            process = self._processes.get(spec.server)
            running = process is not None and process.process.poll() is None
            if process is not None and not running:
                self._processes.pop(spec.server, None)
            executable = which(spec.command)
            result.append(
                {
                    "language": spec.language,
                    "server": spec.server,
                    "command": spec.command,
                    "args": list(spec.args),
                    "available": executable is not None,
                    "executable": executable,
                    "running": running,
                    "workspace_root": str(process.workspace_root) if running and process else None,
                    "pid": process.process.pid if running and process else None,
                    "purpose": spec.purpose,
                }
            )
        return {"ok": True, "servers": result}

    def start(self, server: str, workspace_root: Path) -> dict[str, Any]:
        spec = self._spec_for(server)
        existing = self._processes.get(spec.server)
        if existing is not None and existing.process.poll() is None:
            return {
                "ok": True,
                "server": spec.server,
                "running": True,
                "pid": existing.process.pid,
                "workspace_root": str(existing.workspace_root),
                "already_running": True,
            }

        executable = which(spec.command)
        if executable is None:
            return {
                "ok": False,
                "server": spec.server,
                "command": spec.command,
                "blocked": True,
                "recoverable": True,
                "reason": "language_server_not_installed",
                "guidance": f"Install {spec.server} so `{spec.command}` is available on PATH.",
            }

        workspace_root = workspace_root.expanduser().resolve()
        try:
            process = subprocess.Popen(
                [executable, *spec.args],
                cwd=workspace_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            return {
                "ok": False,
                "server": spec.server,
                "command": spec.command,
                "blocked": True,
                "recoverable": True,
                "reason": "language_server_start_failed",
                "error": str(exc),
            }

        time.sleep(0.05)
        if process.poll() is not None:
            stderr = ""
            try:
                stderr = process.stderr.read(2000).decode(errors="replace") if process.stderr else ""
            except OSError:
                stderr = ""
            return {
                "ok": False,
                "server": spec.server,
                "command": spec.command,
                "blocked": True,
                "recoverable": True,
                "reason": "language_server_exited",
                "exit_code": process.returncode,
                "stderr": stderr,
            }

        self._processes[spec.server] = LanguageServerProcess(
            spec=spec,
            process=process,
            workspace_root=workspace_root,
            started_at=time.time(),
        )
        return {
            "ok": True,
            "server": spec.server,
            "running": True,
            "pid": process.pid,
            "workspace_root": str(workspace_root),
            "command": spec.command,
            "args": list(spec.args),
        }

    def stop(self, server: str | None = None) -> dict[str, Any]:
        names = [self._spec_for(server).server] if server else list(self._processes)
        stopped: list[dict[str, Any]] = []
        for name in names:
            running = self._processes.pop(name, None)
            if running is None:
                stopped.append({"server": name, "stopped": False, "reason": "not_running"})
                continue
            process = running.process
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            stopped.append({"server": name, "stopped": True, "exit_code": process.returncode})
        return {"ok": True, "stopped": stopped}

    def stop_all(self) -> None:
        self.stop()

    def initialize(self, server: str, workspace_root: Path) -> dict[str, Any]:
        running = self._running_process(server, workspace_root)
        if isinstance(running, dict):
            return running
        if running.initialized:
            return {"ok": True, "server": running.spec.server, "initialized": True}

        response = self._request(
            running,
            "initialize",
            {
                "processId": None,
                "rootUri": running.workspace_root.as_uri(),
                "capabilities": {
                    "textDocument": {
                        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                        "definition": {"linkSupport": True},
                        "references": {},
                    },
                    "workspace": {"symbol": {}},
                },
            },
        )
        if not response.get("ok"):
            return response
        self._notify(running, "initialized", {})
        running.initialized = True
        return {
            "ok": True,
            "server": running.spec.server,
            "initialized": True,
            "capabilities": response.get("result", {}).get("capabilities", {}),
        }

    def workspace_symbols(
        self,
        server: str,
        query: str,
        workspace_root: Path,
        *,
        limit: int = 50,
    ) -> dict[str, Any]:
        ready = self.initialize(server, workspace_root)
        if not ready.get("ok"):
            return ready
        running = self._processes[self._spec_for(server).server]
        response = self._request(running, "workspace/symbol", {"query": query})
        if not response.get("ok"):
            return response
        symbols = _normalize_symbol_items(response.get("result"))[:limit]
        return {"ok": True, "server": running.spec.server, "symbols": symbols, "count": len(symbols)}

    def document_symbols(
        self,
        server: str,
        path: Path,
        workspace_root: Path,
        *,
        limit: int = 100,
    ) -> dict[str, Any]:
        running_or_error = self._ensure_document_ready(server, path, workspace_root)
        if isinstance(running_or_error, dict):
            return running_or_error
        running = running_or_error
        response = self._request(
            running,
            "textDocument/documentSymbol",
            {"textDocument": {"uri": path.expanduser().resolve().as_uri()}},
        )
        if not response.get("ok"):
            return response
        symbols = _normalize_symbol_items(response.get("result"))[:limit]
        return {"ok": True, "server": running.spec.server, "path": str(path), "symbols": symbols, "count": len(symbols)}

    def definition(
        self,
        server: str,
        path: Path,
        workspace_root: Path,
        *,
        line: int,
        character: int,
        limit: int = 20,
    ) -> dict[str, Any]:
        running_or_error = self._ensure_document_ready(server, path, workspace_root)
        if isinstance(running_or_error, dict):
            return running_or_error
        running = running_or_error
        response = self._request(
            running,
            "textDocument/definition",
            _text_document_position_params(path, line=line, character=character),
        )
        if not response.get("ok"):
            return response
        locations = _normalize_locations(response.get("result"))[:limit]
        return {"ok": True, "server": running.spec.server, "path": str(path), "locations": locations, "count": len(locations)}

    def references(
        self,
        server: str,
        path: Path,
        workspace_root: Path,
        *,
        line: int,
        character: int,
        include_declaration: bool = True,
        limit: int = 50,
    ) -> dict[str, Any]:
        running_or_error = self._ensure_document_ready(server, path, workspace_root)
        if isinstance(running_or_error, dict):
            return running_or_error
        running = running_or_error
        params = _text_document_position_params(path, line=line, character=character)
        params["context"] = {"includeDeclaration": include_declaration}
        response = self._request(running, "textDocument/references", params)
        if not response.get("ok"):
            return response
        locations = _normalize_locations(response.get("result"))[:limit]
        return {"ok": True, "server": running.spec.server, "path": str(path), "locations": locations, "count": len(locations)}

    def _spec_for(self, name: str | None) -> LanguageServerSpec:
        normalized = (name or "").strip().casefold()
        for spec in self.specs:
            if normalized in {
                spec.server.casefold(),
                spec.language.casefold(),
                spec.command.casefold(),
            }:
                return spec
        raise ValueError(f"Unknown language server: {name}")

    def _running_process(self, server: str, workspace_root: Path) -> LanguageServerProcess | dict[str, Any]:
        start_result = self.start(server, workspace_root)
        if not start_result.get("ok"):
            return start_result
        spec = self._spec_for(server)
        running = self._processes.get(spec.server)
        if running is None or running.process.poll() is not None:
            return {
                "ok": False,
                "server": spec.server,
                "blocked": True,
                "recoverable": True,
                "reason": "language_server_not_running",
                "guidance": "Start the language server before issuing LSP requests.",
            }
        return running

    def _ensure_document_ready(
        self,
        server: str,
        path: Path,
        workspace_root: Path,
    ) -> LanguageServerProcess | dict[str, Any]:
        path = path.expanduser().resolve()
        if not path.exists() or not path.is_file():
            return {
                "ok": False,
                "server": server,
                "blocked": True,
                "recoverable": True,
                "reason": "document_not_found",
                "path": str(path),
                "guidance": "Resolve an existing source file before asking the language server.",
            }
        ready = self.initialize(server, workspace_root)
        if not ready.get("ok"):
            return ready
        running = self._processes[self._spec_for(server).server]
        try:
            text = path.read_text(errors="ignore")
        except OSError as exc:
            return {
                "ok": False,
                "server": running.spec.server,
                "blocked": True,
                "recoverable": True,
                "reason": "document_read_failed",
                "path": str(path),
                "error": str(exc),
            }
        self._notify(
            running,
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": path.as_uri(),
                    "languageId": _language_id(path),
                    "version": 1,
                    "text": text,
                },
            },
        )
        return running

    def _request(
        self,
        running: LanguageServerProcess,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        request_id = running.next_id
        running.next_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        write_result = self._write_message(running, message)
        if write_result is not None:
            return write_result

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self._read_message(running, deadline=deadline)
            if response is None:
                continue
            if response.get("id") != request_id:
                continue
            if "error" in response:
                return {
                    "ok": False,
                    "server": running.spec.server,
                    "blocked": True,
                    "recoverable": True,
                    "reason": "language_server_request_failed",
                    "method": method,
                    "error": response.get("error"),
                }
            return {
                "ok": True,
                "server": running.spec.server,
                "method": method,
                "result": response.get("result"),
            }
        return {
            "ok": False,
            "server": running.spec.server,
            "blocked": True,
            "recoverable": True,
            "reason": "language_server_timeout",
            "method": method,
            "guidance": "The language server did not respond before the timeout.",
        }

    def _notify(self, running: LanguageServerProcess, method: str, params: dict[str, Any]) -> None:
        self._write_message(
            running,
            {"jsonrpc": "2.0", "method": method, "params": params},
        )

    def _write_message(
        self,
        running: LanguageServerProcess,
        message: dict[str, Any],
    ) -> dict[str, Any] | None:
        if running.process.stdin is None:
            return {
                "ok": False,
                "server": running.spec.server,
                "blocked": True,
                "recoverable": True,
                "reason": "language_server_stdin_unavailable",
            }
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        try:
            running.process.stdin.write(header + payload)
            running.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            return {
                "ok": False,
                "server": running.spec.server,
                "blocked": True,
                "recoverable": True,
                "reason": "language_server_write_failed",
                "error": str(exc),
            }
        return None

    def _read_message(
        self,
        running: LanguageServerProcess,
        *,
        deadline: float,
    ) -> dict[str, Any] | None:
        stdout = running.process.stdout
        if stdout is None:
            return None
        content_length: int | None = None
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                return None
            readable, _, _ = select.select([stdout], [], [], remaining)
            if not readable:
                return None
            line = stdout.readline()
            if not line:
                return None
            stripped = line.strip()
            if not stripped:
                break
            name, _, value = stripped.partition(b":")
            if name.lower() == b"content-length":
                try:
                    content_length = int(value.strip())
                except ValueError:
                    return None
        if content_length is None:
            return None
        remaining = max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([stdout], [], [], remaining)
        if not readable:
            return None
        payload = stdout.read(content_length)
        if len(payload) != content_length:
            return None
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


def _text_document_position_params(path: Path, *, line: int, character: int) -> dict[str, Any]:
    return {
        "textDocument": {"uri": path.expanduser().resolve().as_uri()},
        "position": {
            "line": max(0, line - 1),
            "character": max(0, character - 1),
        },
    }


def _language_id(path: Path) -> str:
    suffix = path.suffix.casefold()
    return {
        ".py": "python",
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascriptreact",
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".go": "go",
        ".rs": "rust",
    }.get(suffix, "plaintext")


def _normalize_symbol_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    symbols: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        location = item.get("location")
        selection_range = item.get("selectionRange")
        symbols.append(
            {
                "name": item.get("name"),
                "kind": item.get("kind"),
                "container": item.get("containerName"),
                "detail": item.get("detail"),
                "location": _normalize_location(location) if isinstance(location, dict) else None,
                "range": _normalize_range(selection_range) if isinstance(selection_range, dict) else _normalize_range(item.get("range")),
            }
        )
    return symbols


def _normalize_locations(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    locations: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_location(item)
        if normalized is not None:
            locations.append(normalized)
    return locations


def _normalize_location(value: dict[str, Any]) -> dict[str, Any] | None:
    uri = value.get("targetUri") or value.get("uri")
    range_value = value.get("targetSelectionRange") or value.get("targetRange") or value.get("range")
    if not isinstance(uri, str):
        return None
    return {
        "uri": uri,
        "path": _path_from_uri(uri),
        "range": _normalize_range(range_value),
    }


def _normalize_range(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    start = value.get("start")
    end = value.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    return {
        "start": {
            "line": int(start.get("line", 0)) + 1,
            "character": int(start.get("character", 0)) + 1,
        },
        "end": {
            "line": int(end.get("line", 0)) + 1,
            "character": int(end.get("character", 0)) + 1,
        },
    }


def _path_from_uri(uri: str) -> str | None:
    if not uri.startswith("file://"):
        return None
    try:
        from urllib.parse import unquote, urlparse

        parsed = urlparse(uri)
        return unquote(parsed.path)
    except Exception:
        return None


def default_language_servers() -> tuple[LanguageServerSpec, ...]:
    return DEFAULT_LANGUAGE_SERVERS


def language_server_context_text() -> str:
    lines = ["Preferred language servers:"]
    for spec in DEFAULT_LANGUAGE_SERVERS:
        status = "available" if which(spec.command) else "not found"
        command = " ".join([spec.command, *spec.args]).strip()
        lines.append(f"- {spec.language}: {spec.server} ({command}, {status})")
    return "\n".join(lines)
