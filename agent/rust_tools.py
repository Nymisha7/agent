from __future__ import annotations

import atexit
import json
import queue
import time
from copy import deepcopy
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DESKTOP_CAPABILITIES_CACHE_SECONDS = 300
DESKTOP_APPLICATION_RESOLVE_CACHE_SECONDS = 60


@dataclass
class RustTools:
    rust_bin: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _worker_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _active_process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _worker_process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _worker_reader: threading.Thread | None = field(default=None, init=False, repr=False)
    _pending_worker: dict[int, queue.Queue[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _next_request_id: int = field(default=0, init=False, repr=False)
    _cancel_requested: bool = field(default=False, init=False, repr=False)
    _cache: dict[tuple[Any, ...], tuple[float, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        atexit.register(self._stop_worker)

    def cancel_active(self) -> bool:
        with self._lock:
            process = self._active_process
            if process is None or process.poll() is not None:
                return False
            self._cancel_requested = True
            process.terminate()
            return True

    def run_json(self, args: list[str], *, timeout: float | None = None) -> Any:
        try:
            return self._run_worker_json(args, timeout=timeout)
        except (OSError, BrokenPipeError):
            self._stop_worker()
            return self._run_oneshot_json(args, timeout=timeout)

    def _run_oneshot_json(self, args: list[str], *, timeout: float | None = None) -> Any:
        command = [str(self.rust_bin), *args]
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with self._lock:
            self._active_process = process
            self._cancel_requested = False

        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                stdout, stderr = process.communicate()
                raise TimeoutError(
                    f"Rust tool timed out after {timeout:g}s: {' '.join(command)}"
                ) from exc

            with self._lock:
                canceled = self._cancel_requested

            if canceled:
                raise RuntimeError("Rust tool canceled.")

            if process.returncode != 0:
                raise RuntimeError(
                    f"Rust tool failed:\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}"
                )
        finally:
            with self._lock:
                if self._active_process is process:
                    self._active_process = None
                    self._cancel_requested = False

        output = stdout.strip()

        if not output:
            return None

        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

        lines = output.splitlines()
        return [json.loads(line) for line in lines]

    def _ensure_worker(self) -> subprocess.Popen[str]:
        process = self._worker_process
        if process is not None and process.poll() is None:
            return process
        process = subprocess.Popen(
            [str(self.rust_bin), "serve"],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1,
        )
        self._worker_process = process
        self._worker_reader = threading.Thread(
            target=self._read_worker_responses,
            args=(process,),
            daemon=True,
        )
        self._worker_reader.start()
        return process

    def _read_worker_responses(self, process: subprocess.Popen[str]) -> None:
        try:
            if process.stdout is None:
                return
            for line in process.stdout:
                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    response = {"id": None, "ok": False, "error": str(exc)}
                request_id = response.get("id")
                with self._worker_lock:
                    waiter = self._pending_worker.get(request_id)
                if waiter is not None:
                    waiter.put(response)
        finally:
            with self._worker_lock:
                pending = list(self._pending_worker.values())
                self._pending_worker.clear()
            for waiter in pending:
                waiter.put({"ok": False, "error": "Rust worker exited without a response."})

    def _stop_worker(self) -> None:
        with self._lock:
            process = self._worker_process
            self._worker_process = None
            self._worker_reader = None
            if self._active_process is process:
                self._active_process = None
                self._cancel_requested = False
        with self._worker_lock:
            pending = list(self._pending_worker.values())
            self._pending_worker.clear()
        for waiter in pending:
            waiter.put({"ok": False, "error": "Rust worker stopped."})
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()

    def _run_worker_json(self, args: list[str], *, timeout: float | None = None) -> Any:
        with self._worker_lock:
            process = self._ensure_worker()
            if process.stdin is None or process.stdout is None:
                raise OSError("Rust worker pipes are unavailable.")
            with self._lock:
                self._next_request_id += 1
                request_id = self._next_request_id
                self._active_process = process
                self._cancel_requested = False
            waiter: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._pending_worker[request_id] = waiter
            try:
                process.stdin.write(json.dumps({"id": request_id, "args": args}) + "\n")
                process.stdin.flush()
            except Exception:
                self._pending_worker.pop(request_id, None)
                raise
        try:
            try:
                response = waiter.get(timeout=timeout)
            except queue.Empty as exc:
                with self._worker_lock:
                    self._pending_worker.pop(request_id, None)
                self._stop_worker()
                raise TimeoutError(
                    f"Rust worker timed out after {timeout:g}s: {' '.join(args)}"
                ) from exc
            with self._lock:
                canceled = self._cancel_requested
            if canceled:
                raise RuntimeError("Rust tool canceled.")
            if not response.get("ok"):
                raise RuntimeError(f"Rust tool failed:\n{response.get('error', '')}")
            return response.get("result")
        finally:
            with self._worker_lock:
                self._pending_worker.pop(request_id, None)
            with self._lock:
                if self._active_process is process:
                    self._active_process = None
                    self._cancel_requested = False

    def _cached_json(self, key: tuple[Any, ...], ttl_seconds: float, run: Any) -> Any:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] < ttl_seconds:
                return deepcopy(cached[1])
        value = run()
        with self._lock:
            self._cache[key] = (now, deepcopy(value))
        return value

    def glob_files(
        self,
        *,
        pattern: str,
        root: Path | None = None,
        limit: int,
        include_hidden: bool = False,
        include_generated: bool = False,
        kind: str = "any",
    ) -> Any:
        args = [
            "glob",
            pattern,
            "--limit",
            str(limit),
            "--kind",
            kind,
        ]
        if include_hidden:
            args.append("--hidden")
        if include_generated:
            args.append("--include-generated")
        if root is not None:
            args.extend(["--root", str(root)])
        return self.run_json(args, timeout=30)

    def grep_files(
        self,
        *,
        pattern: str,
        root: Path | None = None,
        include: str | None = None,
        limit: int,
        literal_text: bool = False,
        include_hidden: bool = False,
    ) -> Any:
        args = [
            "grep",
            pattern,
            "--limit",
            str(limit),
        ]
        if include is not None:
            args.extend(["--include", include])
        if literal_text:
            args.append("--literal-text")
        if include_hidden:
            args.append("--hidden")
        if root is not None:
            args.extend(["--root", str(root)])
        return self.run_json(args, timeout=30)

    def read_path(self, *, path: Path | str, offset: int, limit: int) -> Any:
        return self.run_json(
            [
                "read",
                str(path),
                "--offset",
                str(offset),
                "--limit",
                str(limit),
            ],
            timeout=15,
        )

    def inspect_target(
        self,
        *,
        path: Path | str,
        workspace_root: Path,
        limit: int,
        offset: int,
        kind: str = "any",
    ) -> Any:
        return self.run_json(
            [
                "inspect-target",
                str(path),
                "--workspace-root",
                str(workspace_root),
                "--kind",
                kind,
                "--limit",
                str(limit),
                "--offset",
                str(offset),
                "--system-fallback",
                "false",
                "--contains-fallback",
                "true",
                "--fuzzy-fallback",
                "true",
            ],
            timeout=20,
        )

    def write_file(
        self,
        *,
        path: Path | str,
        workspace_root: Path,
        content: str,
        create_dirs: bool = True,
        overwrite: bool = True,
        preserve_line_endings: bool = True,
        expected_sha256: str | None = None,
    ) -> Any:
        args = [
            "write-file",
            str(path),
            "--workspace-root",
            str(workspace_root),
            "--content",
            content,
            "--create-dirs",
            str(create_dirs).lower(),
            "--overwrite",
            str(overwrite).lower(),
            "--preserve-line-endings",
            str(preserve_line_endings).lower(),
        ]
        if expected_sha256 is not None:
            args.extend(["--expected-sha256", expected_sha256])
        return self.run_json(args, timeout=20)

    def edit_file(
        self,
        *,
        path: Path | str,
        workspace_root: Path,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        expected_sha256: str | None = None,
    ) -> Any:
        args = [
            "edit-file",
            str(path),
            "--workspace-root",
            str(workspace_root),
            "--old-text",
            old_text,
            "--new-text",
            new_text,
            "--replace-all",
            str(replace_all).lower(),
        ]
        if expected_sha256 is not None:
            args.extend(["--expected-sha256", expected_sha256])
        return self.run_json(args, timeout=20)

    def delete_path(
        self,
        *,
        path: Path | str,
        workspace_root: Path,
        recursive: bool = False,
    ) -> Any:
        return self.run_json(
            [
                "delete-path",
                str(path),
                "--workspace-root",
                str(workspace_root),
                "--recursive",
                str(recursive).lower(),
            ],
            timeout=20,
        )

    def system_info(self) -> Any:
        return self.run_json(["system-info"], timeout=15)

    def connected_devices(self, *, scope: str = "all") -> Any:
        return self.run_json(
            [
                "connected-devices",
                "--scope",
                scope,
            ],
            timeout=20,
        )

    def desktop_capabilities(self) -> Any:
        return self._cached_json(
            ("desktop-capabilities",),
            DESKTOP_CAPABILITIES_CACHE_SECONDS,
            lambda: self.run_json(["desktop-capabilities"], timeout=15),
        )

    def desktop_observe(self, *, scope: str = "all", limit: int = 50) -> Any:
        return self.run_json(
            [
                "desktop-observe",
                "--scope",
                scope,
                "--limit",
                str(limit),
            ],
            timeout=20,
        )

    def desktop_resolve(self, *, query: str, kind: str = "any", limit: int = 10) -> Any:
        def run() -> Any:
            return self.run_json(
                [
                    "desktop-resolve",
                    query,
                    "--kind",
                    kind,
                    "--limit",
                    str(limit),
                ],
                timeout=20,
            )

        if kind == "application":
            return self._cached_json(
                ("desktop-resolve", query.strip().casefold(), kind, limit),
                DESKTOP_APPLICATION_RESOLVE_CACHE_SECONDS,
                run,
            )
        return run()

    def desktop_action(
        self,
        *,
        action: str,
        target: str | None = None,
        value: str | None = None,
        backend_bus: str | None = None,
        backend_path: str | None = None,
    ) -> Any:
        args = ["desktop-action", action]
        if target is not None:
            args.extend(["--target", target])
        if value is not None:
            args.extend(["--value", value])
        if backend_bus is not None:
            args.extend(["--backend-bus", backend_bus])
        if backend_path is not None:
            args.extend(["--backend-path", backend_path])
        return self.run_json(args, timeout=30)

    def desktop_clipboard_files(self, *, paths: list[str], operation: str = "copy") -> Any:
        args = ["desktop-clipboard-files", "--operation", operation]
        for path in paths:
            args.extend(["--path", path])
        return self.run_json(args, timeout=30)

    def process_list(self, *, limit: int = 20, sort_by: str = "cpu") -> Any:
        return self.run_json(
            [
                "process-list",
                "--limit",
                str(limit),
                "--sort-by",
                sort_by,
            ],
            timeout=20,
        )

    def run_system_command(
        self,
        *,
        command: str,
        target: str | None = None,
        limit: int = 50,
    ) -> Any:
        args = [
            "run-system-command",
            command,
            "--limit",
            str(limit),
        ]
        if target is not None:
            args.extend(["--target", target])
        return self.run_json(args, timeout=30)
