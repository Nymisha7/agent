from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RustTools:
    rust_bin: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _active_process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _cancel_requested: bool = field(default=False, init=False, repr=False)

    def cancel_active(self) -> bool:
        with self._lock:
            process = self._active_process
            if process is None or process.poll() is not None:
                return False
            self._cancel_requested = True
            process.terminate()
            return True

    def run_json(self, args: list[str], *, timeout: float | None = None) -> Any:
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

    def glob_files(
        self,
        *,
        pattern: str,
        root: Path | None = None,
        limit: int,
        include_hidden: bool = False,
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
