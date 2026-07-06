import unittest
import os
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from nym_agent.main import default_workspace_root, resolve_rust_bin
from nym_agent.planner import (
    AgentSession,
    ModelToolCall,
    _annotate_tool_observation,
    _build_initial_messages,
    _prepare_tool_output,
    _preflight_tool_call,
    _request_frame,
    _requires_tool_use,
    _update_session_from_tool_result,
    agent_session_from_dict,
    agent_session_to_dict,
    run_agent,
)
from nym_agent.prompt_loader import load_system_prompt
from nym_agent.rust_tools import RustTools
from nym_agent.language_servers import LanguageServerManager, LanguageServerSpec
from nym_agent.tools import ToolContext, _context_file_paths, build_tool_registry


class PlannerToolUseTests(unittest.TestCase):
    def test_requires_tool_use_for_file_creation(self) -> None:
        self.assertTrue(_requires_tool_use("create a new file named notes.txt"))

    def test_requires_tool_use_for_project_questions(self) -> None:
        self.assertTrue(_requires_tool_use("tell me about this project"))

    def test_does_not_require_tool_use_for_simple_chat(self) -> None:
        self.assertFalse(_requires_tool_use("hello there"))

    def test_system_prompt_mentions_write_file_tool(self) -> None:
        prompt = load_system_prompt()
        self.assertIn("write_file", prompt)
        self.assertIn("Preserve user intent", prompt)
        self.assertIn("Resolve entities before acting", prompt)
        self.assertIn("named target", prompt)
        self.assertIn("Treat credentials, secrets, API keys", prompt)
        self.assertIn("secret_scan", prompt)
        self.assertIn("generated, dependency, cache, and build output", prompt)
        self.assertIn("Failed path operations only prove that the attempted path failed", prompt)
        self.assertIn("recent file operations", prompt)

    def test_tool_registry_exposes_edit_and_delete_tools(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)
        tool_names = {schema["name"] for schema in registry.schemas()}
        self.assertIn("write_file", tool_names)
        self.assertIn("edit_file", tool_names)
        self.assertIn("delete_path", tool_names)
        self.assertIn("secret_scan", tool_names)
        self.assertIn("language_server", tool_names)

    def test_language_server_status_tool_reports_configured_servers(self) -> None:
        manager = LanguageServerManager(
            specs=(
                LanguageServerSpec(
                    language="Example",
                    server="example-ls",
                    command="definitely-missing-example-ls",
                    args=("--stdio",),
                    purpose="Test language server.",
                ),
            )
        )
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/tmp"),
            search_roots=[],
            language_servers=manager,
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "language_server",
            {"action": "status"},
            ctx,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["servers"][0]["server"], "example-ls")
        self.assertFalse(result["servers"][0]["available"])

    def test_language_server_start_reports_missing_binary(self) -> None:
        manager = LanguageServerManager(
            specs=(
                LanguageServerSpec(
                    language="Example",
                    server="example-ls",
                    command="definitely-missing-example-ls",
                    args=(),
                    purpose="Test language server.",
                ),
            )
        )
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/tmp"),
            search_roots=[],
            language_servers=manager,
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "language_server",
            {"action": "start", "server": "example-ls"},
            ctx,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "language_server_not_installed")

    def test_language_server_workspace_symbol_delegates_to_manager(self) -> None:
        class FakeLanguageServers:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def workspace_symbols(self, server: str, query: str, workspace_root: Path, *, limit: int) -> dict[str, object]:
                self.calls.append({
                    "server": server,
                    "query": query,
                    "workspace_root": workspace_root,
                    "limit": limit,
                })
                return {"ok": True, "symbols": [{"name": query}], "count": 1}

        manager = FakeLanguageServers()
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
            language_servers=manager,  # type: ignore[arg-type]
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "language_server",
            {"action": "workspace_symbol", "server": "pyright", "query": "UserService", "limit": 5},
            ctx,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(manager.calls[0]["server"], "pyright")
        self.assertEqual(manager.calls[0]["query"], "UserService")
        self.assertEqual(manager.calls[0]["limit"], 5)

    def test_language_server_definition_resolves_workspace_file(self) -> None:
        class FakeLanguageServers:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def definition(
                self,
                server: str,
                path: Path,
                workspace_root: Path,
                *,
                line: int,
                character: int,
                limit: int,
            ) -> dict[str, object]:
                self.calls.append({
                    "server": server,
                    "path": path,
                    "workspace_root": workspace_root,
                    "line": line,
                    "character": character,
                    "limit": limit,
                })
                return {"ok": True, "locations": [], "count": 0}

        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            source = workspace_root / "src" / "service.py"
            source.parent.mkdir()
            source.write_text("def run():\n    return 1\n")
            manager = FakeLanguageServers()
            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
                language_servers=manager,  # type: ignore[arg-type]
            )
            registry = build_tool_registry(ctx)

            result = registry.execute(
                "language_server",
                {
                    "action": "definition",
                    "server": "pyright",
                    "path": "src/service.py",
                    "line": 1,
                    "character": 5,
                    "limit": 3,
                },
                ctx,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(manager.calls[0]["path"], source)
        self.assertEqual(manager.calls[0]["line"], 1)
        self.assertEqual(manager.calls[0]["character"], 5)
        self.assertEqual(manager.calls[0]["limit"], 3)

    def test_rust_tool_timeout_is_reported_as_timeout_error(self) -> None:
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "slow-rust-tool"
            script.write_text("#!/bin/sh\nsleep 2\n")
            os.chmod(script, 0o755)

            with self.assertRaises(TimeoutError):
                RustTools(script).run_json([], timeout=0.05)

    def test_tool_registry_returns_structured_timeout_observation(self) -> None:
        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                raise TimeoutError("slow search")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/workspace"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "glob",
            {"pattern": "*.txt", "path": "/workspace", "kind": "file"},
            ctx,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["recoverable"])
        self.assertEqual(result["reason"], "tool_timeout")

    def test_workspace_alias_binds_to_current_root(self) -> None:
        class FakeRust:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def inspect_target(self, **kwargs: object) -> dict[str, object]:
                self.calls.append(kwargs)
                return {
                    "status": "resolved",
                    "target": {
                        "path": str(kwargs["path"]),
                        "kind": "directory",
                    },
                }

        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "nym"
            workspace_root.mkdir()
            rust = FakeRust()
            ctx = ToolContext(
                rust=rust,  # type: ignore[arg-type]
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            result = registry.execute(
                "inspect_target",
                {"path": "nym", "kind": "directory"},
                ctx,
            )

        self.assertEqual(rust.calls[0]["path"], str(workspace_root))
        self.assertEqual(result["target"]["path"], str(workspace_root))

    def test_secret_scan_redacts_secret_values(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "repo"
            project.mkdir()
            (project / ".env").write_text(
                "OPENAI_API_KEY=sk-test-secret-value\n"
                "GITHUB_TOKEN=ghp_123456789012345678901234567890123456\n"
                "APP_NAME=demo\n"
            )

            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute("secret_scan", {"path": "repo"}, ctx)

        text = json.dumps(observation, ensure_ascii=False)
        self.assertTrue(observation["ok"])
        self.assertTrue(observation["redacted"])
        self.assertIn("OPENAI_API_KEY", text)
        self.assertNotIn("sk-test-secret-value", text)
        self.assertNotIn("ghp_123456789012345678901234567890123456", text)

    def test_request_frame_recognizes_workspace_identity_alias(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "nym"
            workspace_root.mkdir()

            frame = _request_frame("tell me about my nym project", workspace_root)

        self.assertEqual(frame.named_scope, "nym")
        self.assertFalse(frame.requires_target_resolution)
        self.assertEqual(frame.resolved_scope, workspace_root.resolve())

    def test_resolve_rust_bin_prefers_repo_binary_over_workspace_parent(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "nym"
            workspace_root = tmp_path / "app1"
            stale_root = tmp_path

            repo_bin = repo_root / "nym-rust" / "target" / "debug" / "nym-rust"
            stale_bin = stale_root / "nym-rust" / "target" / "debug" / "nym-rust"

            repo_bin.parent.mkdir(parents=True)
            stale_bin.parent.mkdir(parents=True)
            repo_bin.write_text("")
            stale_bin.write_text("")

            resolved = resolve_rust_bin(
                Namespace(rust_bin=None),
                workspace_root,
                repo_root=repo_root,
            )

            self.assertEqual(resolved, repo_bin.resolve())

    def test_default_workspace_root_uses_parent_when_running_from_nym_checkout(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            repo_root = workspace_root / "nym"
            (workspace_root / "nym-rust").mkdir()
            (workspace_root / "README.md").write_text("workspace\n")
            (repo_root / "nym-rust").mkdir(parents=True)
            (repo_root / "README.md").write_text("repo\n")

            old_cwd = Path.cwd()
            try:
                os.chdir(repo_root)
                resolved = default_workspace_root(Namespace(root=None))
            finally:
                os.chdir(old_cwd)

        self.assertEqual(resolved, workspace_root.resolve())

    def test_write_file_ignores_expected_hash_for_new_file(self) -> None:
        class FakeRust:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def write_file(self, **kwargs: object) -> dict[str, object]:
                self.calls.append(kwargs)
                return {"ok": True, "path": str(kwargs["path"])}

        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            rust = FakeRust()
            ctx = ToolContext(
                rust=rust,  # type: ignore[arg-type]
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            registry.execute(
                "write_file",
                {
                    "path": "new-file.txt",
                    "content": "testing 566",
                    "expected_sha256": "deadbeef",
                },
                ctx,
            )

            self.assertEqual(len(rust.calls), 1)
            self.assertIsNone(rust.calls[0]["expected_sha256"])

    def test_write_file_ignores_blank_expected_hash(self) -> None:
        class FakeRust:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def write_file(self, **kwargs: object) -> dict[str, object]:
                self.calls.append(kwargs)
                return {"ok": True, "path": str(kwargs["path"])}

        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            rust = FakeRust()
            ctx = ToolContext(
                rust=rust,  # type: ignore[arg-type]
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            registry.execute(
                "write_file",
                {
                    "path": "new-file.txt",
                    "content": "37943",
                    "expected_sha256": "",
                },
                ctx,
            )

            self.assertEqual(len(rust.calls), 1)
            self.assertIsNone(rust.calls[0]["expected_sha256"])

    def test_glob_retries_with_singular_pattern_when_plural_misses(self) -> None:
        class FakeRust:
            def __init__(self) -> None:
                self.patterns: list[str] = []

            def glob_files(self, **kwargs: object) -> dict[str, object]:
                pattern = str(kwargs["pattern"])
                self.patterns.append(pattern)
                if pattern == "*apps*":
                    return {"matches": [], "truncated": False, "backend": "fake"}
                if pattern == "*app*":
                    return {
                        "matches": [{"path": "/tmp/app1", "kind": "directory"}],
                        "truncated": False,
                        "backend": "fake",
                    }
                raise AssertionError(pattern)

        rust = FakeRust()
        ctx = ToolContext(
            rust=rust,  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "glob",
            {"pattern": "*apps*", "path": "/tmp", "kind": "directory"},
            ctx,
        )

        self.assertEqual(rust.patterns, ["*apps*", "*app*"])
        self.assertEqual(result["fallback_pattern"], "*app*")
        self.assertEqual(result["matches"][0]["path"], "/tmp/app1")

    def test_file_glob_retries_recursively_for_bare_filename_pattern(self) -> None:
        class FakeRust:
            def __init__(self) -> None:
                self.patterns: list[str] = []

            def glob_files(self, **kwargs: object) -> dict[str, object]:
                pattern = str(kwargs["pattern"])
                self.patterns.append(pattern)
                if pattern == "README*":
                    return {"matches": [], "truncated": False, "backend": "fake"}
                if pattern == "**/README*":
                    return {
                        "matches": [{"path": "/tmp/sample_project/web_ui/README.md", "kind": "file"}],
                        "truncated": False,
                        "backend": "fake",
                    }
                raise AssertionError(pattern)

        rust = FakeRust()
        ctx = ToolContext(
            rust=rust,  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "glob",
            {"pattern": "README*", "path": "/tmp/sample_project", "kind": "file"},
            ctx,
        )

        self.assertEqual(rust.patterns, ["README*", "**/README*"])
        self.assertEqual(result["fallback_pattern"], "**/README*")
        self.assertEqual(result["matches"][0]["path"], "/tmp/sample_project/web_ui/README.md")

    def test_file_glob_retries_literal_segment_case_variant(self) -> None:
        class FakeRust:
            def __init__(self) -> None:
                self.patterns: list[str] = []

            def glob_files(self, **kwargs: object) -> dict[str, object]:
                pattern = str(kwargs["pattern"])
                self.patterns.append(pattern)
                if pattern == "**/sample_project/**/readme*":
                    return {"matches": [], "truncated": False, "backend": "fake"}
                if pattern == "**/sample_project/**/README*":
                    return {
                        "matches": [{"path": "/tmp/sample_project/web_ui/README.md", "kind": "file"}],
                        "truncated": False,
                        "backend": "fake",
                    }
                return {"matches": [], "truncated": False, "backend": "fake"}

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "glob",
            {"pattern": "**/sample_project/**/readme*", "path": "/tmp", "kind": "file"},
            ctx,
        )

        self.assertIn("**/sample_project/**/README*", ctx.rust.patterns)  # type: ignore[attr-defined]
        self.assertEqual(result["fallback_pattern"], "**/sample_project/**/README*")

    def test_glob_with_ambiguous_relative_root_returns_candidates(self) -> None:
        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                raise AssertionError("glob_files should not run before root is resolved")

            def inspect_target(self, **kwargs: object) -> dict[str, object]:
                return {
                    "status": "candidates",
                    "query": "AlphaSuite",
                    "candidates": [
                        {"path": "AlphaSuite", "kind": "directory"},
                        {"path": "AlphaSuiteLegacy", "kind": "directory"},
                    ],
                }

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "glob",
            {"pattern": "README.md", "path": "AlphaSuite", "kind": "file"},
            ctx,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["recoverable"])
        self.assertEqual(result["reason"], "glob_root_ambiguous")
        self.assertEqual(len(result["candidates"]), 2)

    def test_glob_with_ambiguous_absolute_missing_root_returns_candidates(self) -> None:
        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                raise AssertionError("glob_files should not run before root is resolved")

            def inspect_target(self, **kwargs: object) -> dict[str, object]:
                return {
                    "status": "candidates",
                    "query": "alphaSuite",
                    "candidates": [
                        {"path": "AlphaSuite", "kind": "directory", "root": "/workspace"},
                        {"path": "AlphaSuiteLegacy", "kind": "directory", "root": "/workspace"},
                    ],
                }

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/workspace"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "glob",
            {"pattern": "*", "path": "/workspace/alphaSuite", "kind": "directory"},
            ctx,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["recoverable"])
        self.assertEqual(result["reason"], "glob_root_ambiguous")
        self.assertEqual(result["query"], "/workspace/alphaSuite")
        self.assertEqual(len(result["candidates"]), 2)

    def test_windows_glob_path_requires_approval_before_translation_search(self) -> None:
        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                raise AssertionError("glob_files should not run before external approval")

            def inspect_target(self, **kwargs: object) -> dict[str, object]:
                raise AssertionError("inspect_target should not run before external approval")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/workspace"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        with patch("nym_agent.tools._translate_windows_path", return_value=Path("/mnt/c/Users")):
            result = registry.execute(
                "glob",
                {"pattern": "**/calculator*", "path": "C:/Users", "kind": "directory"},
                ctx,
            )

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "external_windows_path_requires_approval")
        self.assertEqual(result["requested_path"], "C:/Users")
        self.assertEqual(result["translated_path"], "/mnt/c/Users")
        self.assertTrue(result["broad_path"])

    def test_windows_path_unavailable_when_runtime_cannot_translate(self) -> None:
        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                raise AssertionError("glob_files should not run for unavailable Windows path")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/workspace"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        with patch("nym_agent.tools._translate_windows_path", return_value=None):
            result = registry.execute(
                "glob",
                {"pattern": "README*", "path": "C:/Users/alice/Desktop", "kind": "file"},
                ctx,
            )

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "windows_path_unavailable_from_current_runtime")
        self.assertEqual(result["requested_path"], "C:/Users/alice/Desktop")

    def test_approved_windows_read_scope_allows_translated_glob_only_inside_prefix(self) -> None:
        class FakeRust:
            def __init__(self) -> None:
                self.roots: list[Path] = []

            def glob_files(self, **kwargs: object) -> dict[str, object]:
                self.roots.append(kwargs["root"])  # type: ignore[arg-type]
                return {"matches": [], "truncated": False, "backend": "fake"}

        rust = FakeRust()
        ctx = ToolContext(
            rust=rust,  # type: ignore[arg-type]
            workspace_root=Path("/workspace"),
            search_roots=[],
            approved_external_read_roots=[Path("/mnt/c/Users/alice/Desktop")],
        )
        registry = build_tool_registry(ctx)

        with patch(
            "nym_agent.tools._translate_windows_path",
            return_value=Path("/mnt/c/Users/alice/Desktop"),
        ):
            result = registry.execute(
                "glob",
                {"pattern": "calculator*", "path": "C:/Users/alice/Desktop", "kind": "file"},
                ctx,
        )

        self.assertEqual(result["matches"], [])
        self.assertTrue(rust.roots)
        self.assertEqual(set(rust.roots), {Path("/mnt/c/Users/alice/Desktop")})

    def test_external_read_approval_does_not_allow_external_delete(self) -> None:
        class FakeRust:
            def delete_path(self, **kwargs: object) -> dict[str, object]:
                raise AssertionError("delete_path should not run without delete approval")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/workspace"),
            search_roots=[],
            approved_external_read_roots=[Path("/mnt/c/Users/alice/Desktop")],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "delete_path",
            {"path": "/mnt/c/Users/alice/Desktop/calculator.lnk"},
            ctx,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "external_delete_requires_confirmation")
        self.assertEqual(result["operation"], "delete")

    def test_broad_external_linux_root_is_blocked_before_search(self) -> None:
        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                raise AssertionError("glob_files should not run for broad external root")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/workspace"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "glob",
            {"pattern": "**/calculator*", "path": "/mnt/c/Users", "kind": "directory"},
            ctx,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "broad_external_path_blocked")

    def test_glob_collapses_typo_root_candidates_to_single_scope(self) -> None:
        class FakeRust:
            def __init__(self, workspace_root: Path) -> None:
                self.workspace_root = workspace_root
                self.glob_root: Path | None = None

            def inspect_target(self, **kwargs: object) -> dict[str, object]:
                return {
                    "status": "candidates",
                    "query": "sample_projet",
                    "candidates": [
                        {"path": "sample_project", "kind": "directory", "root": str(self.workspace_root)},
                        {"path": "sample_project/web_ui", "kind": "directory", "root": str(self.workspace_root)},
                    ],
                }

            def glob_files(self, **kwargs: object) -> dict[str, object]:
                self.glob_root = kwargs["root"]  # type: ignore[assignment]
                return {
                    "matches": [{"path": str(self.workspace_root / "sample_project" / "README.md"), "kind": "file"}],
                    "truncated": False,
                    "backend": "fake",
                }

        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            (workspace_root / "sample_project").mkdir()
            rust = FakeRust(workspace_root)
            ctx = ToolContext(
                rust=rust,  # type: ignore[arg-type]
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            result = registry.execute(
                "glob",
                {"pattern": "README*", "path": "sample_projet", "kind": "file"},
                ctx,
            )

        self.assertEqual(rust.glob_root, workspace_root / "sample_project")
        self.assertEqual(result["matches"][0]["path"], str(workspace_root / "sample_project" / "README.md"))

    def test_glob_omits_generated_dependency_matches_by_default(self) -> None:
        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                return {
                    "matches": [
                        {
                            "path": "/tmp/sample_project/web_ui/node_modules/@babel/core/README.md",
                            "kind": "file",
                        },
                        {
                            "path": "/tmp/sample_project/web_ui/src/README.md",
                            "kind": "file",
                        },
                    ],
                    "truncated": False,
                    "backend": "fake",
                }

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "glob",
            {"pattern": "**/README*", "path": "/tmp/sample_project/web_ui", "kind": "file"},
            ctx,
        )

        self.assertEqual(
            result["matches"],
            [{"path": "/tmp/sample_project/web_ui/src/README.md", "kind": "file"}],
        )
        self.assertEqual(result["omitted_generated"], 1)

    def test_glob_can_include_generated_dependency_matches_when_requested(self) -> None:
        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                return {
                    "matches": [
                        {
                            "path": "/tmp/sample_project/web_ui/node_modules/@babel/core/README.md",
                            "kind": "file",
                        },
                    ],
                    "truncated": False,
                    "backend": "fake",
                }

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "glob",
            {
                "pattern": "**/README*",
                "path": "/tmp/sample_project/web_ui",
                "kind": "file",
                "include_generated": True,
            },
            ctx,
        )

        self.assertEqual(
            result["matches"],
            [{"path": "/tmp/sample_project/web_ui/node_modules/@babel/core/README.md", "kind": "file"}],
        )

    def test_inspect_tree_output_preserves_direct_children_when_compacted(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            sample_project = workspace_root / "sample_project"
            django = sample_project / "api_service"
            react = sample_project / "web_ui"
            nested = django / "todos"
            nested.mkdir(parents=True)
            react.mkdir(parents=True)
            for index in range(30):
                (nested / f"file_{index}.py").write_text(f"value = {index}\n")

            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute(
                "inspect_tree",
                {
                    "path": "sample_project",
                    "max_files": 25,
                    "max_bytes_per_file": 12000,
                    "max_total_bytes": 80000,
                },
                ctx,
            )
            payload = _prepare_tool_output(observation, max_bytes=4_000)

            self.assertIn("direct_children", observation)
        self.assertIn("sample_project/web_ui", payload)
        self.assertIn("sample_project/api_service", payload)

    def test_inspect_tree_reads_unknown_utf8_file_extensions(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "custom_project"
            project.mkdir()
            custom_file = project / "workflow.nymdsl"
            custom_file.write_text("step build\nstep verify\n")

            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute(
                "inspect_tree",
                {
                    "path": "custom_project",
                    "max_files": 10,
                    "max_bytes_per_file": 12000,
                    "max_total_bytes": 80000,
                },
                ctx,
            )

        self.assertEqual(observation["read_file_count"], 1)
        self.assertEqual(observation["files"][0]["path"], "custom_project/workflow.nymdsl")
        self.assertIn("step verify", observation["files"][0]["content"])

    def test_inspect_tree_respects_gitignore_without_git_repo(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "custom_project"
            project.mkdir()
            (project / ".gitignore").write_text("ignored.log\n")
            (project / "ignored.log").write_text("skip\n")
            (project / "kept.log").write_text("keep\n")

            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute("inspect_tree", {"path": "custom_project"}, ctx)

        file_paths = {item["path"] for item in observation["files"]}
        self.assertIn("custom_project/kept.log", file_paths)
        self.assertNotIn("custom_project/ignored.log", file_paths)

    def test_inspect_tree_merges_nested_gitignore_rules_from_git_root(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "repo"
            nested = project / "apps" / "api"
            nested.mkdir(parents=True)
            (project / ".git").mkdir()
            (project / ".gitignore").write_text("root-secret.txt\n")
            (nested / ".gitignore").write_text("local-secret.txt\n")
            (nested / "root-secret.txt").write_text("skip root\n")
            (nested / "local-secret.txt").write_text("skip local\n")
            (nested / "service.py").write_text("print('ok')\n")

            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute("inspect_tree", {"path": "repo/apps/api"}, ctx)

        file_paths = {item["path"] for item in observation["files"]}
        self.assertIn("repo/apps/api/service.py", file_paths)
        self.assertNotIn("repo/apps/api/root-secret.txt", file_paths)
        self.assertNotIn("repo/apps/api/local-secret.txt", file_paths)

    def test_inspect_tree_allows_important_hidden_configuration_directories(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "repo"
            (project / ".github" / "workflows").mkdir(parents=True)
            (project / ".secret").mkdir(parents=True)
            (project / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
            (project / ".secret" / "token.txt").write_text("hidden\n")

            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute("inspect_tree", {"path": "repo"}, ctx)

        tree_paths = {item["path"] for item in observation["tree"]}
        self.assertIn("repo/.github", tree_paths)
        self.assertIn("repo/.github/workflows/ci.yml", tree_paths)
        self.assertNotIn("repo/.secret", tree_paths)

    def test_inspect_tree_relaxes_skip_list_below_depth_two(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            deep_build = workspace_root / "repo" / "src" / "features" / "build"
            shallow_build = workspace_root / "repo" / "build"
            deep_build.mkdir(parents=True)
            shallow_build.mkdir(parents=True)
            (deep_build / "source.txt").write_text("deep source\n")
            (shallow_build / "generated.txt").write_text("generated\n")

            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute("inspect_tree", {"path": "repo"}, ctx)

        file_paths = {item["path"] for item in observation["files"]}
        self.assertIn("repo/src/features/build/source.txt", file_paths)
        self.assertNotIn("repo/build/generated.txt", file_paths)

    def test_inspect_tree_does_not_loop_on_circular_directory_symlinks(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "repo"
            nested = project / "nested"
            nested.mkdir(parents=True)
            (nested / "note.txt").write_text("hello\n")
            try:
                (nested / "loop").symlink_to(project, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are not supported on this filesystem")

            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute("inspect_tree", {"path": "repo"}, ctx)

        tree_paths = [item["path"] for item in observation["tree"]]
        self.assertEqual(tree_paths.count("repo/nested/note.txt"), 1)
        self.assertLess(len(tree_paths), 10)

    def test_context_file_paths_fallback_to_current_directory_without_git_repo(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "repo"
            child = project / "child"
            child.mkdir(parents=True)
            (project / "AGENTS.md").write_text("root hints\n")
            (child / "AGENTS.md").write_text("child hints\n")

            with patch("nym_agent.tools._git_root_for", return_value=None):
                paths = _context_file_paths(child)

        self.assertEqual(paths, [child / "AGENTS.md"])

    def test_context_file_paths_respect_env_names_and_git_boundary(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "repo"
            child = project / "child"
            child.mkdir(parents=True)
            (project / ".git").mkdir()
            (project / "NYM_HINTS.md").write_text("root hints\n")
            (child / "NYM_HINTS.md").write_text("child hints\n")

            with patch.dict(os.environ, {"CONTEXT_FILE_NAMES": "NYM_HINTS.md"}):
                paths = _context_file_paths(child)

        self.assertEqual(paths, [project / "NYM_HINTS.md", child / "NYM_HINTS.md"])

    def test_inspect_tree_recovers_simple_typo_target_to_single_scope(self) -> None:
        class FakeRust:
            def __init__(self, workspace_root: Path) -> None:
                self.workspace_root = workspace_root

            def inspect_target(self, **kwargs: object) -> dict[str, object]:
                return {
                    "status": "candidates",
                    "query": "sample_projet",
                    "candidates": [
                        {"path": "sample_project", "kind": "directory", "root": str(self.workspace_root)},
                        {"path": "sample_project/api_service", "kind": "directory", "root": str(self.workspace_root)},
                        {"path": "sample_project/web_ui", "kind": "directory", "root": str(self.workspace_root)},
                    ],
                }

        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            (workspace_root / "sample_project" / "api_service").mkdir(parents=True)
            (workspace_root / "sample_project" / "web_ui").mkdir(parents=True)
            ctx = ToolContext(
                rust=FakeRust(workspace_root),  # type: ignore[arg-type]
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute(
                "inspect_tree",
                {
                    "path": "sample_projet",
                    "max_files": 25,
                    "max_bytes_per_file": 12000,
                    "max_total_bytes": 80000,
                },
                ctx,
            )

        self.assertEqual(observation["path"], str(workspace_root / "sample_project"))
        self.assertEqual(
            [item["path"] for item in observation["direct_children"]],
            ["sample_project/api_service", "sample_project/web_ui"],
        )

    def test_inspect_tree_returns_recoverable_ambiguity_for_multiple_scopes(self) -> None:
        class FakeRust:
            def inspect_target(self, **kwargs: object) -> dict[str, object]:
                return {
                    "status": "candidates",
                    "query": "AlphaSuite",
                    "candidates": [
                        {"path": "AlphaSuiteRelease", "kind": "directory"},
                        {"path": "AlphaSuiteClean", "kind": "directory"},
                    ],
                }

        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            ctx = ToolContext(
                rust=FakeRust(),  # type: ignore[arg-type]
                workspace_root=workspace_root,
                search_roots=[],
            )
            registry = build_tool_registry(ctx)

            observation = registry.execute(
                "inspect_tree",
                {"path": "AlphaSuite"},
                ctx,
            )

        self.assertFalse(observation["ok"])
        self.assertTrue(observation["recoverable"])
        self.assertEqual(observation["reason"], "target_ambiguous")
        self.assertEqual(len(observation["candidates"]), 2)

    def test_agent_session_round_trips_path_context(self) -> None:
        session = AgentSession(
            active_root="/workspace/sample_project",
            focus_paths=["/workspace/sample_project/api_service"],
            last_candidates=[{"path": "/workspace/sample_project/web_ui", "kind": "directory"}],
            pending_action={
                "status": "unresolved",
                "action": "delete",
                "candidates": [{"path": "/workspace/AlphaSuiteClean", "kind": "directory"}],
            },
        )

        restored = agent_session_from_dict(agent_session_to_dict(session))

        self.assertEqual(restored.active_root, session.active_root)
        self.assertEqual(restored.focus_paths, session.focus_paths)
        self.assertEqual(restored.last_candidates, session.last_candidates)
        self.assertEqual(restored.recent_files, session.recent_files)
        self.assertEqual(restored.pending_action, session.pending_action)

    def test_session_remembers_successful_file_mutations(self) -> None:
        session = AgentSession()

        _update_session_from_tool_result(
            session,
            tool="write_file",
            args={"path": "/workspace/sample_project/web_ui/new_file.txt"},
            observation={
                "path": "/workspace/sample_project/web_ui/new_file.txt",
                "resource": "sample_project/web_ui/new_file.txt",
                "created": True,
            },
            workspace_root=Path("/workspace"),
        )
        _update_session_from_tool_result(
            session,
            tool="delete_path",
            args={"path": "/workspace/sample_project/web_ui/new_file.txt"},
            observation={
                "path": "/workspace/sample_project/web_ui/new_file.txt",
                "resource": "sample_project/web_ui/new_file.txt",
                "deleted": True,
                "kind": "file",
            },
            workspace_root=Path("/workspace"),
        )

        self.assertEqual(session.recent_files[0]["path"], "/workspace/sample_project/web_ui/new_file.txt")
        self.assertEqual(session.recent_files[0]["action"], "delete")
        self.assertEqual(session.recent_files[0]["status"], "deleted")

    def test_initial_message_includes_session_path_context(self) -> None:
        messages = _build_initial_messages(
            workspace_root="/workspace",
            context_text="",
            session=AgentSession(
                active_root="/workspace/sample_project",
                focus_paths=["/workspace/sample_project/api_service"],
                last_candidates=[{"path": "/workspace/sample_project/web_ui", "kind": "directory"}],
            ),
            user_prompt="inside those?",
            conversation_history=None,
        )

        content = messages[0]["content"]
        self.assertIn("Session path context", content)
        self.assertIn("Active root: /workspace/sample_project", content)
        self.assertIn("/workspace/sample_project/web_ui (directory)", content)
        self.assertIn("resolve that target directly", content)

    def test_initial_message_includes_recent_file_operations(self) -> None:
        messages = _build_initial_messages(
            workspace_root="/workspace",
            context_text="",
            session=AgentSession(
                recent_files=[
                    {
                        "path": "/workspace/sample_project/web_ui/new_file.txt",
                        "action": "delete",
                        "status": "deleted",
                    }
                ],
            ),
            user_prompt="from sample_project",
            conversation_history=None,
        )

        content = messages[0]["content"]
        self.assertIn("Recent file operations", content)
        self.assertIn("/workspace/sample_project/web_ui/new_file.txt (delete, deleted)", content)
        self.assertIn("recent file operation path", content)

    def test_session_records_pending_action_for_ambiguous_mutation_target(self) -> None:
        session = AgentSession()

        _update_session_from_tool_result(
            session,
            tool="inspect_target",
            args={"path": "AlphaSuite", "kind": "directory"},
            observation={
                "status": "candidates",
                "query": "AlphaSuite",
                "candidates": [
                    {
                        "path": "browser/env/lib/python3.12/site-packages/django/contrib/gis/gdal/raster",
                        "kind": "directory",
                    },
                    {"path": "AlphaSuiteRelease/env", "kind": "directory"},
                    {"path": "AlphaSuiteClean", "kind": "directory"},
                    {"path": "AlphaSuiteV2Clean", "kind": "directory"},
                    {"path": "AlphaSuiteRelease", "kind": "directory"},
                ],
            },
            workspace_root=Path("/workspace"),
            user_prompt="delete readme from my AlphaSuite project",
        )

        self.assertIsNotNone(session.pending_action)
        self.assertEqual(session.pending_action["status"], "unresolved")
        self.assertEqual(session.pending_action["action"], "delete")
        self.assertEqual(
            session.pending_action["candidates"],
            [
                {"path": "/workspace/AlphaSuiteRelease", "kind": "directory"},
                {"path": "/workspace/AlphaSuiteClean", "kind": "directory"},
                {"path": "/workspace/AlphaSuiteV2Clean", "kind": "directory"},
            ],
        )

    def test_session_records_pending_action_for_ambiguous_non_mutation_target(self) -> None:
        session = AgentSession()

        _update_session_from_tool_result(
            session,
            tool="inspect_target",
            args={"path": "AlphaSuite", "kind": "directory"},
            observation={
                "status": "candidates",
                "query": "AlphaSuite",
                "candidates": [
                    {"path": "AlphaSuiteRelease", "kind": "directory"},
                    {"path": "AlphaSuiteClean", "kind": "directory"},
                    {"path": "AlphaSuiteV2Clean", "kind": "directory"},
                ],
            },
            workspace_root=Path("/workspace"),
            user_prompt="which AlphaSuite project are you referring to",
        )

        self.assertIsNotNone(session.pending_action)
        self.assertEqual(session.pending_action["status"], "unresolved")
        self.assertEqual(session.pending_action["action"], "answer")
        self.assertEqual(len(session.pending_action["candidates"]), 3)

    def test_initial_message_includes_pending_action(self) -> None:
        messages = _build_initial_messages(
            workspace_root="/workspace",
            context_text="",
            session=AgentSession(
                pending_action={
                    "status": "unresolved",
                    "action": "delete",
                    "candidates": [
                        {"path": "/workspace/AlphaSuiteRelease", "kind": "directory"},
                        {"path": "/workspace/AlphaSuiteClean", "kind": "directory"},
                    ],
                }
            ),
            user_prompt="yup why not?",
            conversation_history=None,
        )

        content = messages[0]["content"]
        self.assertIn("Pending unresolved action: delete", content)
        self.assertIn("1. /workspace/AlphaSuiteRelease (directory)", content)
        self.assertIn("do not continue it until the user selects exactly one", content)

    def test_preflight_blocks_tools_when_pending_action_unresolved_by_meta_question(self) -> None:
        session = AgentSession(
            pending_action={
                "status": "unresolved",
                "action": "delete",
                "candidates": [
                    {"path": "/workspace/AlphaSuiteRelease", "kind": "directory"},
                    {"path": "/workspace/AlphaSuiteClean", "kind": "directory"},
                    {"path": "/workspace/AlphaSuiteV2Clean", "kind": "directory"},
                ],
            }
        )
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="glob",
                call_id="call-1",
                arguments={
                    "pattern": "README*",
                    "path": "/workspace/AlphaSuiteRelease",
                    "kind": "file",
                },
            ),
            tool_ctx=ctx,
            session=session,
            user_prompt="yup why weren't you able to get it the first time?",
        )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertEqual(observation["reason"], "pending_target_clarification_unresolved")

    def test_preflight_blocks_tools_when_pending_action_unresolved_by_transcript_snippet(self) -> None:
        session = AgentSession(
            pending_action={
                "status": "unresolved",
                "action": "answer",
                "query": "AlphaSuite",
                "candidates": [
                    {"path": "/workspace/AlphaSuiteRelease", "kind": "directory"},
                    {"path": "/workspace/AlphaSuiteClean", "kind": "directory"},
                ],
            }
        )
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="glob",
                call_id="call-1",
                arguments={"pattern": "*.md", "path": "/workspace/AlphaSuiteRelease", "kind": "file"},
            ),
            tool_ctx=ctx,
            session=session,
            user_prompt="Assistant  2026-07-02T12:36:15.813394+00:00",
        )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertEqual(observation["reason"], "pending_target_clarification_unresolved")

    def test_preflight_blocks_workspace_file_search_for_named_project_mutation(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="glob",
                call_id="call-1",
                arguments={
                    "pattern": "*.txt",
                    "path": "/workspace",
                    "kind": "file",
                },
            ),
            tool_ctx=ctx,
            session=AgentSession(),
            user_prompt="delete read me from my AlphaSuite project",
        )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertEqual(observation["reason"], "named_project_scope_not_resolved")
        self.assertEqual(observation["named_scope"], "AlphaSuite")

    def test_preflight_blocks_workspace_inspect_for_named_project_answer(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="inspect_tree",
                call_id="call-1",
                arguments={
                    "path": "/workspace",
                    "max_files": 10,
                    "max_bytes_per_file": 2000,
                    "max_total_bytes": 8000,
                },
            ),
            tool_ctx=ctx,
            session=AgentSession(),
            user_prompt="can you tell me about my alphaSuite PROJECT?",
        )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertTrue(observation["recoverable"])
        self.assertEqual(observation["reason"], "named_project_scope_not_resolved")
        self.assertEqual(observation["intent"], "inspect")
        self.assertEqual(observation["named_scope"], "alphaSuite")
        self.assertIn("inspect_target", observation["guidance"])

    def test_preflight_blocks_default_grep_for_named_project_search(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="grep",
                call_id="call-1",
                arguments={"pattern": "README"},
            ),
            tool_ctx=ctx,
            session=AgentSession(),
            user_prompt="search for AlphaSuite project",
        )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertEqual(observation["reason"], "named_project_scope_not_resolved")
        self.assertEqual(observation["intent"], "search")
        self.assertEqual(observation["named_scope"], "AlphaSuite")

    def test_preflight_named_scope_guard_is_generic_across_intents_and_names(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )
        cases = [
            (
                "inspect_tree",
                {"path": "/workspace"},
                "tell me about ledger-api",
                "inspect",
                "ledger-api",
            ),
            (
                "list_path",
                {"path": "/workspace"},
                "show files in mobile.repo folder",
                "inspect",
                "mobile.repo",
            ),
            (
                "read_path",
                {"path": "/workspace"},
                "explain DataLake42 project",
                "inspect",
                "DataLake42",
            ),
            (
                "grep",
                {"pattern": "README"},
                "search for billing_app project",
                "search",
                "billing_app",
            ),
            (
                "glob",
                {"pattern": "README*", "path": "/workspace", "kind": "file"},
                "delete readme from my docs-site repo",
                "delete",
                "docs-site",
            ),
        ]

        for tool_name, arguments, prompt, intent, named_scope in cases:
            with self.subTest(prompt=prompt, tool=tool_name):
                observation = _preflight_tool_call(
                    ModelToolCall(
                        name=tool_name,
                        call_id="call-1",
                        arguments=arguments,
                    ),
                    tool_ctx=ctx,
                    session=AgentSession(),
                    user_prompt=prompt,
                )

                self.assertIsNotNone(observation)
                self.assertTrue(observation["blocked"])
                self.assertEqual(observation["reason"], "named_project_scope_not_resolved")
                self.assertEqual(observation["intent"], intent)
                self.assertEqual(observation["named_scope"], named_scope)

    def test_preflight_handles_additional_prompt_variants_without_special_casing(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )
        cases = [
            (
                "can you give me the rundown of telemetry-platform",
                "inspect_tree",
                "inspect",
                "telemetry-platform",
            ),
            (
                "locate the readme in commerce_gateway",
                "glob",
                "search",
                "commerce_gateway",
            ),
            (
                "remove the readme from docs-core repo",
                "glob",
                "delete",
                "docs-core",
            ),
        ]

        for prompt, tool_name, intent, named_scope in cases:
            with self.subTest(prompt=prompt):
                observation = _preflight_tool_call(
                    ModelToolCall(
                        name=tool_name,
                        call_id="call-1",
                        arguments={
                            "pattern": "README*",
                            "path": "/workspace",
                            "kind": "file",
                        },
                    ),
                    tool_ctx=ctx,
                    session=AgentSession(),
                    user_prompt=prompt,
                )

                self.assertIsNotNone(observation)
                self.assertTrue(observation["blocked"])
                self.assertEqual(observation["reason"], "named_project_scope_not_resolved")
                self.assertEqual(observation["intent"], intent)
                self.assertEqual(observation["named_scope"], named_scope)

    def test_preflight_uses_container_scope_for_inside_phrase(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="glob",
                call_id="call-1",
                arguments={"pattern": "README*", "path": "/workspace", "kind": "file"},
            ),
            tool_ctx=ctx,
            session=AgentSession(),
            user_prompt="check README in frontend inside inventory-service",
        )

        self.assertIsNotNone(observation)
        self.assertEqual(observation["reason"], "named_project_scope_not_resolved")
        self.assertEqual(observation["named_scope"], "inventory-service")

    def test_preflight_allows_workspace_inspect_for_workspace_request(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="inspect_tree",
                call_id="call-1",
                arguments={"path": "/workspace"},
            ),
            tool_ctx=ctx,
            session=AgentSession(),
            user_prompt="tell me about my workspace",
        )

        self.assertIsNone(observation)

    def test_preflight_allows_current_project_inspect_without_named_scope(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="inspect_tree",
                call_id="call-1",
                arguments={"path": "/workspace"},
            ),
            tool_ctx=ctx,
            session=AgentSession(),
            user_prompt="tell me about this project",
        )

        self.assertIsNone(observation)

    def test_preflight_blocks_named_summary_at_workspace_root_but_allows_named_path(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        blocked = _preflight_tool_call(
            ModelToolCall(
                name="inspect_tree",
                call_id="call-1",
                arguments={"path": "/workspace"},
            ),
            tool_ctx=ctx,
            session=AgentSession(),
            user_prompt="tell me about inventory-service",
        )
        allowed = _preflight_tool_call(
            ModelToolCall(
                name="inspect_tree",
                call_id="call-2",
                arguments={"path": "/workspace/inventory-service"},
            ),
            tool_ctx=ctx,
            session=AgentSession(),
            user_prompt="tell me about inventory-service",
        )

        self.assertIsNotNone(blocked)
        self.assertEqual(blocked["reason"], "named_project_scope_not_resolved")
        self.assertEqual(blocked["named_scope"], "inventory-service")
        self.assertIsNone(allowed)

    def test_preflight_allows_only_selected_pending_candidate(self) -> None:
        session = AgentSession(
            pending_action={
                "status": "unresolved",
                "action": "delete",
                "candidates": [
                    {"path": "/workspace/AlphaSuiteRelease", "kind": "directory"},
                    {"path": "/workspace/AlphaSuiteClean", "kind": "directory"},
                    {"path": "/workspace/AlphaSuiteV2Clean", "kind": "directory"},
                ],
            }
        )
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/nym-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        blocked = _preflight_tool_call(
            ModelToolCall(
                name="delete_path",
                call_id="call-1",
                arguments={"path": "/workspace/AlphaSuiteRelease/README.md"},
            ),
            tool_ctx=ctx,
            session=session,
            user_prompt="3",
        )
        allowed = _preflight_tool_call(
            ModelToolCall(
                name="glob",
                call_id="call-2",
                arguments={"pattern": "README*", "path": "/workspace/AlphaSuiteV2Clean", "kind": "file"},
            ),
            tool_ctx=ctx,
            session=session,
            user_prompt="3",
        )

        self.assertIsNotNone(blocked)
        self.assertEqual(blocked["reason"], "tool_path_outside_selected_candidate")
        self.assertIsNone(allowed)
        self.assertEqual(session.pending_action["status"], "resolved")
        self.assertEqual(session.pending_action["selected"]["path"], "/workspace/AlphaSuiteV2Clean")

    def test_preflight_blocks_write_file_create_for_edit_intent(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )

            observation = _preflight_tool_call(
                ModelToolCall(
                    name="write_file",
                    call_id="call-1",
                    arguments={"path": "new_file.txt", "content": "ny test"},
                ),
                tool_ctx=ctx,
                session=AgentSession(),
                user_prompt="edit new_file.txt and add ny test in it",
            )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertEqual(observation["reason"], "edit_intent_would_create_missing_file")

    def test_preflight_redirects_missing_delete_to_recent_file(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            recent_path = workspace_root / "sample_project" / "web_ui" / "new_file.txt"
            recent_path.parent.mkdir(parents=True)
            recent_path.write_text("ny test")
            session = AgentSession(
                recent_files=[
                    {
                        "path": str(recent_path),
                        "action": "write",
                        "status": "exists",
                    }
                ]
            )
            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )

            observation = _preflight_tool_call(
                ModelToolCall(
                    name="delete_path",
                    call_id="call-1",
                    arguments={"path": str(workspace_root / "sample_project" / "new_file.txt")},
                ),
                tool_ctx=ctx,
                session=session,
                user_prompt="delete from there",
            )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertEqual(observation["reason"], "delete_path_missed_recent_file")
        self.assertEqual(observation["recent_matches"][0]["path"], str(recent_path))

    def test_preflight_short_circuits_already_deleted_recent_file(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            deleted_path = workspace_root / "sample_project" / "web_ui" / "new_file.txt"
            session = AgentSession(
                recent_files=[
                    {
                        "path": str(deleted_path),
                        "action": "delete",
                        "status": "deleted",
                    }
                ]
            )
            ctx = ToolContext(
                rust=RustTools(Path("/tmp/nym-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )

            observation = _preflight_tool_call(
                ModelToolCall(
                    name="delete_path",
                    call_id="call-1",
                    arguments={"path": str(deleted_path)},
                ),
                tool_ctx=ctx,
                session=session,
                user_prompt="delete it from sample_project",
            )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["already_absent"])
        self.assertEqual(observation["path"], str(deleted_path))

    def test_missing_delete_path_error_is_marked_recoverable(self) -> None:
        observation = _annotate_tool_observation(
            "delete_path",
            {"path": "/workspace/sample_project/web_ui/README.md"},
            {
                "ok": False,
                "tool": "delete_path",
                "error": "Path does not exist or cannot be inspected: /workspace/sample_project/web_ui/README.md",
            },
        )

        self.assertTrue(observation["recoverable"])
        self.assertEqual(observation["failed_path"], "/workspace/sample_project/web_ui/README.md")
        self.assertIn("incorrect guess", observation["guidance"])

    def test_missing_inspect_tree_error_is_marked_recoverable(self) -> None:
        observation = _annotate_tool_observation(
            "inspect_tree",
            {"path": "sample_projet"},
            {
                "ok": False,
                "tool": "inspect_tree",
                "error": "Path does not exist: /workspace/sample_projet",
            },
        )

        self.assertTrue(observation["recoverable"])
        self.assertEqual(observation["failed_path"], "sample_projet")
        self.assertIn("incorrect guess", observation["guidance"])

    def test_non_missing_tool_error_is_not_marked_recoverable(self) -> None:
        observation = _annotate_tool_observation(
            "delete_path",
            {"path": "/workspace/sample_project"},
            {
                "ok": False,
                "tool": "delete_path",
                "error": "Permission denied",
            },
        )

        self.assertNotIn("recoverable", observation)

    def test_session_updates_from_directory_read_path(self) -> None:
        session = AgentSession()

        _update_session_from_tool_result(
            session,
            tool="read_path",
            args={"path": "/workspace/sample_project"},
            observation={
                "path": "/workspace/sample_project",
                "detection": {"kind": "directory"},
                "content": "api_service/\nweb_ui/\nREADME.md",
            },
            workspace_root=Path("/workspace"),
        )

        self.assertEqual(session.active_root, "/workspace/sample_project")
        self.assertEqual(
            session.last_candidates,
            [
                {"path": "/workspace/sample_project/api_service", "kind": "directory"},
                {"path": "/workspace/sample_project/web_ui", "kind": "directory"},
                {"path": "/workspace/sample_project/README.md", "kind": "file"},
            ],
        )

    def test_session_updates_from_inspect_tree_direct_children(self) -> None:
        session = AgentSession()

        _update_session_from_tool_result(
            session,
            tool="inspect_tree",
            args={"path": "/workspace/sample_project"},
            observation={
                "path": "/workspace/sample_project",
                "kind": "directory",
                "tree": [
                    {"path": "sample_project/api_service", "kind": "directory"},
                    {"path": "sample_project/api_service/manage.py", "kind": "file"},
                    {"path": "sample_project/web_ui", "kind": "directory"},
                ],
            },
            workspace_root=Path("/workspace"),
        )

        self.assertEqual(session.active_root, "/workspace/sample_project")
        self.assertEqual(
            session.last_candidates,
            [
                {"path": "/workspace/sample_project/api_service", "kind": "directory"},
                {"path": "/workspace/sample_project/web_ui", "kind": "directory"},
            ],
        )

    def test_session_prefers_inspect_tree_direct_children_field(self) -> None:
        session = AgentSession()

        _update_session_from_tool_result(
            session,
            tool="inspect_tree",
            args={"path": "/workspace/sample_project"},
            observation={
                "path": "/workspace/sample_project",
                "kind": "directory",
                "direct_children": [
                    {"path": "sample_project/api_service", "kind": "directory"},
                    {"path": "sample_project/web_ui", "kind": "directory"},
                ],
                "tree": [
                    {"path": "sample_project/api_service/manage.py", "kind": "file"},
                ],
            },
            workspace_root=Path("/workspace"),
        )

        self.assertEqual(session.active_root, "/workspace/sample_project")
        self.assertEqual(
            session.last_candidates,
            [
                {"path": "/workspace/sample_project/api_service", "kind": "directory"},
                {"path": "/workspace/sample_project/web_ui", "kind": "directory"},
            ],
        )

    def test_run_agent_updates_session_after_tool_call(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def respond(self, **kwargs: Any) -> Any:
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(
                        output=[
                            {
                                "type": "function_call",
                                "name": "glob",
                                "call_id": "call-1",
                                "arguments": '{"pattern":"sample_project","kind":"directory"}',
                            }
                        ],
                        output_text="",
                    )
                return SimpleNamespace(output=[], output_text="done")

        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                return {
                    "matches": [
                        {"path": "/workspace/sample_project", "kind": "directory"},
                    ],
                    "truncated": False,
                    "backend": "fake",
                }

        session = AgentSession()

        answer = run_agent(
            llm=FakeLLM(),  # type: ignore[arg-type]
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root="/workspace",
            user_prompt="find sample_project",
            session=session,
        )

        self.assertEqual(answer, "done")
        self.assertEqual(session.active_root, "/workspace/sample_project")
        self.assertEqual(session.focus_paths, ["/workspace/sample_project"])
        self.assertEqual(session.last_candidates, [{"path": "/workspace/sample_project", "kind": "directory"}])

    def test_run_agent_waits_for_approval_before_external_delete(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def respond(self, **kwargs: Any) -> Any:
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(
                        output=[
                            {
                                "type": "function_call",
                                "name": "delete_path",
                                "call_id": "call-1",
                                "arguments": '{"path":"/tmp/external.txt"}',
                            }
                        ],
                        output_text="",
                    )
                return SimpleNamespace(output=[], output_text="done")

        class FakeRust:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def delete_path(self, **kwargs: object) -> dict[str, object]:
                self.calls.append(kwargs)
                return {"ok": True, "path": str(kwargs["path"])}

        session = AgentSession()
        seen_requests: list[dict[str, Any]] = []

        def approve_request(request: dict[str, Any]) -> str:
            seen_requests.append(dict(request))
            return "approved"

        answer = run_agent(
            llm=FakeLLM(),  # type: ignore[arg-type]
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root="/workspace",
            user_prompt="delete the file in /tmp/external.txt",
            session=session,
            approval_requester=approve_request,
        )

        self.assertEqual(answer, "done")
        self.assertEqual(len(seen_requests), 1)
        self.assertEqual(seen_requests[0]["reason"], "external_delete_requires_confirmation")
        self.assertEqual(seen_requests[0]["requested_path"], "/tmp/external.txt")
        self.assertIn("/tmp/external.txt", session.approved_external_delete_roots)
        self.assertTrue(session.pending_approvals)
        self.assertEqual(session.pending_approvals[0]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
