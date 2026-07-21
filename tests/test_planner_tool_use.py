import unittest
import hashlib
import os
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from agent.main import default_workspace_root, resolve_rust_bin
from agent.planner import (
    AgentSession,
    LOCAL_GATE_PROMPT,
    ModelToolCall,
    _approval_request_from_observation,
    _build_initial_messages,
    _prepare_tool_output,
    _preflight_tool_call,
    _is_explicit_delete_request,
    _verify_mutation_observation,
    _update_session_from_tool_result,
    agent_session_from_dict,
    agent_session_to_dict,
    run_agent,
)
from agent.prompt_loader import load_system_prompt
from agent.rust_tools import RustTools
from agent.language_servers import LanguageServerManager, LanguageServerSpec
from agent.tools import ToolContext, _context_file_paths, build_tool_registry


class PlannerToolUseTests(unittest.TestCase):
    def test_system_prompt_mentions_write_file_tool(self) -> None:
        prompt = load_system_prompt()
        self.assertIn("write_file", prompt)
        self.assertIn("Preserve user intent", prompt)
        self.assertIn("Resolve entities before acting", prompt)
        self.assertIn("named target", prompt)
        self.assertIn("Treat credentials, secrets, API keys", prompt)
        self.assertIn("secret_scan", prompt)
        self.assertIn("system_info", prompt)
        self.assertIn("connected_devices", prompt)
        self.assertIn("desktop_capabilities", prompt)
        self.assertIn("desktop_observe", prompt)
        self.assertIn("run_system_command", prompt)
        self.assertIn("desktop_action", prompt)
        self.assertIn("generated, dependency, cache, and build output", prompt)
        self.assertIn("Failed path operations only prove that the attempted path failed", prompt)
        self.assertIn("recent file operations", prompt)
        self.assertIn("finish_task", prompt)
        self.assertIn("A partial inspection", prompt)

    def test_build_initial_messages_appends_current_prompt_after_history(self) -> None:
        messages = _build_initial_messages(
            workspace_root="/workspace",
            context_text="",
            session=AgentSession(),
            user_prompt="current request",
            conversation_history=[
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
            ],
        )

        self.assertEqual(messages[-1], {"role": "user", "content": "current request"})

    def test_tool_registry_exposes_edit_and_delete_tools(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/agent-rust")),
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)
        tool_names = {schema["name"] for schema in registry.schemas()}
        self.assertIn("write_file", tool_names)
        self.assertIn("edit_file", tool_names)
        self.assertIn("delete_path", tool_names)
        self.assertIn("path_status", tool_names)
        self.assertIn("secret_scan", tool_names)
        self.assertIn("language_server", tool_names)
        self.assertIn("system_info", tool_names)
        self.assertIn("connected_devices", tool_names)
        self.assertIn("desktop_capabilities", tool_names)
        self.assertIn("desktop_observe", tool_names)
        self.assertIn("desktop_resolve", tool_names)
        self.assertIn("process_list", tool_names)
        self.assertIn("run_system_command", tool_names)
        self.assertIn("desktop_action", tool_names)
        self.assertIn("desktop_clipboard_files", tool_names)
        desktop_action = next(schema for schema in registry.schemas() if schema["name"] == "desktop_action")
        actions = desktop_action["parameters"]["properties"]["action"]["enum"]
        self.assertIn("focus_window", actions)
        self.assertIn("close_window", actions)
        self.assertIn("clipboard_write", actions)
        self.assertIn("send_key", actions)
        self.assertIn("type_text", actions)
        self.assertIn("mouse_click", actions)
        self.assertIn("scroll", actions)

    def test_device_tool_schema_accepts_runtime_discovered_categories(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/agent-rust")),
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        schemas = {schema["name"]: schema for schema in build_tool_registry(ctx).schemas()}
        scope = schemas["connected_devices"]["parameters"]["properties"]["scope"]

        self.assertNotIn("enum", scope)
        self.assertIn("discovered category", scope["description"])

    def test_desktop_capabilities_is_read_only(self) -> None:
        class FakeRust:
            def desktop_capabilities(self) -> dict[str, object]:
                return {
                    "ok": True,
                    "tool": "desktop_capabilities",
                    "actions": [{"action": "set_volume", "available": True}],
                }

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute("desktop_capabilities", {}, ctx)

        self.assertEqual(result["tool"], "desktop_capabilities")
        self.assertFalse(result.get("blocked", False))

    def test_desktop_observe_is_read_only_and_bounded(self) -> None:
        class FakeRust:
            def desktop_observe(self, *, scope: str, limit: int) -> dict[str, object]:
                return {
                    "ok": True,
                    "tool": "desktop_observe",
                    "scope": scope,
                    "limit": limit,
                }

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_observe",
            {"scope": "windows", "limit": 500},
            ctx,
        )

        self.assertEqual(result["tool"], "desktop_observe")
        self.assertEqual(result["scope"], "windows")
        self.assertEqual(result["limit"], 200)
        self.assertFalse(result.get("blocked", False))

    def test_desktop_observe_accepts_clipboard_scope(self) -> None:
        class FakeRust:
            def desktop_observe(self, *, scope: str, limit: int) -> dict[str, object]:
                return {"ok": True, "tool": "desktop_observe", "scope": scope, "limit": limit}

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_observe",
            {"scope": "clipboard"},
            ctx,
        )

        self.assertEqual(result["scope"], "clipboard")

    def test_desktop_observe_accepts_ui_tree_scope(self) -> None:
        class FakeRust:
            def desktop_observe(self, *, scope: str, limit: int) -> dict[str, object]:
                return {"ok": True, "tool": "desktop_observe", "scope": scope, "limit": limit}

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_observe",
            {"scope": "ui_tree", "limit": 500},
            ctx,
        )

        self.assertEqual(result["scope"], "ui_tree")
        self.assertEqual(result["limit"], 200)

    def test_desktop_observe_accepts_displays_scope(self) -> None:
        class FakeRust:
            def desktop_observe(self, *, scope: str, limit: int) -> dict[str, object]:
                return {"ok": True, "tool": "desktop_observe", "scope": scope, "limit": limit}

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_observe",
            {"scope": "displays", "limit": 10},
            ctx,
        )

        self.assertEqual(result["scope"], "displays")
        self.assertEqual(result["limit"], 10)

    def test_desktop_observe_accepts_audio_scope(self) -> None:
        class FakeRust:
            def desktop_observe(self, *, scope: str, limit: int) -> dict[str, object]:
                return {"ok": True, "tool": "desktop_observe", "scope": scope, "limit": limit}

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_observe",
            {"scope": "audio"},
            ctx,
        )

        self.assertEqual(result["scope"], "audio")

    def test_desktop_observe_accepts_dialogs_scope(self) -> None:
        class FakeRust:
            def desktop_observe(self, *, scope: str, limit: int) -> dict[str, object]:
                return {"ok": True, "tool": "desktop_observe", "scope": scope, "limit": limit}

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_observe",
            {"scope": "dialogs"},
            ctx,
        )

        self.assertEqual(result["scope"], "dialogs")

    def test_desktop_observe_accepts_downloads_scope(self) -> None:
        class FakeRust:
            def desktop_observe(self, *, scope: str, limit: int) -> dict[str, object]:
                return {"ok": True, "tool": "desktop_observe", "scope": scope, "limit": limit}

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_observe", {"scope": "downloads", "limit": 25}, ctx
        )

        self.assertEqual(result["scope"], "downloads")
        self.assertEqual(result["limit"], 25)

    def test_desktop_resolve_is_read_only_and_bounded(self) -> None:
        class FakeRust:
            def desktop_resolve(self, *, query: str, kind: str, limit: int) -> dict[str, object]:
                return {
                    "ok": True,
                    "tool": "desktop_resolve",
                    "query": query,
                    "kind": kind,
                    "limit": limit,
                    "candidates": [],
                }

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_resolve",
            {"query": "vim", "kind": "application", "limit": 500},
            ctx,
        )

        self.assertEqual(result["tool"], "desktop_resolve")
        self.assertEqual(result["query"], "vim")
        self.assertEqual(result["kind"], "application")
        self.assertEqual(result["limit"], 50)

    def test_desktop_action_requires_exact_approval(self) -> None:
        class FakeRust:
            def desktop_action(self, **_kwargs: Any) -> dict[str, object]:
                raise AssertionError("desktop action must not run before approval")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_action",
            {"action": "set_volume", "value": "25"},
            ctx,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "desktop_action_requires_approval")
        self.assertEqual(result["requested_path"], "desktop set_volume 25")

    def test_volume_action_ignores_empty_optional_target_from_local_model(self) -> None:
        class FakeRust:
            def desktop_action(self, **_kwargs: Any) -> dict[str, object]:
                raise AssertionError("desktop action must not run before approval")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_action",
            {"action": "set_volume", "target": "", "value": "0"},
            ctx,
        )

        self.assertEqual(result["reason"], "desktop_action_requires_approval")
        self.assertEqual(result["requested_path"], "desktop set_volume 0")

    def test_window_action_requires_target_before_approval(self) -> None:
        class FakeRust:
            def desktop_action(self, **_kwargs: Any) -> dict[str, object]:
                raise AssertionError("window action must not run without target")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_action",
            {"action": "focus_window"},
            ctx,
        )

        self.assertEqual(result["reason"], "desktop_action_target_required")

    def test_clipboard_write_approval_key_redacts_content(self) -> None:
        class FakeRust:
            def desktop_action(self, **_kwargs: Any) -> dict[str, object]:
                raise AssertionError("clipboard write must not run before approval")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_action",
            {"action": "clipboard_write", "value": "secret clipboard text"},
            ctx,
        )

        self.assertEqual(result["reason"], "desktop_action_requires_approval")
        self.assertIn("sha256:", result["requested_path"])
        self.assertIn("bytes:", result["requested_path"])
        self.assertNotIn("secret clipboard text", result["requested_path"])

    def test_file_clipboard_requires_exact_path_set_approval(self) -> None:
        class FakeRust:
            def desktop_clipboard_files(self, **_kwargs: Any) -> dict[str, object]:
                raise AssertionError("file clipboard must not run before approval")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.txt"
            source.write_text("private contents", encoding="utf-8")
            ctx = ToolContext(
                rust=FakeRust(),  # type: ignore[arg-type]
                workspace_root=root,
                search_roots=[],
            )
            result = build_tool_registry(ctx).execute(
                "desktop_clipboard_files",
                {"paths": ["notes.txt"], "operation": "copy"},
                ctx,
            )

        self.assertEqual(result["reason"], "desktop_action_requires_approval")
        self.assertIn("sha256:", result["requested_path"])
        self.assertIn("items:1", result["requested_path"])
        self.assertNotIn("private contents", str(result))

    def test_file_clipboard_runs_after_exact_approval(self) -> None:
        class FakeRust:
            def desktop_clipboard_files(self, **kwargs: Any) -> dict[str, object]:
                return {"ok": True, "verified": True, **kwargs}

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.txt"
            source.write_text("contents", encoding="utf-8")
            initial = ToolContext(rust=FakeRust(), workspace_root=root, search_roots=[])  # type: ignore[arg-type]
            blocked = build_tool_registry(initial).execute(
                "desktop_clipboard_files", {"paths": ["notes.txt"]}, initial
            )
            approved = ToolContext(
                rust=FakeRust(),  # type: ignore[arg-type]
                workspace_root=root,
                search_roots=[],
                approved_system_commands=[blocked["requested_path"]],
            )
            result = build_tool_registry(approved).execute(
                "desktop_clipboard_files", {"paths": ["notes.txt"]}, approved
            )

        self.assertTrue(result["verified"])
        self.assertEqual(result["paths"], [str(source)])

    def test_type_text_approval_key_redacts_content(self) -> None:
        class FakeRust:
            def desktop_action(self, **_kwargs: Any) -> dict[str, object]:
                raise AssertionError("type_text must not run before approval")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_action",
            {"action": "type_text", "value": "private typed text"},
            ctx,
        )

        self.assertEqual(result["reason"], "desktop_action_requires_approval")
        self.assertIn("sha256:", result["requested_path"])
        self.assertNotIn("private typed text", result["requested_path"])

    def test_send_key_requires_value(self) -> None:
        class FakeRust:
            def desktop_action(self, **_kwargs: Any) -> dict[str, object]:
                raise AssertionError("send_key must not run without value")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_action",
            {"action": "send_key"},
            ctx,
        )

        self.assertEqual(result["reason"], "desktop_action_value_required")

    def test_mouse_click_requires_coordinates(self) -> None:
        class FakeRust:
            def desktop_action(self, **_kwargs: Any) -> dict[str, object]:
                raise AssertionError("mouse_click must not run without coordinates")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_action",
            {"action": "mouse_click"},
            ctx,
        )

        self.assertEqual(result["reason"], "desktop_action_target_required")

    def test_scroll_requires_value(self) -> None:
        class FakeRust:
            def desktop_action(self, **_kwargs: Any) -> dict[str, object]:
                raise AssertionError("scroll must not run without value")

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_action",
            {"action": "scroll"},
            ctx,
        )

        self.assertEqual(result["reason"], "desktop_action_value_required")

    def test_desktop_action_runs_after_exact_approval(self) -> None:
        class FakeRust:
            def desktop_action(self, **kwargs: Any) -> dict[str, object]:
                return {"ok": True, "verified": True, **kwargs}

        ctx = ToolContext(
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
            approved_system_commands=["desktop set_volume 25"],
        )
        result = build_tool_registry(ctx).execute(
            "desktop_action",
            {"action": "set_volume", "value": "25"},
            ctx,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["action"], "set_volume")

    def test_desktop_action_observation_creates_approval_request(self) -> None:
        request = _approval_request_from_observation(
            ModelToolCall(
                name="desktop_action",
                call_id="call-desktop",
                arguments={"action": "bluetooth_connect", "target": "AA:BB:CC:DD:EE:FF"},
            ),
            {
                "ok": False,
                "blocked": True,
                "recoverable": True,
                "reason": "desktop_action_requires_approval",
                "operation": "desktop",
                "requested_path": "desktop bluetooth_connect AA:BB:CC:DD:EE:FF",
            },
            user_prompt="connect my headphones",
            workspace_root=Path("/workspace"),
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request["operation"], "desktop")
        self.assertEqual(request["tool"], "desktop_action")

    def test_write_mutation_is_verified_against_file_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "created.txt"
            target.write_text("real content")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()

            result = _verify_mutation_observation(
                "write_file",
                {"path": "created.txt", "content": "real content"},
                {"path": str(target), "after_sha256": digest},
                workspace_root=workspace,
            )

        self.assertTrue(result["verified"])
        self.assertEqual(result["verification"]["sha256"], digest)

    def test_write_success_report_is_rejected_when_file_was_not_created(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "missing.txt"

            result = _verify_mutation_observation(
                "write_file",
                {"path": "missing.txt", "content": "claimed content"},
                {"path": str(target), "after_sha256": "0" * 64},
                workspace_root=workspace,
            )

        self.assertFalse(result["ok"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "mutation_verification_failed")

    def test_delete_success_report_is_rejected_when_target_still_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "still-here.txt"
            target.write_text("keep")

            result = _verify_mutation_observation(
                "delete_path",
                {"path": str(target)},
                {"path": str(target), "deleted": True},
                workspace_root=workspace,
            )

        self.assertFalse(result["ok"])
        self.assertIn("still exists", result["error"])

    def test_path_status_detects_empty_project_with_git_tracked_deletions(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project = workspace / "project"
            project.mkdir()
            tracked = project / "app.py"
            tracked.write_text("print('hello')\n")
            subprocess_args = {"cwd": workspace, "check": True, "capture_output": True}

            subprocess.run(["git", "init"], **subprocess_args)
            subprocess.run(["git", "add", "project/app.py"], **subprocess_args)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Agent Tests",
                    "-c",
                    "user.email=agent@example.invalid",
                    "commit",
                    "-m",
                    "fixture",
                ],
                **subprocess_args,
            )
            tracked.unlink()
            ctx = ToolContext(
                rust=RustTools(Path("/tmp/agent-rust")),
                workspace_root=workspace,
                search_roots=[],
            )

            result = build_tool_registry(ctx).execute("path_status", {"path": "project"}, ctx)

        self.assertTrue(result["exists"])
        self.assertTrue(result["empty"])
        self.assertEqual(result["git_change_count"], 1)
        self.assertEqual(result["tracked_deleted_count"], 1)
        self.assertEqual(result["state"], "tracked_content_deleted")

    def test_delete_verification_is_not_explicit_delete_request(self) -> None:
        self.assertFalse(_is_explicit_delete_request("has my project been deleted?"))
        self.assertFalse(_is_explicit_delete_request("check whether project/ was removed"))
        self.assertTrue(_is_explicit_delete_request("delete project/"))

    def test_preflight_blocks_delete_for_read_only_verification(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/agent-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="delete_path",
                call_id="call-verify",
                arguments={"path": "project"},
            ),
            tool_ctx=ctx,
            user_prompt="can you tell me if my project has been deleted or not?",
        )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertEqual(observation["reason"], "delete_not_explicitly_requested")

    def test_preflight_blocks_active_workspace_root_deletion(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/agent-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="delete_path",
                call_id="call-delete-root",
                arguments={"path": "/workspace", "recursive": True},
            ),
            tool_ctx=ctx,
            user_prompt="delete this project",
        )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertEqual(observation["reason"], "workspace_root_delete_blocked")

    def test_session_remembers_desktop_observe_targets(self) -> None:
        session = AgentSession()

        _update_session_from_tool_result(
            session,
            tool="desktop_observe",
            args={"scope": "windows"},
            observation={
                "ok": True,
                "tool": "desktop_observe",
                "snapshot_id": "desktop-123",
                "observed_at_unix_ms": 123,
                "scope": "windows",
                "windows": {
                    "items": [
                        {"id": "0x3a00007", "title": "Editor"},
                    ],
                },
            },
            workspace_root=Path("/workspace"),
        )

        self.assertEqual(session.last_desktop_snapshot["snapshot_id"], "desktop-123")
        self.assertEqual(session.desktop_targets[0]["kind"], "window")
        self.assertEqual(session.desktop_targets[0]["target"], "0x3a00007")

    def test_session_remembers_snapshot_bound_ui_elements(self) -> None:
        session = AgentSession()

        _update_session_from_tool_result(
            session,
            tool="desktop_observe",
            args={"scope": "ui_tree"},
            observation={
                "ok": True,
                "tool": "desktop_observe",
                "snapshot_id": "desktop-456",
                "observed_at_unix_ms": 456,
                "scope": "ui_tree",
                "ui_tree": {
                    "items": [{
                        "id": "ui-deadbeef",
                        "snapshot_id": "desktop-456",
                        "name": "Save",
                        "role": "push button",
                        "backend_ref": {
                            "bus": ":1.42",
                            "path": "/org/example/save",
                        },
                    }],
                },
            },
            workspace_root=Path("/workspace"),
        )

        target = session.desktop_targets[0]
        self.assertEqual(target["kind"], "ui_element")
        self.assertEqual(target["target"], "ui-deadbeef")
        self.assertEqual(target["snapshot_id"], "desktop-456")
        self.assertEqual(target["backend_bus"], ":1.42")

    def test_session_remembers_controls_from_dialog_observation(self) -> None:
        session = AgentSession()
        backend = {"bus": ":1.42", "path": "/org/example/dialog"}
        control_backend = {"bus": ":1.42", "path": "/org/example/dialog/save"}

        _update_session_from_tool_result(
            session,
            tool="desktop_observe",
            args={"scope": "dialogs"},
            observation={
                "ok": True,
                "tool": "desktop_observe",
                "snapshot_id": "desktop-dialog",
                "scope": "dialogs",
                "dialogs": {
                    "items": [{
                        "id": "ui-dialog",
                        "snapshot_id": "desktop-dialog",
                        "role": "file chooser",
                        "backend_ref": backend,
                        "controls": [{
                            "id": "ui-save",
                            "snapshot_id": "desktop-dialog",
                            "name": "Save",
                            "role": "push button",
                            "actions": ["click"],
                            "backend_ref": control_backend,
                        }],
                    }],
                },
            },
            workspace_root=Path("/workspace"),
        )

        targets = {item["id"]: item for item in session.desktop_targets}
        self.assertEqual(targets["ui-dialog"]["role"], "file chooser")
        self.assertEqual(targets["ui-save"]["name"], "Save")
        self.assertEqual(targets["ui-save"]["actions_json"], '["click"]')

    def test_preflight_resolves_latest_ui_element_backend_reference(self) -> None:
        session = AgentSession(
            last_desktop_snapshot={"snapshot_id": "desktop-456"},
            desktop_targets=[{
                "kind": "ui_element",
                "id": "ui-deadbeef",
                "target": "ui-deadbeef",
                "snapshot_id": "desktop-456",
                "backend_bus": ":1.42",
                "backend_path": "/org/example/save",
                "actions_json": '["click"]',
            }],
        )
        call = ModelToolCall(
            name="desktop_action",
            call_id="invoke-save",
            arguments={"action": "invoke_element", "target": "ui-deadbeef", "value": "click"},
        )

        result = _preflight_tool_call(
            call,
            tool_ctx=ToolContext(
                rust=RustTools(Path("/tmp/agent-rust")),
                workspace_root=Path("/workspace"),
                search_roots=[],
            ),
            session=session,
            user_prompt="click Save",
        )

        self.assertIsNone(result)
        self.assertEqual(call.arguments["_backend_bus"], ":1.42")
        self.assertEqual(call.arguments["_backend_path"], "/org/example/save")

    def test_preflight_rejects_stale_ui_element(self) -> None:
        session = AgentSession(
            last_desktop_snapshot={"snapshot_id": "desktop-new"},
            desktop_targets=[{
                "kind": "ui_element",
                "id": "ui-old",
                "target": "ui-old",
                "snapshot_id": "desktop-old",
                "backend_bus": ":1.42",
                "backend_path": "/org/example/old",
            }],
        )

        result = _preflight_tool_call(
            ModelToolCall(
                name="desktop_action",
                call_id="focus-old",
                arguments={"action": "focus_element", "target": "ui-old"},
            ),
            tool_ctx=ToolContext(
                rust=RustTools(Path("/tmp/agent-rust")),
                workspace_root=Path("/workspace"),
                search_roots=[],
            ),
            session=session,
            user_prompt="focus it",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "desktop_element_not_observed")

    def test_preflight_blocks_unobserved_window_action(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/agent-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="desktop_action",
                call_id="focus-1",
                arguments={"action": "focus_window", "target": "0x3a00007"},
            ),
            tool_ctx=ctx,
            session=AgentSession(),
            user_prompt="focus that window",
        )

        self.assertIsNotNone(observation)
        self.assertTrue(observation["blocked"])
        self.assertEqual(observation["reason"], "desktop_target_not_observed")

    def test_preflight_allows_observed_window_action(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/agent-rust")),
            workspace_root=Path("/workspace"),
            search_roots=[],
        )
        session = AgentSession(
            desktop_targets=[
                {
                    "kind": "window",
                    "id": "0x3a00007",
                    "target": "0x3a00007",
                    "action": "focus_window",
                }
            ]
        )

        observation = _preflight_tool_call(
            ModelToolCall(
                name="desktop_action",
                call_id="focus-1",
                arguments={"action": "focus_window", "target": "60817415"},
            ),
            tool_ctx=ctx,
            session=session,
            user_prompt="focus that window",
        )

        self.assertIsNone(observation)

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
            rust=RustTools(Path("/tmp/agent-rust")),
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
            rust=RustTools(Path("/tmp/agent-rust")),
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
            rust=RustTools(Path("/tmp/agent-rust")),
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
                rust=RustTools(Path("/tmp/agent-rust")),
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
            workspace_root = Path(tmp) / "agent"
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
                {"path": "agent", "kind": "directory"},
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
                rust=RustTools(Path("/tmp/agent-rust")),
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

    def test_resolve_rust_bin_prefers_repo_binary_over_workspace_parent(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "agent"
            workspace_root = tmp_path / "app1"
            stale_root = tmp_path

            repo_bin = repo_root / "agent-rust" / "target" / "debug" / "agent-rust"
            stale_bin = stale_root / "agent-rust" / "target" / "debug" / "agent-rust"

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

    def test_default_workspace_root_uses_parent_when_running_from_agent_checkout(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            repo_root = workspace_root / "agent"
            (workspace_root / "agent-rust").mkdir()
            (workspace_root / "README.md").write_text("workspace\n")
            (repo_root / "agent-rust").mkdir(parents=True)
            (repo_root / "README.md").write_text("repo\n")

            old_cwd = Path.cwd()
            try:
                os.chdir(repo_root)
                resolved = default_workspace_root(Namespace(root=None))
            finally:
                os.chdir(old_cwd)

        self.assertEqual(resolved, workspace_root.resolve())

    def test_default_workspace_root_does_not_fall_back_to_home_from_root(self) -> None:
        old_cwd = Path.cwd()
        try:
            os.chdir("/")
            resolved = default_workspace_root(Namespace(root=None))
        finally:
            os.chdir(old_cwd)

        self.assertEqual(resolved, Path(__file__).resolve().parents[1])
        self.assertNotEqual(resolved, Path.home().resolve())

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

        with patch("agent.tools._translate_windows_path", return_value=Path("/mnt/c/Users")):
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

        with patch("agent.tools._translate_windows_path", return_value=None):
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
            "agent.tools._translate_windows_path",
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
                rust=RustTools(Path("/tmp/agent-rust")),
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

    def test_inspect_tree_blocks_recursive_scan_of_multi_project_workspace_root(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            (workspace_root / "RA_publish").mkdir()
            (workspace_root / "RA_clean").mkdir()
            (workspace_root / "agent").mkdir()
            ctx = ToolContext(
                rust=RustTools(Path("/tmp/agent-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )

            observation = build_tool_registry(ctx).execute(
                "inspect_tree",
                {"path": str(workspace_root)},
                ctx,
            )

        self.assertFalse(observation["ok"])
        self.assertEqual(observation["reason"], "workspace_container_requires_target")
        child_paths = {item["path"] for item in observation["direct_children"]}
        self.assertTrue({"RA_publish", "RA_clean", "agent"} <= child_paths)

    def test_inspect_tree_caps_metadata_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "repo"
            project.mkdir()
            (project / "pyproject.toml").write_text("[project]\nname='repo'\n")
            for index in range(50):
                (project / f"file_{index:02}.txt").write_text("value\n")
            ctx = ToolContext(
                rust=RustTools(Path("/tmp/agent-rust")),
                workspace_root=workspace_root,
                search_roots=[],
            )

            observation = build_tool_registry(ctx).execute(
                "inspect_tree",
                {"path": "repo", "max_entries": 10},
                ctx,
            )

        self.assertEqual(len(observation["tree"]), 10)
        self.assertTrue(observation["tree_truncated"])
        self.assertTrue(observation["truncated"])

    def test_inspect_tree_reads_unknown_utf8_file_extensions(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "custom_project"
            project.mkdir()
            custom_file = project / "workflow.agentdsl"
            custom_file.write_text("step build\nstep verify\n")

            ctx = ToolContext(
                rust=RustTools(Path("/tmp/agent-rust")),
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
        self.assertEqual(observation["files"][0]["path"], "custom_project/workflow.agentdsl")
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
                rust=RustTools(Path("/tmp/agent-rust")),
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
                rust=RustTools(Path("/tmp/agent-rust")),
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
                rust=RustTools(Path("/tmp/agent-rust")),
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
                rust=RustTools(Path("/tmp/agent-rust")),
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
                rust=RustTools(Path("/tmp/agent-rust")),
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

            with patch("agent.tools._git_root_for", return_value=None):
                paths = _context_file_paths(child)

        self.assertEqual(paths, [child / "AGENTS.md"])

    def test_context_file_paths_respect_env_names_and_git_boundary(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp)
            project = workspace_root / "repo"
            child = project / "child"
            child.mkdir(parents=True)
            (project / ".git").mkdir()
            (project / "AGENT_HINTS.md").write_text("root hints\n")
            (child / "AGENT_HINTS.md").write_text("child hints\n")

            with patch.dict(os.environ, {"CONTEXT_FILE_NAMES": "AGENT_HINTS.md"}):
                paths = _context_file_paths(child)

        self.assertEqual(paths, [project / "AGENT_HINTS.md", child / "AGENT_HINTS.md"])

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

    def test_inspect_target_recovers_model_added_project_suffix(self) -> None:
        class FakeRust:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def inspect_target(self, **kwargs: object) -> dict[str, object]:
                query = str(kwargs["path"])
                self.queries.append(query)
                if query == "ra_project":
                    return {"status": "not_found", "query": query}
                return {
                    "status": "candidates",
                    "query": query,
                    "candidates": [
                        {"path": "RA_clean", "kind": "directory"},
                        {"path": "RA_publish", "kind": "directory"},
                    ],
                }

        with TemporaryDirectory() as tmp:
            rust = FakeRust()
            ctx = ToolContext(
                rust=rust,  # type: ignore[arg-type]
                workspace_root=Path(tmp),
                search_roots=[],
            )

            observation = build_tool_registry(ctx).execute(
                "inspect_target",
                {"path": "ra_project", "kind": "directory"},
                ctx,
            )

        self.assertEqual(rust.queries, ["ra_project", "ra"])
        self.assertEqual(observation["recovered_query"], "ra")
        self.assertEqual(
            [item["path"] for item in observation["candidates"]],
            ["RA_clean", "RA_publish"],
        )

    def test_agent_session_round_trips_path_context(self) -> None:
        session = AgentSession(
            active_root="/workspace/sample_project",
            focus_paths=["/workspace/sample_project/api_service"],
            last_candidates=[{"path": "/workspace/sample_project/web_ui", "kind": "directory"}],
            reasoning_effort="high",
            approved_system_commands=["restart_service docker"],
            desktop_targets=[
                {
                    "kind": "window",
                    "id": "0x3a00007",
                    "target": "0x3a00007",
                    "title": "Editor",
                    "action": "focus_window",
                    "source": "desktop_observe",
                }
            ],
            last_desktop_snapshot={
                "snapshot_id": "desktop-1",
                "scope": "windows",
                "observed_at_unix_ms": "123",
            },
            tool_loop_history=[
                {
                    "tool": "glob",
                    "args_hash": "args",
                    "result_hash": "result",
                    "run_id": "run-1",
                }
            ],
        )

        restored = agent_session_from_dict(agent_session_to_dict(session))

        self.assertEqual(restored.active_root, session.active_root)
        self.assertEqual(restored.focus_paths, session.focus_paths)
        self.assertEqual(restored.last_candidates, session.last_candidates)
        self.assertEqual(restored.recent_files, session.recent_files)
        self.assertEqual(restored.reasoning_effort, "high")
        self.assertEqual(restored.approved_system_commands, session.approved_system_commands)
        self.assertEqual(restored.tool_loop_history, session.tool_loop_history)
        self.assertEqual(restored.desktop_targets, session.desktop_targets)
        self.assertEqual(restored.last_desktop_snapshot, session.last_desktop_snapshot)

    def test_system_command_mutation_requires_approval(self) -> None:
        ctx = ToolContext(
            rust=RustTools(Path("/tmp/agent-rust")),
            workspace_root=Path("/tmp"),
            search_roots=[],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "run_system_command",
            {"command": "restart_service", "target": "docker"},
            ctx,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "system_command_requires_approval")
        self.assertEqual(result["requested_path"], "restart_service docker")

    def test_system_command_runs_after_session_approval(self) -> None:
        class FakeRust:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def run_system_command(self, *, command: str, target: str | None = None, limit: int = 50) -> dict[str, object]:
                self.calls.append({"command": command, "target": target, "limit": limit})
                return {"ok": True, "tool": "run_system_command", "command": command, "target": target}

        rust = FakeRust()
        ctx = ToolContext(
            rust=rust,  # type: ignore[arg-type]
            workspace_root=Path("/tmp"),
            search_roots=[],
            approved_system_commands=["restart_service docker"],
        )
        registry = build_tool_registry(ctx)

        result = registry.execute(
            "run_system_command",
            {"command": "restart_service", "target": "docker", "limit": 5},
            ctx,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(rust.calls[0]["command"], "restart_service")
        self.assertEqual(rust.calls[0]["target"], "docker")

    def test_system_command_approval_request_is_created(self) -> None:
        call = ModelToolCall(
            name="run_system_command",
            call_id="call-1",
            arguments={"command": "restart_service", "target": "docker"},
        )

        request = _approval_request_from_observation(
            call,
            {
                "ok": False,
                "tool": "run_system_command",
                "blocked": True,
                "recoverable": True,
                "reason": "system_command_requires_approval",
                "operation": "system",
                "requested_path": "restart_service docker",
                "guidance": "Ask for approval.",
            },
            user_prompt="restart docker",
            workspace_root=Path("/workspace"),
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request["operation"], "system")
        self.assertEqual(request["requested_path"], "restart_service docker")

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
        self.assertIn("choose tools and resolve references", content)

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
        self.assertIn("choose tools and resolve references", content)

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

        llm = FakeLLM()
        answer = run_agent(
            llm=llm,  # type: ignore[arg-type]
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root="/workspace",
            user_prompt="find sample_project",
            session=session,
        )

        self.assertEqual(answer, "done")
        self.assertEqual(session.active_root, "/workspace/sample_project")
        self.assertEqual(session.focus_paths, ["/workspace/sample_project"])
        self.assertEqual(session.last_candidates, [{"path": "/workspace/sample_project", "kind": "directory"}])

    def test_run_agent_accepts_plain_assistant_response_as_completion(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def respond(self, **kwargs: Any) -> Any:
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(output=[], output_text="A few things stood out.")
                return SimpleNamespace(
                    output=[
                        {
                            "type": "function_call",
                            "name": "finish_task",
                            "call_id": "finish-1",
                            "arguments": '{"answer":"final answer"}',
                        }
                    ],
                    output_text="",
                )

        llm = FakeLLM()
        answer = run_agent(
            llm=llm,  # type: ignore[arg-type]
            rust=object(),
            workspace_root="/workspace",
            user_prompt="inspect the repo",
        )

        self.assertEqual(answer, "A few things stood out.")
        self.assertEqual(llm.calls, 1)

    def test_local_host_prompt_requires_desktop_observation_before_answer(self) -> None:
        class FakeLLM:
            mode = "local"

            def __init__(self) -> None:
                self.calls = 0
                self.instructions: list[str] = []
                self.tool_names: list[set[str]] = []

            def respond(self, **kwargs: Any) -> Any:
                self.calls += 1
                self.instructions.append(str(kwargs.get("instructions") or ""))
                self.tool_names.append({
                    str(tool.get("name"))
                    for tool in kwargs.get("tools", [])
                    if isinstance(tool, dict)
                })
                if self.calls == 1:
                    return SimpleNamespace(
                        output=[],
                        output_text="Likely processes include systemd and bash.",
                    )
                if self.calls == 2:
                    return SimpleNamespace(
                        output=[{
                            "type": "function_call",
                            "name": "desktop_observe",
                            "call_id": "observe-1",
                            "arguments": '{"scope":"applications","limit":20}',
                        }],
                        output_text="",
                    )
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "finish_task",
                        "call_id": "finish-1",
                        "arguments": '{"answer":"Open applications: Terminal."}',
                    }],
                    output_text="",
                )

        class FakeRust:
            def __init__(self) -> None:
                self.observed: list[dict[str, object]] = []

            def desktop_observe(self, **kwargs: object) -> dict[str, object]:
                self.observed.append(dict(kwargs))
                return {
                    "ok": True,
                    "tool": "desktop_observe",
                    "scope": kwargs["scope"],
                    "applications": {
                        "items": [{"name": "Terminal", "id": "terminal.desktop"}],
                    },
                }

        llm = FakeLLM()
        rust = FakeRust()
        answer = run_agent(
            llm=llm,  # type: ignore[arg-type]
            rust=rust,  # type: ignore[arg-type]
            workspace_root="/workspace",
            user_prompt="can you tell me what all applications are open",
        )

        self.assertEqual(answer, "Open applications: Terminal.")
        self.assertEqual(rust.observed, [{"scope": "applications", "limit": 20}])
        self.assertNotIn(LOCAL_GATE_PROMPT, llm.instructions)
        self.assertIn("desktop_observe", llm.tool_names[0])
        self.assertNotIn("glob", llm.tool_names[0])

    def test_run_agent_accepts_model_finish_task_without_prompt_classification(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def respond(self, **kwargs: Any) -> Any:
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(
                        output=[{
                            "type": "function_call",
                            "name": "finish_task",
                            "call_id": "finish-1",
                            "arguments": '{"answer":"Created the file."}',
                        }],
                        output_text="",
                    )
                return SimpleNamespace(output=[], output_text="Created the file.")

        with TemporaryDirectory() as tmp:
            answer = run_agent(
                llm=FakeLLM(),  # type: ignore[arg-type]
                rust=object(),
                workspace_root=tmp,
                user_prompt="create a new file named proof.txt",
                max_steps=1,
            )

        self.assertEqual(answer, "Created the file.")

    def test_run_agent_reports_fake_write_result_to_model_as_unverified(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0
                self.messages: list[list[dict[str, Any]]] = []

            def respond(self, **kwargs: Any) -> Any:
                self.calls += 1
                self.messages.append(list(kwargs.get("messages", [])))
                if self.calls == 1:
                    return SimpleNamespace(
                        output=[{
                            "type": "function_call",
                            "name": "write_file",
                            "call_id": "write-1",
                            "arguments": '{"path":"proof.txt","content":"hello"}',
                        }],
                        output_text="",
                    )
                return SimpleNamespace(output=[], output_text="Created proof.txt.")

        class FakeRust:
            def write_file(self, **kwargs: object) -> dict[str, object]:
                return {
                    "path": str(kwargs["path"]),
                    "created": True,
                    "after_sha256": "0" * 64,
                }

        llm = FakeLLM()
        with TemporaryDirectory() as tmp:
            answer = run_agent(
                llm=llm,  # type: ignore[arg-type]
                rust=FakeRust(),  # type: ignore[arg-type]
                workspace_root=tmp,
                user_prompt="create a new file named proof.txt",
            )

            self.assertFalse((Path(tmp) / "proof.txt").exists())

        self.assertEqual(answer, "Created proof.txt.")
        self.assertIn("mutation_verification_failed", json.dumps(llm.messages[1]))

    def test_run_agent_accepts_plain_text_clarification(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def respond(self, **kwargs: Any) -> Any:
                self.calls += 1
                return SimpleNamespace(
                    output=[],
                    output_text=(
                        "Your workspace contains multiple possible RA projects. "
                        "Choose one target before I continue."
                    ),
                )

        answer = run_agent(
            llm=FakeLLM(),  # type: ignore[arg-type]
            rust=object(),
            workspace_root="/workspace",
            user_prompt="tell me about RA",
        )

        self.assertIn("multiple possible RA projects", answer)

    def test_local_obvious_chat_skips_router_call(self) -> None:
        class FakeLLM:
            mode = "local"

            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def respond(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                return SimpleNamespace(output=[], output_text="Hi there.")

        llm = FakeLLM()
        answer = run_agent(
            llm=llm,  # type: ignore[arg-type]
            rust=object(),
            workspace_root="/workspace",
            user_prompt="hi",
        )

        self.assertEqual(answer, "Hi there.")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["tools"], [])

    def test_run_agent_waits_for_approval_before_external_delete(self) -> None:
        class FakeLLM:
            def __init__(self, target: Path) -> None:
                self.calls = 0
                self.target = target

            def respond(self, **kwargs: Any) -> Any:
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(
                        output=[
                            {
                                "type": "function_call",
                                "name": "delete_path",
                                "call_id": "call-1",
                                "arguments": json.dumps({"path": str(self.target)}),
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
                path = Path(str(kwargs["path"]))
                path.unlink()
                return {"ok": True, "path": str(path), "deleted": True}

        session = AgentSession()
        seen_requests: list[dict[str, Any]] = []

        def approve_request(request: dict[str, Any]) -> str:
            seen_requests.append(dict(request))
            return "approved"

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            external = Path(tmp) / "external.txt"
            external.write_text("delete me")
            answer = run_agent(
                llm=FakeLLM(external),  # type: ignore[arg-type]
                rust=FakeRust(),  # type: ignore[arg-type]
                workspace_root=str(workspace),
                user_prompt=f"delete the file in {external}",
                session=session,
                approval_requester=approve_request,
            )

            self.assertFalse(external.exists())

        self.assertEqual(answer, "done")
        self.assertEqual(len(seen_requests), 1)
        self.assertEqual(seen_requests[0]["reason"], "external_delete_requires_confirmation")
        self.assertEqual(seen_requests[0]["requested_path"], str(external))
        self.assertIn(str(external), session.approved_external_delete_roots)
        self.assertTrue(session.pending_approvals)
        self.assertEqual(session.pending_approvals[0]["status"], "pending")

    def test_run_agent_trims_tool_names_before_dispatch(self) -> None:
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
                                "name": " glob ",
                                "call_id": "call-1",
                                "arguments": '{"pattern":"*.py"}',
                            }
                        ],
                        output_text="",
                    )
                return SimpleNamespace(output=[], output_text="done")

        class FakeRust:
            def glob_files(self, **kwargs: object) -> dict[str, object]:
                return {"matches": [], "truncated": False}

        answer = run_agent(
            llm=FakeLLM(),  # type: ignore[arg-type]
            rust=FakeRust(),  # type: ignore[arg-type]
            workspace_root="/workspace",
            user_prompt="find python files",
        )

        self.assertEqual(answer, "done")

    def test_run_agent_breaks_repeated_unknown_tool_loop(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def respond(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                if kwargs.get("tools"):
                    return SimpleNamespace(
                        output=[
                            {
                                "type": "function_call",
                                "name": "exec",
                                "call_id": f"call-{len(self.calls)}",
                                "arguments": json.dumps({"command": f"echo {len(self.calls)}"}),
                            }
                        ],
                        output_text="",
                    )
                return SimpleNamespace(output=[], output_text="Answered without exec.")

        llm = FakeLLM()
        answer = run_agent(
            llm=llm,  # type: ignore[arg-type]
            rust=object(),
            workspace_root="/workspace",
            user_prompt="run a shell command",
            max_steps=20,
        )

        self.assertEqual(answer, "Answered without exec.")
        self.assertEqual(len(llm.calls), 11)
        self.assertEqual(llm.calls[-1]["tools"], [])
        self.assertIn("unavailable tool exec 10 times", llm.calls[-1]["messages"][-1]["content"])

    def test_run_agent_retries_one_empty_assistant_response(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def respond(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return SimpleNamespace(output=[], output_text="")
                return SimpleNamespace(output=[], output_text="Visible answer.")

        llm = FakeLLM()
        answer = run_agent(
            llm=llm,  # type: ignore[arg-type]
            rust=object(),
            workspace_root="/workspace",
            user_prompt="answer me",
            max_steps=3,
        )

        self.assertEqual(answer, "Visible answer.")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn(
            "previous attempt did not produce a user-visible answer",
            llm.calls[1]["messages"][-1]["content"],
        )

    def test_run_agent_blocks_generic_repeated_no_progress_tool_loop(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def respond(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                if "generic_repeat" in json.dumps(kwargs.get("messages", [])):
                    return SimpleNamespace(output=[], output_text="Stopped after loop block.")
                return SimpleNamespace(
                    output=[
                        {
                            "type": "function_call",
                            "name": "glob",
                            "call_id": f"call-{len(self.calls)}",
                            "arguments": '{"pattern":"*.missing"}',
                        }
                    ],
                    output_text="",
                )

        class FakeRust:
            def __init__(self) -> None:
                self.calls = 0

            def glob_files(self, **kwargs: object) -> dict[str, object]:
                self.calls += 1
                return {"matches": [], "truncated": False}

        llm = FakeLLM()
        rust = FakeRust()
        session = AgentSession()
        answer = run_agent(
            llm=llm,  # type: ignore[arg-type]
            rust=rust,  # type: ignore[arg-type]
            workspace_root="/workspace",
            user_prompt="keep searching",
            session=session,
            max_steps=25,
        )

        self.assertEqual(answer, "Stopped after loop block.")
        self.assertEqual(rust.calls, 20)
        self.assertTrue(session.tool_loop_history)
        self.assertTrue(all(record.get("run_id") for record in session.tool_loop_history))

        second_llm = FakeLLM()
        second_answer = run_agent(
            llm=second_llm,  # type: ignore[arg-type]
            rust=rust,  # type: ignore[arg-type]
            workspace_root="/workspace",
            user_prompt="new turn",
            session=session,
            max_steps=3,
        )

        self.assertNotEqual(second_answer, "Stopped after loop block.")


if __name__ == "__main__":
    unittest.main()
