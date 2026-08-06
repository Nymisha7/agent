import io
import json
import os
import subprocess
from contextlib import nullcontext, redirect_stdout
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent.main import (
    LiveTurnState,
    TUI_TRANSCRIPT_LIMIT,
    _activate_local_runtime,
    _api_key_prompt_provider,
    _approval_panel_lines,
    _approval_display_text,
    _cli_approval_requester,
    _compact_usage_text,
    _complete_slash_command,
    _ensure_ollama_running,
    _expire_orphaned_approvals,
    _exclusive_bridge_turn,
    _handle_local_command,
    _is_exit_command,
    _local_runtime_status_lines,
    _load_agent_name,
    _load_tui_copy_keys,
    _load_tui_mouse_capture,
    _load_tui_paste_keys,
    _provider_api_key_needed,
    _load_persisted_api_keys,
    _persist_api_key,
    _persist_agent_name,
    _queue_status,
    _record_local_command_exchange,
    _redact_local_command,
    _render_tui_transcript,
    _run_tui_bridge,
    _slash_command_lines,
    _slash_palette_entries,
    _selectable_palette_index,
    _tui_bridge_completions,
    _tui_bridge_apply_approval_decision,
    _tui_bridge_snapshot,
    _usage_panel_lines,
    build_parser,
    handle_prompt,
    main,
    repl,
    run_tui,
)
from agent.planner import AgentSession
from agent.session_store import SessionStore, TokenUsage
from agent.system_events import enqueue_system_event, reset_system_events_for_test


class TuiExitTests(unittest.TestCase):
    def test_plain_cli_approval_asks_once_for_exact_target(self) -> None:
        request = {
            "tool": "delete_path",
            "operation": "delete",
            "resolved_path": "/home/nymisha/test78.py",
        }

        with patch("builtins.input", return_value="yes") as read_input:
            decision = _cli_approval_requester(request)

        self.assertEqual(decision, "approved")
        read_input.assert_called_once()
        self.assertIn("delete", read_input.call_args.args[0])
        self.assertIn("/home/nymisha/test78.py", read_input.call_args.args[0])

    def test_plain_repl_connects_structured_approval_requester(self) -> None:
        ctx = SimpleNamespace(
            debug=False,
            language_servers=None,
            session_id="plain-cli-test",
        )
        captured: list[object] = []

        def handle(_ctx: object, _prompt: str, **kwargs: object) -> str:
            captured.append(kwargs.get("approval_requester"))
            return "Deleted."

        with (
            patch("builtins.input", side_effect=["delete that project", "/exit"]),
            patch("agent.main._handle_local_command", return_value=None),
            patch("agent.main.handle_prompt", side_effect=handle),
            patch("agent.main._stop_language_servers"),
            redirect_stdout(io.StringIO()),
        ):
            result = repl(ctx)

        self.assertEqual(result, 0)
        self.assertEqual(captured, [_cli_approval_requester])

    def test_api_key_credentials_persist_with_private_permissions(self) -> None:
        with TemporaryDirectory() as config_home:
            with patch.dict(
                "os.environ",
                {"XDG_CONFIG_HOME": config_home},
                clear=True,
            ):
                _persist_api_key("OPENAI_API_KEY", "sk-persisted")
                credential_path = Path(config_home) / "agent" / "credentials.enc"
                key_path = Path(config_home) / "agent" / "credentials.key"
                self.assertEqual(credential_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)
                self.assertNotIn(b"sk-persisted", credential_path.read_bytes())
                self.assertNotIn("OPENAI_API_KEY", os.environ)

                _load_persisted_api_keys()

                self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-persisted")

    def test_plaintext_credentials_are_migrated_to_encrypted_storage(self) -> None:
        with TemporaryDirectory() as config_home:
            credential_dir = Path(config_home) / "agent"
            credential_dir.mkdir()
            legacy = credential_dir / "credentials.json"
            legacy.write_text('{"OPENAI_API_KEY":"sk-legacy"}', encoding="utf-8")
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": config_home}, clear=True):
                _load_persisted_api_keys()

                encrypted = credential_dir / "credentials.enc"
                self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-legacy")
                self.assertTrue(encrypted.is_file())
                self.assertNotIn(b"sk-legacy", encrypted.read_bytes())
                self.assertFalse(legacy.exists())

    def test_bridge_turn_lock_rejects_second_active_turn_for_same_session(self) -> None:
        with _exclusive_bridge_turn("single-active-session"):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                with _exclusive_bridge_turn("single-active-session"):
                    self.fail("a second bridge turn must not acquire the session lock")

    def test_run_tui_launches_ratatui_subcommand(self) -> None:
        ctx = SimpleNamespace(
            rust=SimpleNamespace(rust_bin=Path("/tmp/agent-rust")),
            session_id="abc123",
            language_servers=None,
        )

        with (
            patch("agent.main.sys.stdin.isatty", return_value=True),
            patch("agent.main.sys.stdout.isatty", return_value=True),
            patch("agent.main.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run_subprocess,
        ):
            self.assertEqual(run_tui(ctx), 0)

        command = run_subprocess.call_args.args[0]
        self.assertEqual(command[0], "/tmp/agent-rust")
        self.assertEqual(command[1], "tui")
        self.assertIn("--session-id", command)
        self.assertIn("abc123", command)
        paste_key_indexes = [
            index for index, value in enumerate(command) if value == "--paste-key"
        ]
        paste_keys = [command[index + 1] for index in paste_key_indexes]
        self.assertIn("ctrl+v", paste_keys)
        self.assertIn("ctrl+shift+v", paste_keys)
        self.assertIn("shift+insert", paste_keys)
        self.assertIn("alt+v", paste_keys)
        copy_key_indexes = [
            index for index, value in enumerate(command) if value == "--copy-key"
        ]
        copy_keys = [command[index + 1] for index in copy_key_indexes]
        self.assertIn("alt+c", copy_keys)
        self.assertIn("ctrl+y", copy_keys)
        self.assertIn("--mouse-capture", command)

    def test_stream_bridge_startup_failure_emits_final_frame(self) -> None:
        output = io.StringIO()
        with (
            patch("agent.main.SessionStore.default", side_effect=RuntimeError("readonly database")),
            redirect_stdout(output),
        ):
            rc = main([
                "--tui-bridge",
                "stream-submit",
                "--bridge-session-id",
                "abc123",
                "--bridge-prompt",
                "hi",
            ])

        payload = json.loads(output.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(payload["kind"], "final")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "readonly database")

    def test_queue_status_shows_pending_prompt_count(self) -> None:
        self.assertEqual(_queue_status("Thinking...", 0), "Thinking...")
        self.assertEqual(_queue_status("Thinking...", 2), "Thinking... | queued 2")


class TuiRenderingTests(unittest.TestCase):
    def test_agent_name_persists_to_preferences(self) -> None:
        with TemporaryDirectory() as config_home:
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": config_home}, clear=True):
                self.assertEqual(_load_agent_name(), "Agent")

                saved = _persist_agent_name("Nymisha Helper")

                self.assertEqual(saved, "Nymisha Helper")
                self.assertEqual(_load_agent_name(), "Nymisha Helper")
                preferences = Path(config_home) / "agent" / "preferences.json"
                self.assertEqual(preferences.stat().st_mode & 0o777, 0o600)

    def test_tui_paste_keys_include_user_configured_shortcuts(self) -> None:
        with TemporaryDirectory() as config_home:
            preferences = Path(config_home) / "agent" / "preferences.json"
            preferences.parent.mkdir(parents=True)
            preferences.write_text('{"paste_keys":["alt+v"," ctrl + alt + p "]}\n', encoding="utf-8")
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": config_home}, clear=True):
                paste_keys = _load_tui_paste_keys()

        self.assertEqual(
            paste_keys[:4],
            ("ctrl+v", "ctrl+shift+v", "shift+insert", "alt+v"),
        )
        self.assertIn("alt+v", paste_keys)
        self.assertIn("ctrl+alt+p", paste_keys)

    def test_tui_copy_keys_include_user_configured_shortcuts(self) -> None:
        with TemporaryDirectory() as config_home:
            preferences = Path(config_home) / "agent" / "preferences.json"
            preferences.parent.mkdir(parents=True)
            preferences.write_text('{"copy_keys":["alt+shift+c"]}\n', encoding="utf-8")
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": config_home}, clear=True):
                copy_keys = _load_tui_copy_keys()

        self.assertEqual(copy_keys[:2], ("alt+c", "ctrl+y"))
        self.assertIn("alt+shift+c", copy_keys)

    def test_tui_mouse_capture_is_enabled_for_in_app_selection_by_default(self) -> None:
        with TemporaryDirectory() as config_home:
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": config_home}, clear=True):
                self.assertTrue(_load_tui_mouse_capture())

    def test_tui_mouse_capture_can_be_disabled_from_preferences(self) -> None:
        with TemporaryDirectory() as config_home:
            preferences = Path(config_home) / "agent" / "preferences.json"
            preferences.parent.mkdir(parents=True)
            preferences.write_text('{"mouse_capture": false}\n', encoding="utf-8")
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": config_home}, clear=True):
                self.assertFalse(_load_tui_mouse_capture())

    def test_name_command_shows_and_changes_agent_name(self) -> None:
        with TemporaryDirectory() as config_home:
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": config_home}, clear=True):
                ctx = SimpleNamespace(agent_name="Agent")

                current = _handle_local_command(ctx, "/name")
                changed = _handle_local_command(ctx, "/name Nymi")

                self.assertIn("Agent name: Agent", current)
                self.assertEqual(changed, "Okay, my name is now Nymi.")
                self.assertEqual(ctx.agent_name, "Nymi")
                self.assertEqual(_load_agent_name(), "Nymi")

    def test_name_command_accepts_apostrophes_without_shell_quoting(self) -> None:
        with TemporaryDirectory() as config_home:
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": config_home}, clear=True):
                ctx = SimpleNamespace(agent_name="Agent")

                changed = _handle_local_command(ctx, "/name Nymisha's Helper")

                self.assertEqual(changed, "Okay, my name is now Nymisha's Helper.")
                self.assertEqual(_load_agent_name(), "Nymisha's Helper")

    def test_render_tui_transcript_uses_custom_agent_name(self) -> None:
        messages = [
            SimpleNamespace(role="assistant", content="Ready.", created_at="now", attachments=[]),
        ]

        rendered = _render_tui_transcript(messages, {}, 80, agent_name="Nymi")

        self.assertIn("Nymi  now", rendered)
        self.assertNotIn("Agent  now", rendered)

    def test_tools_are_agent_managed_and_not_selectable(self) -> None:
        ctx = SimpleNamespace()

        text = _handle_local_command(ctx, "/tools")

        self.assertIn("managed automatically", text)
        self.assertFalse(any(entry.value == "/tools" for entry in _slash_palette_entries("/")))

    def test_devices_and_capabilities_are_not_slash_commands(self) -> None:
        ctx = SimpleNamespace()

        self.assertEqual(_handle_local_command(ctx, "/devices"), "Unknown local command: /devices")
        self.assertEqual(_handle_local_command(ctx, "/capabilities"), "Unknown local command: /capabilities")
        self.assertNotIn("/devices", "\n".join(_slash_command_lines("/", 80)))
        self.assertNotIn("/capabilities", "\n".join(_slash_command_lines("/", 80)))
        self.assertFalse(any(entry.value in {"/devices", "/capabilities"} for entry in _slash_palette_entries("/")))

    def test_attachments_are_composer_actions_not_slash_commands(self) -> None:
        hidden = {"/attach", "/attachments", "/screenshot"}

        for command in hidden:
            self.assertEqual(
                _handle_local_command(SimpleNamespace(), command),
                f"Unknown local command: {command}",
            )
        self.assertTrue(hidden.isdisjoint({entry.value for entry in _slash_palette_entries("/")}))
        help_text = "\n".join(_slash_command_lines("/", 100))
        self.assertTrue(all(command not in help_text for command in hidden))

    def test_private_composer_bridge_still_adds_selected_file(self) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "notes.txt"
            source.write_text("attachment content", encoding="utf-8")
            ctx = SimpleNamespace(
                session_id="attachment-session",
                pending_attachments=[],
                session=AgentSession(),
                store=SimpleNamespace(save_agent_state=Mock()),
            )
            with patch.dict("os.environ", {"XDG_DATA_HOME": tmp}):
                result = _handle_local_command(ctx, f'/__nym_attach "{source}"')

        self.assertIn("Attached for next message: notes.txt", result)
        self.assertEqual([item.filename for item in ctx.pending_attachments], ["notes.txt"])
        self.assertEqual(len(ctx.session.pending_attachments), 1)

    def test_bridge_local_command_round_trip_includes_prompt_and_answer(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(
                workspace_root=root,
                provider="openai",
                model="gpt-4o-mini",
            )
            args = build_parser().parse_args([
                "--tui-bridge",
                "submit",
                "--bridge-session-id",
                session.id,
                "--bridge-prompt",
                "/help",
            ])
            stdout = io.StringIO()

            with (
                patch.dict("os.environ", {"NYM_AGENT_NAME": "Nymi"}, clear=False),
                redirect_stdout(stdout),
            ):
                exit_code = _run_tui_bridge(args, store)

            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["snapshot"]["agent_name"], "Nymi")
        self.assertEqual(
            [item["role"] for item in payload["snapshot"]["messages"]],
            ["user", "assistant"],
        )
        self.assertEqual(payload["snapshot"]["messages"][0]["content"], "/help")
        self.assertIn("Commands", payload["snapshot"]["messages"][1]["content"])
        self.assertEqual(
            payload["command_result"],
            {"code": "ok", "setup_required": False, "error": False},
        )

    def test_normal_prompt_round_trip_persists_user_and_assistant_messages(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(workspace_root=root)
            llm = SimpleNamespace(
                reset_turn_usage=Mock(),
                consume_turn_usage=Mock(return_value={}),
                estimate_cost_usd=Mock(return_value=0.0),
            )
            ctx = SimpleNamespace(
                store=store,
                session_id=session.id,
                llm=llm,
                rust=object(),
                workspace_root=root,
                search_roots=[],
                session=AgentSession(),
                stored_context=None,
                language_servers=None,
                debug=False,
            )

            with patch("agent.main.run_agent", return_value="Agent answer"):
                answer = handle_prompt(ctx, "User prompt")

            messages = store.list_messages(session.id, limit=None)
            updated = store.get_session(session.id)

        self.assertEqual(answer, "Agent answer")
        self.assertEqual(
            [(message.role, message.content) for message in messages],
            [("user", "User prompt"), ("assistant", "Agent answer")],
        )
        self.assertEqual(updated.last_prompt, "User prompt")
        self.assertEqual(updated.title, "User prompt")

    def test_system_events_are_prefixed_to_agent_prompt_without_changing_transcript(self) -> None:
        reset_system_events_for_test()
        try:
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = SessionStore(root / "sessions.sqlite3")
                session = store.create_session(workspace_root=root)
                llm = SimpleNamespace(
                    reset_turn_usage=Mock(),
                    consume_turn_usage=Mock(return_value={}),
                    estimate_cost_usd=Mock(return_value=0.0),
                )
                ctx = SimpleNamespace(
                    store=store,
                    session_id=session.id,
                    llm=llm,
                    rust=object(),
                    workspace_root=root,
                    search_roots=[],
                    session=AgentSession(),
                    stored_context=None,
                    language_servers=None,
                    debug=False,
                )
                enqueue_system_event("Node: worker · mode busy", session_key="agent:main:main")

                with patch("agent.main.run_agent", return_value="Agent answer") as run:
                    answer = handle_prompt(ctx, "User prompt")

                messages = store.list_messages(session.id, limit=None)

            self.assertEqual(answer, "Agent answer")
            self.assertIn("System events since the last prompt:", run.call_args.kwargs["user_prompt"])
            self.assertIn("Node: worker · mode busy", run.call_args.kwargs["user_prompt"])
            self.assertEqual(
                [(message.role, message.content) for message in messages],
                [("user", "User prompt"), ("assistant", "Agent answer")],
            )
        finally:
            reset_system_events_for_test()

    def test_normal_prompt_rejects_rebounded_route_before_agent_loop(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            session, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
            )
            replacement = store.create_session(workspace_root=root, title="replacement")
            with store._connect() as conn:
                conn.execute(
                    "update session_routes set session_id = ? where route_key = ?",
                    (replacement.id, route_key),
                )
            llm = SimpleNamespace(
                reset_turn_usage=Mock(),
                consume_turn_usage=Mock(return_value={}),
                estimate_cost_usd=Mock(return_value=0.0),
            )
            ctx = SimpleNamespace(
                store=store,
                session_id=session.id,
                route_key=route_key,
                llm=llm,
                rust=object(),
                workspace_root=root,
                search_roots=[],
                session=AgentSession(),
                stored_context=None,
                language_servers=None,
                debug=False,
            )

            with (
                patch("agent.main.run_agent", return_value="Agent answer") as run_agent,
                self.assertRaisesRegex(RuntimeError, "Session route rebound"),
            ):
                handle_prompt(ctx, "User prompt")

            run_agent.assert_not_called()
            self.assertEqual(store.list_messages(session.id, limit=None), [])

    def test_normal_prompt_publishes_user_and_assistant_transcript_updates(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            session, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
            )
            llm = SimpleNamespace(
                reset_turn_usage=Mock(),
                consume_turn_usage=Mock(return_value={}),
                estimate_cost_usd=Mock(return_value=0.0),
            )
            hooks = SimpleNamespace(emit=Mock())
            ctx = SimpleNamespace(
                store=store,
                session_id=session.id,
                route_key=route_key,
                agent_id="main",
                gateway=SimpleNamespace(
                    hooks=hooks,
                    session_lease=Mock(return_value=nullcontext()),
                ),
                llm=llm,
                rust=object(),
                workspace_root=root,
                search_roots=[],
                session=AgentSession(),
                stored_context=None,
                language_servers=None,
                debug=False,
            )

            with patch("agent.main.run_agent", return_value="Agent answer"):
                handle_prompt(ctx, "User prompt")

        transcript_updates = [
            call.args[1]
            for call in hooks.emit.call_args_list
            if call.args[0] == "session_transcript_updated"
        ]
        self.assertEqual(
            [update["message"] for update in transcript_updates],
            [
                {"role": "user", "content": "User prompt"},
                {"role": "assistant", "content": "Agent answer"},
            ],
        )
        self.assertEqual(transcript_updates[0]["message_seq"], 1)
        self.assertEqual(transcript_updates[1]["message_seq"], 2)
        self.assertEqual(
            transcript_updates[0]["target"],
            {
                "agent_id": "main",
                "session_id": session.id,
                "session_key": route_key,
            },
        )

    def test_local_command_exchange_is_persisted_and_secrets_are_redacted(self) -> None:
        store = SimpleNamespace(
            add_messages=Mock(return_value=[
                SimpleNamespace(id=1, seq=1, role="user", content="/apikey anthropic <redacted>"),
                SimpleNamespace(id=2, seq=2, role="assistant", content="Anthropic key loaded; ready."),
            ]),
        )
        ctx = SimpleNamespace(session_id="abc123", store=store)

        _record_local_command_exchange(
            ctx,
            "/apikey anthropic sk-ant-secret",
            "Anthropic key loaded; ready.",
        )

        self.assertEqual(
            store.add_messages.call_args,
            unittest.mock.call(
                "abc123",
                [
                    ("user", "/apikey anthropic <redacted>"),
                    ("assistant", "Anthropic key loaded; ready."),
                ],
                last_prompt="/apikey anthropic <redacted>",
            ),
        )

    def test_local_command_exchange_uses_route_guard_when_available(self) -> None:
        store = SimpleNamespace(
            add_messages=Mock(return_value=[
                SimpleNamespace(id=1, seq=1, role="user", content="/help"),
                SimpleNamespace(id=2, seq=2, role="assistant", content="Commands"),
            ]),
        )
        ctx = SimpleNamespace(
            session_id="abc123",
            route_key="agent:main:tui:alice",
            store=store,
        )

        _record_local_command_exchange(ctx, "/help", "Commands")

        self.assertEqual(
            store.add_messages.call_args.kwargs["expected_route_key"],
            "agent:main:tui:alice",
        )

    def test_local_command_exchange_publishes_transcript_update_after_write(self) -> None:
        store = SimpleNamespace(
            add_messages=Mock(return_value=[
                SimpleNamespace(id=1, seq=1, role="user", content="/help"),
                SimpleNamespace(id=2, seq=2, role="assistant", content="Commands"),
            ]),
        )
        hooks = SimpleNamespace(emit=Mock())
        ctx = SimpleNamespace(
            session_id="abc123",
            route_key="agent:main:tui:alice",
            agent_id="main",
            store=store,
            gateway=SimpleNamespace(hooks=hooks),
        )

        _record_local_command_exchange(ctx, "/help", "Commands")

        hooks.emit.assert_called_once()
        event, payload = hooks.emit.call_args.args
        self.assertEqual(event, "session_transcript_updated")
        self.assertEqual(payload["message"], {"role": "assistant", "content": "Commands"})
        self.assertEqual(payload["message_id"], "2")
        self.assertEqual(payload["message_seq"], 2)
        self.assertEqual(
            payload["target"],
            {
                "agent_id": "main",
                "session_id": "abc123",
                "session_key": "agent:main:tui:alice",
            },
        )

    def test_tui_bridge_completions_returns_model_command_entries(self) -> None:
        payload = _tui_bridge_completions("/mo")

        self.assertEqual(payload["title"], "Commands")
        self.assertEqual(payload["entries"][0]["label"], "/model")

    def test_complete_bridge_skips_full_agent_context_and_runtime_probes(self) -> None:
        args = SimpleNamespace(
            bridge_session_id="palette-session",
            bridge_prompt="/mo",
            tui_bridge="complete",
        )
        output = io.StringIO()

        with (
            patch("agent.main.build_context") as build_context,
            patch("agent.main._discover_local_provider_availability") as discover,
            redirect_stdout(output),
        ):
            result = _run_tui_bridge(args, SimpleNamespace())

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["completions"]["entries"][0]["value"], "/model")
        build_context.assert_not_called()
        discover.assert_not_called()

    def test_tui_bridge_model_completions_are_not_capped(self) -> None:
        payload = _tui_bridge_completions("/model ")
        expected_count = len(_slash_palette_entries("/model "))

        self.assertGreater(expected_count, 12)
        self.assertEqual(len(payload["entries"]), expected_count)

    def test_tui_bridge_model_completions_show_provider_and_runtime_state(self) -> None:
        ctx = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="gpt-5.5",
                configuration_error=None,
            )
        )
        availability = {
            provider: ([], "offline")
            for provider in {"ollama", "lmstudio", "llamacpp", "vllm", "localai"}
        }
        availability["ollama"] = (["qwen3:latest"], None)

        with patch(
            "agent.main._discover_local_provider_availability",
            return_value=availability,
        ):
            payload = _tui_bridge_completions("/model ", ctx=ctx)

        entries = {entry["value"]: entry for entry in payload["entries"]}
        self.assertEqual(
            entries["ollama/qwen3"]["description"],
            "Ollama · Ready · 8B params · ~5.2 GB · 40K ctx",
        )
        self.assertEqual(
            entries["ollama/llama3.3"]["description"],
            "Ollama · Not installed · 70B params · ~43 GB · 128K ctx",
        )
        self.assertNotIn("gemini/gemini-2.5-pro", entries)
        labels = [entry["label"] for entry in payload["entries"]]
        self.assertIn("── OpenAI ──", labels)
        self.assertIn("── Ollama ──", labels)
        self.assertNotIn("── Ready / installed ──", labels)
        self.assertNotIn("── Needs setup / not installed ──", labels)
        ollama_section = labels.index("── Ollama ──")
        self.assertEqual(
            labels[ollama_section + 1 : ollama_section + 4],
            ["qwen2.5-coder", "qwen3-coder", "qwen3"],
        )

    def test_model_completions_are_grouped_by_provider_ranked_models_first(self) -> None:
        ctx = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="gpt-5.5",
                configuration_error=None,
            )
        )
        availability = {
            provider: ([], "offline")
            for provider in {"ollama", "lmstudio", "llamacpp", "vllm", "localai"}
        }
        availability["ollama"] = (["qwen3:latest", "custom-local:latest"], None)
        availability["llamacpp"] = (["qwen2.5-coder-7b-instruct"], None)

        with patch(
            "agent.main._discover_local_provider_availability",
            return_value=availability,
        ):
            payload = _tui_bridge_completions("/model ", ctx=ctx)

        values = [entry["value"] for entry in payload["entries"] if entry["execute"]]
        self.assertLess(values.index("openai/gpt-5.5"), values.index("anthropic/claude-sonnet-4.5"))
        self.assertLess(values.index("anthropic/claude-sonnet-4.5"), values.index("ollama/qwen2.5-coder"))
        self.assertLess(values.index("ollama/qwen2.5-coder"), values.index("ollama/qwen3-coder"))
        self.assertLess(values.index("ollama/qwen3-coder"), values.index("ollama/qwen3"))
        self.assertLess(values.index("ollama/qwen3"), values.index("ollama/custom-local:latest"))
        self.assertLess(values.index("ollama/qwen3:latest"), values.index("ollama/custom-local:latest"))
        self.assertLess(
            values.index("llamacpp/gemma-3-1b-it"),
            values.index("llamacpp/qwen2.5-coder-7b-instruct"),
        )

    def test_model_completions_select_current_model_first(self) -> None:
        ctx = SimpleNamespace(
            llm=SimpleNamespace(
                provider="ollama",
                model="qwen3",
                configuration_error=None,
            )
        )
        availability = {
            provider: ([], "offline")
            for provider in {"ollama", "lmstudio", "llamacpp", "vllm", "localai"}
        }
        availability["ollama"] = (["qwen3:latest"], None)

        with patch(
            "agent.main._discover_local_provider_availability",
            return_value=availability,
        ):
            payload = _tui_bridge_completions("/model ", ctx=ctx)

        selected = payload["entries"][payload["selected_index"]]
        self.assertEqual(selected["value"], "ollama/qwen3")
        self.assertTrue(selected["execute"])

    def test_model_complete_bridge_uses_stored_session_model(self) -> None:
        args = SimpleNamespace(
            bridge_session_id="palette-session",
            bridge_prompt="/model ",
            tui_bridge="complete",
        )
        store = SimpleNamespace(
            get_session=lambda _session_id: SimpleNamespace(
                provider="ollama",
                model="qwen3",
            )
        )
        output = io.StringIO()
        availability = {
            provider: ([], "offline")
            for provider in {"ollama", "lmstudio", "llamacpp", "vllm", "localai"}
        }
        availability["ollama"] = (["qwen3:latest"], None)

        with (
            patch("agent.main.build_context") as build_context,
            patch("agent.main._discover_local_provider_availability", return_value=availability),
            redirect_stdout(output),
        ):
            result = _run_tui_bridge(args, store)

        payload = json.loads(output.getvalue())
        selected = payload["completions"]["entries"][payload["completions"]["selected_index"]]
        self.assertEqual(result, 0)
        self.assertEqual(selected["value"], "ollama/qwen3")
        build_context.assert_not_called()

    def test_tui_bridge_snapshot_includes_agent_name(self) -> None:
        ctx = SimpleNamespace(
            session_id="name-session",
            agent_name="Nymi",
            llm=SimpleNamespace(model="gpt-5.4-mini", provider="openai", mode="hosted", configuration_error=None),
            session=SimpleNamespace(pending_approvals=[]),
            store=SimpleNamespace(
                get_session=lambda _: SimpleNamespace(
                    id="name-session",
                    title="Test",
                    workspace_root="/workspace",
                    updated_at="now",
                    cost_usd=0.0,
                    tokens=TokenUsage(),
                ),
                list_messages=lambda _session_id, limit=None: [],
            ),
        )

        snapshot = _tui_bridge_snapshot(ctx)

        self.assertEqual(snapshot["agent_name"], "Nymi")

    def test_tui_bridge_snapshot_includes_pending_approvals(self) -> None:
        seen_limits: list[int | None] = []
        ctx = SimpleNamespace(
            session_id="abc123",
            llm=SimpleNamespace(model="gpt-5.4-mini", provider="openai", mode="hosted", configuration_error=None),
            session=SimpleNamespace(pending_approvals=[{
                "id": "req-1",
                "status": "pending",
                "tool": "delete_path",
                "requested_path": "/tmp/example.txt",
                "display_path": "Example file",
            }]),
            store=SimpleNamespace(
                get_session=lambda _session_id: SimpleNamespace(
                    id="abc123",
                    title="Test",
                    workspace_root="/workspace",
                    updated_at="now",
                    cost_usd=0.0123,
                    tokens=TokenUsage(),
                ),
                list_messages=lambda _session_id, limit=None: seen_limits.append(limit) or [],
            ),
        )

        snapshot = _tui_bridge_snapshot(ctx)

        self.assertEqual(snapshot["approvals"][0]["id"], "req-1")
        self.assertEqual(snapshot["approvals"][0]["tool"], "delete_path")
        self.assertEqual(snapshot["approvals"][0]["display_path"], "Example file")
        self.assertEqual(snapshot["session"]["configuration_state"], "ready")
        self.assertNotIn("credential_provider", snapshot["session"])
        self.assertEqual(
            snapshot["session"]["tokens"],
            {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0},
        )
        self.assertEqual(snapshot["session"]["cost_usd"], 0.0123)
        self.assertEqual(seen_limits, [TUI_TRANSCRIPT_LIMIT])

    def test_tui_bridge_snapshot_exposes_stored_attachment_for_ui_opening(self) -> None:
        attachment = SimpleNamespace(
            filename="report.pdf",
            mime="application/pdf",
            size_bytes=123,
            storage_path="/private/attachments/report",
        )
        ctx = SimpleNamespace(
            session_id="attachment-session",
            llm=SimpleNamespace(
                model="gpt-5.4-mini",
                provider="openai",
                mode="hosted",
                configuration_error=None,
            ),
            session=SimpleNamespace(pending_approvals=[]),
            pending_attachments=[attachment],
            store=SimpleNamespace(
                get_session=lambda _session_id: SimpleNamespace(
                    id="attachment-session",
                    title="Test",
                    workspace_root="/workspace",
                    updated_at="now",
                    cost_usd=0.0,
                    tokens=TokenUsage(),
                ),
                list_messages=lambda _session_id, limit=None: [
                    SimpleNamespace(
                        role="user",
                        content="Review this",
                        created_at="now",
                        attachments=[attachment],
                    )
                ],
            ),
        )

        snapshot = _tui_bridge_snapshot(ctx)

        self.assertEqual(
            snapshot["session"]["pending_attachments"][0]["storage_path"],
            attachment.storage_path,
        )
        self.assertEqual(
            snapshot["messages"][0]["attachments"][0]["storage_path"],
            attachment.storage_path,
        )

    def test_tui_bridge_approval_decision_persists_for_waiting_turn(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions.sqlite3")
            session_info = store.create_session(
                workspace_root=Path(tmp),
                provider="openai",
                model="gpt-5.4-mini",
            )
            session = AgentSession(pending_approvals=[{
                "id": "req-volume",
                "status": "pending",
                "tool": "desktop_action",
                "requested_path": "desktop set_volume system 0",
            }])
            store.save_agent_state(session_info.id, {**session.__dict__})
            ctx = SimpleNamespace(
                session_id=session_info.id,
                session=session,
                store=store,
                llm=SimpleNamespace(
                    model="gpt-5.4-mini",
                    provider="openai",
                    mode="hosted",
                    configuration_error=None,
                    reasoning_effort="high",
                ),
            )

            result = _tui_bridge_apply_approval_decision(
                ctx,
                "req-volume",
                "approved",
            )
            persisted = store.get_session(session_info.id).state

        self.assertTrue(result["ok"])
        self.assertEqual(persisted["pending_approvals"][0]["status"], "approved")
        self.assertEqual(persisted["pending_approvals"][0]["decision"], "approved")

    def test_tui_startup_expires_approval_left_without_a_waiting_turn(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions.sqlite3")
            session_info = store.create_session(
                workspace_root=Path(tmp),
                provider="openai",
                model="gpt-5.4-mini",
            )
            session = AgentSession(pending_approvals=[{
                "id": "req-orphaned-volume",
                "status": "pending",
                "tool": "desktop_action",
                "requested_path": "desktop set_volume system 0",
            }])
            store.save_agent_state(session_info.id, {**session.__dict__})
            ctx = SimpleNamespace(
                session_id=session_info.id,
                session=session,
                store=store,
            )

            expired = _expire_orphaned_approvals(ctx)
            persisted = store.get_session(session_info.id).state
            events = store.list_events(session_info.id)

        self.assertEqual(expired, 1)
        self.assertEqual(persisted["pending_approvals"][0]["status"], "expired")
        self.assertEqual(persisted["pending_approvals"][0]["decision"], "denied")
        self.assertEqual(
            persisted["pending_approvals"][0]["expired_reason"],
            "no_active_turn",
        )
        self.assertEqual(events[0].event_type, "approval_expired")

    def test_tui_startup_preserves_approval_while_turn_lock_is_active(self) -> None:
        session = AgentSession(pending_approvals=[{
            "id": "req-active-volume",
            "status": "pending",
            "tool": "desktop_action",
        }])
        ctx = SimpleNamespace(
            session_id="active-approval-session",
            session=session,
            store=SimpleNamespace(),
        )

        with _exclusive_bridge_turn(ctx.session_id):
            expired = _expire_orphaned_approvals(ctx)

        self.assertEqual(expired, 0)
        self.assertEqual(session.pending_approvals[0]["status"], "pending")

    def test_render_tui_transcript_shows_historical_messages(self) -> None:
        messages = [
            SimpleNamespace(role="user", content="hi", created_at="now"),
            SimpleNamespace(role="assistant", content="hello", created_at="now"),
        ]
        live_turn = {"phase": "idle", "active": False, "feed": [], "error": None}

        rendered = _render_tui_transcript(messages, live_turn, 80)

        self.assertIn("You  ", rendered[0])
        self.assertIn("  hi", rendered)
        self.assertTrue(any("Agent" in line for line in rendered))
        self.assertIn("  hello", rendered)

    def test_render_tui_transcript_uses_live_turn_content(self) -> None:
        live_turn = {
            "phase": "streaming",
            "active": True,
            "prompt": "tell me about the repo",
            "feed": [("text", "Working on it")],
            "error": None,
        }

        rendered = _render_tui_transcript([], live_turn, 80)
        text = "\n".join(rendered)

        self.assertIn("You", rendered)
        self.assertIn("tell me about the repo", text)
        self.assertIn("Working on it", text)

    def test_render_tui_transcript_deduplicates_active_user_message(self) -> None:
        messages = [
            SimpleNamespace(role="user", content="tell me about the repo", created_at="now"),
        ]
        live_turn = {
            "phase": "streaming",
            "active": True,
            "prompt": "tell me about the repo",
            "feed": [("text", "Working on it")],
            "error": None,
        }

        rendered = _render_tui_transcript(messages, live_turn, 80)
        text = "\n".join(rendered)

        self.assertEqual(rendered.count("You"), 1)
        self.assertIn("tell me about the repo", text)

    def test_render_tui_transcript_shows_guardrail_activity(self) -> None:
        live_turn = {
            "phase": "observing",
            "active": True,
            "prompt": "delete outside the workspace",
            "feed": [
                ("tool", "Tool: delete_path(...)"),
                ("guardrail", "external_path_requires_approval: ask for approval first"),
            ],
            "error": None,
        }

        rendered = _render_tui_transcript([], live_turn, 80)
        text = "\n".join(rendered)

        self.assertIn("Activity", text)
        self.assertIn("Tool: delete_path", text)
        self.assertIn("Guardrail: external_path_requires_approval", text)

    def test_live_turn_renders_reasoning_state_without_raw_chain_of_thought(self) -> None:
        live_turn = LiveTurnState()
        live_turn.start("inspect the repo")
        live_turn.update({
            "kind": "reasoning_delta",
            "delta": "private detailed chain of thought",
        })

        rendered = _render_tui_transcript([], live_turn.snapshot(), 80)
        text = "\n".join(rendered)

        self.assertIn("Reasoning", text)
        self.assertNotIn("private detailed chain of thought", text)

    def test_live_turn_hides_reasoning_after_answer_text_starts(self) -> None:
        live_turn = LiveTurnState()
        live_turn.start("inspect the repo")
        live_turn.update({
            "kind": "reasoning_delta",
            "delta": "private detailed chain of thought",
        })
        live_turn.update({
            "kind": "text_delta",
            "delta": "Here is the answer",
        })

        rendered = _render_tui_transcript([], live_turn.snapshot(), 80)
        text = "\n".join(rendered)

        self.assertIn("Here is the answer", text)
        self.assertNotIn("private detailed chain of thought", text)

    def test_live_turn_renders_parallel_subagent_lifecycle(self) -> None:
        live_turn = LiveTurnState()
        live_turn.start("inspect independent workstreams")
        live_turn.update({
            "kind": "subagent_run_started",
            "summary": "Spawned 2 parallel subagents · log: .agent/parallel-work.md",
        })
        live_turn.update({
            "kind": "subagent_task_started",
            "task_id": "architecture",
            "summary": "inspect architecture",
        })
        live_turn.update({
            "kind": "subagent_task_completed",
            "task_id": "architecture",
            "status": "complete",
            "summary": "found planner entry point",
        })

        rendered = _render_tui_transcript([], live_turn.snapshot(), 100)
        text = "\n".join(rendered)

        self.assertIn("Subagents", text)
        self.assertIn("Spawned 2 parallel subagents", text)
        self.assertIn("architecture · running", text)
        self.assertIn("architecture · complete", text)

    def test_finished_turn_discards_ephemeral_tool_activity(self) -> None:
        live_turn = LiveTurnState()
        live_turn.start("inspect the repo")
        live_turn.update({"kind": "tool_call_started", "name": "read_path"})
        live_turn.update({"kind": "tool_call_arguments_done"})
        live_turn.update({
            "kind": "tool_result",
            "name": "read_path",
            "summary": "read_path read a private implementation file",
        })

        live_turn.finish()
        snapshot = live_turn.snapshot()

        self.assertFalse(snapshot["active"])
        self.assertEqual(snapshot["phase"], "completed")
        self.assertEqual(snapshot["prompt"], "")
        self.assertEqual(snapshot["feed"], [])

    def test_finished_turn_retains_recent_parallel_summary(self) -> None:
        live_turn = LiveTurnState()
        live_turn.start("inspect two workstreams")
        live_turn.update({
            "kind": "subagent_run_started",
            "summary": "Spawned 2 parallel subagents",
        })
        live_turn.update({
            "kind": "subagent_run_completed",
            "summary": "Parallel subagents finished · 2/2 complete",
        })

        live_turn.finish()
        snapshot = live_turn.snapshot()
        rendered = "\n".join(_render_tui_transcript([], snapshot, 80))

        self.assertFalse(snapshot["active"])
        self.assertIn("Subagents", rendered)
        self.assertIn("2/2 complete", rendered)

    def test_failed_turn_keeps_error_but_discards_tool_trace(self) -> None:
        live_turn = LiveTurnState()
        live_turn.start("inspect the repo")
        live_turn.update({"kind": "tool_call_started", "name": "read_path"})
        live_turn.update({"kind": "tool_call_arguments_done"})

        live_turn.finish("bridge failed")
        rendered = "\n".join(_render_tui_transcript([], live_turn.snapshot(), 80))

        self.assertIn("bridge failed", rendered)
        self.assertNotIn("Reading files", rendered)

    def test_usage_panel_shows_tokens_context_and_cost(self) -> None:
        session = SimpleNamespace(
            tokens=TokenUsage(input=1000, output=500, reasoning=200, cache_read=300),
            cost_usd=0.0123,
        )

        lines = _usage_panel_lines(session, "gpt-4o", "openai", 30)
        text = "\n".join(lines)

        self.assertIn("Usage", text)
        self.assertIn("Source     OpenAI", text)
        self.assertIn("Model      gpt-4o", text)
        self.assertIn("Tokens     1,500", text)
        self.assertIn("Context    1.2%", text)
        self.assertIn("Cost       $0.01", text)
        self.assertIn("Reasoning  200", text)
        self.assertIn("Guardrails", text)

    def test_approval_panel_shows_pending_request(self) -> None:
        approvals = {
            "pending": [
                {
                    "tool": "delete_path",
                    "requested_path": "/tmp/external.txt",
                    "display_path": "External file",
                    "reason": "external_path_requires_approval",
                }
            ],
            "selected_index": 0,
        }

        lines = _approval_panel_lines(approvals, 48)
        text = "\n".join(lines)

        self.assertIn("Approvals", text)
        self.assertIn("delete_path", text)
        self.assertIn("External file", text)
        self.assertNotIn("/tmp/external.txt", text)
        self.assertIn("external_path_requires_approval", text)
        self.assertIn("Enter/Y approve", text)
        self.assertIn("N/Esc deny", text)

    def test_approval_display_hides_encoded_windows_targets(self) -> None:
        self.assertEqual(
            _approval_display_text({
                "requested_path": "desktop launch_application windows-app:abcdef Vitelglobal",
            }),
            "desktop launch_application Vitelglobal",
        )
        self.assertEqual(
            _approval_display_text({
                "requested_path": "desktop close_window 0x800e8",
            }),
            "desktop close_window selected window",
        )

    def test_compact_usage_handles_unknown_model_context(self) -> None:
        session = SimpleNamespace(
            tokens=TokenUsage(input=100, output=25),
            cost_usd=0,
        )

        self.assertEqual(
            _compact_usage_text(session, "custom-model"),
            "openai/custom-model tokens 125 (n/a) cost $0",
        )

    def test_slash_command_lines_show_command_palette(self) -> None:
        lines = _slash_command_lines("/", 80)
        text = "\n".join(lines)

        self.assertIn("Commands", text)
        self.assertIn("> /model", text)
        self.assertIn("/model", text)
        self.assertNotIn("/providers", text)
        self.assertNotIn("/provider", text)
        self.assertIn("/status", text)

    def test_slash_command_lines_filter_by_prefix(self) -> None:
        lines = _slash_command_lines("/mo", 80)
        text = "\n".join(lines)

        self.assertIn("/model", text)
        self.assertNotIn("/providers", text)

    def test_model_command_opens_model_picker(self) -> None:
        entries = _slash_palette_entries("/model")

        self.assertEqual(entries[0].value, "/model")
        self.assertFalse(entries[0].execute)
        self.assertEqual(entries[0].complete_to, "/model ")

    def test_palette_selection_keeps_submenus_but_skips_sections(self) -> None:
        top_entries = _slash_palette_entries("/")

        self.assertEqual(_selectable_palette_index(top_entries, 0), 0)
        self.assertEqual(top_entries[0].value, "/model")

        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-5.5", configuration_error=None))
        with patch("agent.main._discover_local_provider_availability", return_value={}):
            model_entries = _slash_palette_entries("/model ", ctx=ctx)
        section = next(index for index, entry in enumerate(model_entries) if entry.value.startswith("section:"))

        self.assertNotEqual(_selectable_palette_index(model_entries, section), section)

    def test_tab_completion_completes_single_slash_command(self) -> None:
        self.assertEqual(_complete_slash_command("/sta"), "/status ")
        self.assertIsNone(_complete_slash_command("/pro"))

    def test_provider_palette_filters_and_completes_provider(self) -> None:
        entries = _slash_palette_entries("/provider de")

        self.assertEqual(entries[0].value, "deepseek")
        self.assertTrue(entries[0].execute)
        self.assertEqual(_complete_slash_command("/provider de"), "/provider deepseek")

    def test_api_key_palette_filters_and_completes_provider(self) -> None:
        entries = _slash_palette_entries("/apikey an")

        self.assertEqual(entries[0].value, "anthropic")
        self.assertTrue(entries[0].execute)
        self.assertEqual(_complete_slash_command("/apikey an"), "/apikey anthropic")

    def test_model_palette_completes_to_provider_model_pair(self) -> None:
        entries = _slash_palette_entries("/model llama")

        self.assertEqual(entries[0].value, "ollama/llama3.3")
        self.assertIn("open source · local runtime/install · no login", entries[0].description)
        self.assertEqual(_complete_slash_command("/model llama"), "/model ollama llama3.3")

    def test_model_palette_lines_keep_full_long_model_names(self) -> None:
        model = "Qwen/Qwen2.5-Coder-32B-Instruct"
        lines = _slash_command_lines(f"/model {model}", 24)
        text = "\n".join(lines)

        self.assertIn(model, text)
        self.assertNotIn("...", text)

    def test_install_palette_offers_explicit_ollama_download_action(self) -> None:
        entries = _slash_palette_entries("/install ollama")

        self.assertEqual(entries[0].value, "ollama/qwen3")
        self.assertEqual(entries[0].complete_to, "/install ollama qwen3")
        self.assertTrue(entries[0].execute)
        self.assertIn("Open-source/open-weight", entries[0].description)
        self.assertIn("Provider: Ollama", entries[0].description)
        self.assertIn("8B params", entries[0].description)
        self.assertIn("~5.2 GB", entries[0].description)
        self.assertIn("8 GB+ RAM", entries[0].description)
        self.assertIn("preview first", entries[0].description)
        self.assertIn("installs locally", entries[0].description)
        self.assertIn("no login", entries[0].description)

    def test_install_palette_contains_every_supported_local_provider(self) -> None:
        entries = _slash_palette_entries("/install ")
        providers = {entry.value.split("/", 1)[0] for entry in entries}

        self.assertEqual(
            providers,
            {"ollama", "lmstudio", "llamacpp", "vllm", "localai"},
        )
        self.assertEqual(
            {entry.value.split("/", 1)[0] for entry in entries[:5]},
            {"ollama", "lmstudio", "llamacpp", "vllm", "localai"},
        )
        self.assertTrue(all("installs locally" in entry.description for entry in entries))
        self.assertTrue(all("no login" in entry.description for entry in entries))

    def test_reasoning_palette_exposes_effort_without_raw_thinking(self) -> None:
        entries = _slash_palette_entries("/reasoning ")

        self.assertEqual([entry.value for entry in entries], ["minimal", "low", "medium", "high"])
        self.assertEqual(entries[-1].complete_to, "/reasoning high")

    def test_model_palette_defaults_to_hosted_models_first(self) -> None:
        entries = _slash_palette_entries("/model ")

        self.assertEqual(entries[0].value, "openai/gpt-5.5")
        self.assertIn("sign in or API key", entries[0].description)

    def test_model_palette_scrolls_to_selected_entry(self) -> None:
        entries = _slash_palette_entries("/model ")
        selected_index = 12

        lines = _slash_command_lines("/model ", 80, selected_index=selected_index)
        text = "\n".join(lines)

        self.assertIn(f"> {entries[selected_index].label}", text)
        self.assertNotIn(f"> {entries[0].label}", text)

    def test_provider_palette_does_not_override_explicit_model(self) -> None:
        entries = _slash_palette_entries("/provider ollama llama3.1")

        self.assertEqual(entries[0].value, "/model")
        self.assertFalse(entries[0].execute)


class LocalCommandTests(unittest.TestCase):
    def test_install_requires_metadata_preview_before_download(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with patch("agent.main.subprocess.Popen") as popen:
            result = _handle_local_command(ctx, "/install ollama llama3.3")

        popen.assert_not_called()
        self.assertIn("Parameters: 70B", result)
        self.assertIn("Exact artifact: llama3.3:70b", result)
        self.assertIn("Download: ~43 GB", result)
        self.assertIn("Recommended memory: 48 GB+ RAM", result)
        self.assertIn("/install ollama llama3.3 --yes", result)
        self.assertIn("Nothing has been downloaded yet", result)

    def test_reasoning_effort_is_persisted_for_supported_model(self) -> None:
        ctx = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="gpt-5.4-mini",
                reasoning_effort="medium",
            ),
            session=AgentSession(),
            session_id="abc123",
            store=SimpleNamespace(save_agent_state=Mock()),
        )

        result = _handle_local_command(ctx, "/reasoning high")

        self.assertEqual(ctx.llm.reasoning_effort, "high")
        self.assertEqual(ctx.session.reasoning_effort, "high")
        self.assertIn("Reasoning effort set to high", result)
        ctx.store.save_agent_state.assert_called_once()

    def test_status_shows_loaded_ollama_models_for_local_runtime(self) -> None:
        ctx = SimpleNamespace(
            session_id="abc123",
            workspace_root=Path("/workspace"),
            llm=SimpleNamespace(
                provider="ollama",
                model="qwen2.5-coder:7b",
                mode="local",
                endpoint="http://localhost:11434/v1",
                configuration_error=None,
                reasoning_effort=None,
            ),
            session=AgentSession(),
            skills=SimpleNamespace(skills={}),
        )

        with (
            patch("agent.main._discover_provider_models", return_value=(["qwen2.5-coder:7b"], None)),
            patch("agent.main._discover_ollama_loaded_models", return_value=[{
                "model": "qwen2.5-coder:7b",
                "parameters": "7B",
                "quantization": "Q4_K_M",
            }]),
        ):
            result = "\n".join(_local_runtime_status_lines(ctx))

        self.assertIn("Local runtime:", result)
        self.assertIn("- Server: reachable", result)
        self.assertIn("- Installed/API models: qwen2.5-coder:7b", result)
        self.assertIn("- Loaded/warm models: qwen2.5-coder:7b (7B, Q4_K_M)", result)
        self.assertIn("- Active model loaded: yes", result)

    def test_status_explains_cold_ollama_model_latency(self) -> None:
        ctx = SimpleNamespace(
            session_id="abc123",
            workspace_root=Path("/workspace"),
            llm=SimpleNamespace(
                provider="ollama",
                model="qwen2.5-coder:7b",
                mode="local",
                endpoint="http://localhost:11434/v1",
                configuration_error=None,
                reasoning_effort=None,
            ),
            session=AgentSession(),
            skills=SimpleNamespace(skills={}),
        )

        with (
            patch("agent.main._discover_provider_models", return_value=(["qwen2.5-coder:7b"], None)),
            patch("agent.main._discover_ollama_loaded_models", return_value=[]),
        ):
            result = "\n".join(_local_runtime_status_lines(ctx))

        self.assertIn("- Loaded/warm models: none", result)
        self.assertIn("first prompt may be slow", result)

    def test_status_explains_unreachable_local_runtime_latency(self) -> None:
        ctx = SimpleNamespace(
            session_id="abc123",
            workspace_root=Path("/workspace"),
            llm=SimpleNamespace(
                provider="ollama",
                model="qwen2.5-coder:7b",
                mode="local",
                endpoint="http://localhost:11434/v1",
                configuration_error=None,
                reasoning_effort=None,
            ),
            session=AgentSession(),
            skills=SimpleNamespace(skills={}),
        )

        with patch(
            "agent.main._discover_provider_models",
            return_value=([], "connection refused"),
        ):
            result = "\n".join(_local_runtime_status_lines(ctx))

        self.assertIn("- Server: unreachable", result)
        self.assertIn("- Details: connection refused", result)
        self.assertIn("must wait for the runtime/model to start", result)

    def test_model_command_lists_available_model_sources(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o", mode="hosted", endpoint="OpenAI", configuration_error=None))

        result = _handle_local_command(ctx, "/model")

        self.assertIsNotNone(result)
        self.assertIn("Model source: OpenAI", result)
        self.assertIn("Ollama", result)
        self.assertIn("DeepSeek", result)
        self.assertIn("GLM", result)

    def test_models_alias_still_opens_model_picker(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o", mode="hosted", endpoint="OpenAI", configuration_error=None))

        result = _handle_local_command(ctx, "/models")

        self.assertIn("Model source: OpenAI", result)
        self.assertIn("Switch model: /model <model>", result)

    def test_model_command_shows_local_runtime_availability(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o", mode="hosted", endpoint="OpenAI", configuration_error=None))

        with patch("agent.main._discover_provider_models", side_effect=AssertionError("network probe")):
            result = _handle_local_command(ctx, "/model")

        self.assertIn("gpt-5.5", result)
        self.assertIn("qwen3", result)
        self.assertIn("OpenAI", result)
        self.assertIn("Ollama - open source · local runtime/install · no login", result)
        self.assertIn("Server offline", result)

    def test_models_command_includes_models_discovered_from_local_runtime(self) -> None:
        ctx = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="gpt-4o",
                configuration_error=None,
            )
        )
        availability = {
            provider: ([], "offline")
            for provider in {"ollama", "lmstudio", "llamacpp", "vllm", "localai"}
        }
        availability["ollama"] = (["my-local-coder:latest"], None)

        with patch(
            "agent.main._discover_local_provider_availability",
            return_value=availability,
        ):
            result = _handle_local_command(ctx, "/models")

        self.assertIn("my-local-coder:latest  Ready", result)
        self.assertIn("Ollama - open source · local runtime/install · no login", result)

    def test_provider_command_switches_provider_and_model(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with (
            patch("agent.main._resolve_local_model_name", return_value=("llama3.1", None)),
            patch("agent.main.LLMClient", return_value=SimpleNamespace(provider="ollama", model="llama3.1", configuration_error=None)),
        ):
            result = _handle_local_command(ctx, "/model ollama llama3.1")

        self.assertEqual(ctx.llm.provider, "ollama")
        self.assertEqual(ctx.llm.model, "llama3.1")
        self.assertEqual(result, "Ollama · llama3.1")

    def test_provider_command_persists_provider_and_model(self) -> None:
        store = SimpleNamespace(update_llm_config=Mock())
        ctx = SimpleNamespace(
            session_id="abc123",
            store=store,
            llm=SimpleNamespace(provider="openai", model="gpt-4o"),
        )

        with (
            patch("agent.main._resolve_local_model_name", return_value=("llama3.1", None)),
            patch("agent.main.LLMClient", return_value=SimpleNamespace(provider="ollama", model="llama3.1", configuration_error=None)),
        ):
            _handle_local_command(ctx, "/model ollama llama3.1")

        store.update_llm_config.assert_called_once_with(
            "abc123",
            provider="ollama",
            model="llama3.1",
        )

    def test_provider_command_uses_provider_default_model_when_unspecified(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with patch("agent.main.LLMClient", return_value=SimpleNamespace(provider="deepseek", model="deepseek-chat", configuration_error=None)) as client:
            result = _handle_local_command(ctx, "/model deepseek deepseek-chat")

        client.assert_called_once_with(model="deepseek-chat", provider="deepseek")
        self.assertEqual(ctx.llm.provider, "deepseek")
        self.assertEqual(ctx.llm.model, "deepseek-chat")
        self.assertEqual(result, "DeepSeek · deepseek-chat")

    def test_deepseek_hosted_model_prompts_for_api_key(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="ollama", model="llama3.1"))

        with (
            patch(
                "agent.main.LLMClient",
                return_value=SimpleNamespace(
                    provider="deepseek",
                    model="deepseek-v4-flash",
                    configuration_error="DeepSeek is not configured. Set DEEPSEEK_API_KEY.",
                ),
            ),
            patch("agent.main.webbrowser.open", return_value=True) as open_browser,
        ):
            result = _handle_local_command(ctx, "/model deepseek deepseek-v4-flash")

        self.assertIn("Status: API key required", result)
        self.assertIn("/apikey deepseek", result)
        self.assertIn("Opened DeepSeek API-key page", result)
        self.assertEqual(ctx.last_local_command_result["code"], "api_key_required")
        self.assertEqual(ctx.last_local_command_result["secret_provider"], "deepseek")
        open_browser.assert_called_once_with("https://platform.deepseek.com/api_keys", new=2, autoraise=True)

    def test_provider_command_surfaces_configuration_error(self) -> None:
        store = SimpleNamespace(update_llm_config=Mock())
        ctx = SimpleNamespace(
            session_id="session-1",
            store=store,
            llm=SimpleNamespace(provider="ollama", model="llama3.1"),
        )

        with (
            patch(
                "agent.main.LLMClient",
                return_value=SimpleNamespace(
                    provider="openai",
                    model="gpt-4o",
                    configuration_error="OpenAI is not configured. Set OPENAI_API_KEY.",
                ),
            ),
            patch("agent.main.webbrowser.open", return_value=True) as open_browser,
        ):
            result = _handle_local_command(ctx, "/model openai gpt-5.5")

        self.assertIn("Status: API key required", result)
        self.assertIn("/apikey openai", result)
        self.assertIn("Opened OpenAI API-key page", result)
        self.assertEqual(ctx.last_local_command_result["code"], "api_key_required")
        self.assertEqual(ctx.last_local_command_result["secret_provider"], "openai")
        self.assertEqual(ctx.llm.provider, "openai")
        self.assertEqual(ctx.llm.model, "gpt-4o")
        store.update_llm_config.assert_called_once_with(
            "session-1",
            provider="openai",
            model="gpt-4o",
        )
        open_browser.assert_called_once_with("https://platform.openai.com/api-keys", new=2, autoraise=True)

    def test_anthropic_selection_stays_active_while_key_is_missing(self) -> None:
        store = SimpleNamespace(update_llm_config=Mock())
        ctx = SimpleNamespace(
            session_id="session-1",
            store=store,
            llm=SimpleNamespace(provider="openai", model="gpt-4o"),
        )

        candidate = SimpleNamespace(
            provider="anthropic",
            model="claude-sonnet-4.5",
            configuration_error="Anthropic is not configured. Set ANTHROPIC_API_KEY.",
        )
        with (
            patch("agent.main.LLMClient", return_value=candidate),
            patch("agent.main.webbrowser.open", return_value=True),
        ):
            result = _handle_local_command(
                ctx,
                "/model anthropic claude-sonnet-4.5",
            )

        self.assertEqual(ctx.llm, candidate)
        self.assertEqual(ctx.pending_provider, "anthropic")
        self.assertEqual(ctx.pending_model, "claude-sonnet-4.5")
        self.assertIn("Anthropic · claude-sonnet-4.5", result)
        self.assertIn("/apikey anthropic", result)
        store.update_llm_config.assert_called_once_with(
            "session-1",
            provider="anthropic",
            model="claude-sonnet-4.5",
        )

    def test_unimplemented_cloud_provider_does_not_request_credentials(self) -> None:
        store = SimpleNamespace(update_llm_config=Mock())
        ctx = SimpleNamespace(
            session_id="session-1",
            store=store,
            llm=SimpleNamespace(provider="openai", model="gpt-4o"),
        )

        with (
            patch("agent.main.LLMClient") as llm_client,
            patch("agent.main.webbrowser.open") as browser,
        ):
            result = _handle_local_command(
                ctx,
                "/model bedrock amazon.nova-pro-v1:0",
            )

        self.assertEqual(ctx.llm.provider, "openai")
        self.assertIn("Status: unavailable", result)
        self.assertIn("transport is not implemented", result)
        self.assertIn("No credentials were requested or changed", result)
        self.assertNotIn("sign-in", result)
        llm_client.assert_not_called()
        browser.assert_not_called()

    def test_unimplemented_provider_transport_is_unavailable_not_auth_required(self) -> None:
        store = SimpleNamespace(update_llm_config=Mock())
        ctx = SimpleNamespace(
            session_id="session-1",
            store=store,
            llm=SimpleNamespace(provider="openai", model="gpt-4o"),
        )
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("agent.main.LLMClient") as llm_client,
            patch("agent.main.webbrowser.open") as browser,
        ):
            result = _handle_local_command(
                ctx,
                "/model gemini gemini-2.5-pro",
            )

        self.assertIn("Status: unavailable", result)
        self.assertNotIn("API key required", result)
        llm_client.assert_not_called()
        browser.assert_not_called()

    def test_api_key_command_sets_key_and_reloads_active_provider(self) -> None:
        ctx = SimpleNamespace(
            session_id="abc123",
            store=SimpleNamespace(update_llm_config=Mock()),
            llm=SimpleNamespace(provider="anthropic", model="claude-3-5-sonnet-latest"),
        )

        with TemporaryDirectory() as config_home:
            with patch.dict(
                "os.environ",
                {"XDG_CONFIG_HOME": config_home},
                clear=True,
            ):
                result = _handle_local_command(ctx, "/apikey anthropic sk-ant-test")

        self.assertIn("Anthropic key loaded", result)
        self.assertIn("ready", result)
        self.assertEqual(ctx.llm.provider, "anthropic")
        ctx.store.update_llm_config.assert_called_once()

    def test_api_key_command_without_key_prompts_for_hidden_tui_entry(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="anthropic", model="claude-3-5-sonnet-latest"))

        with patch.dict("os.environ", {}, clear=True):
            result = _handle_local_command(ctx, "/apikey anthropic")

        self.assertIn("hidden TUI prompt", result)
        self.assertEqual(_api_key_prompt_provider("/apikey anthropic"), "anthropic")

    def test_api_key_command_uses_process_scoped_key_from_environment(self) -> None:
        ctx = SimpleNamespace(
            session_id="abc123",
            store=SimpleNamespace(update_llm_config=Mock()),
            llm=SimpleNamespace(provider="anthropic", model="claude-sonnet-4.5"),
        )

        with TemporaryDirectory() as config_home:
            with patch.dict(
                "os.environ",
                {
                    "ANTHROPIC_API_KEY": "sk-ant-test",
                    "XDG_CONFIG_HOME": config_home,
                },
                clear=True,
            ):
                result = _handle_local_command(ctx, "/apikey anthropic")

        self.assertIn("Anthropic key loaded", result)
        self.assertEqual(ctx.llm.provider, "anthropic")
        self.assertEqual(ctx.llm.model, "claude-sonnet-4.5")

    def test_api_key_command_is_redacted_for_history(self) -> None:
        self.assertEqual(
            _redact_local_command("/apikey anthropic sk-ant-test"),
            "/apikey anthropic <redacted>",
        )

    def test_login_command_opens_provider_key_page(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="anthropic", model="claude-3-5-sonnet-latest"))

        with patch("agent.main.webbrowser.open", return_value=True) as open_browser:
            result = _handle_local_command(ctx, "/login anthropic")

        self.assertIn("Anthropic account/API-key page", result)
        self.assertIn("/apikey anthropic", result)
        open_browser.assert_called_once()

    def test_missing_hosted_key_triggers_automatic_key_prompt(self) -> None:
        ctx = SimpleNamespace(
            llm=SimpleNamespace(
                provider="anthropic",
                model="claude-3-5-sonnet-latest",
                configuration_error="Anthropic is not configured. Set ANTHROPIC_API_KEY.",
            )
        )

        self.assertEqual(_provider_api_key_needed(ctx), "anthropic")

    def test_missing_compatible_base_url_does_not_trigger_key_prompt(self) -> None:
        ctx = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai-compatible",
                model="local-model",
                configuration_error="OpenAI-compatible provider is not configured. Set AGENT_OPENAI_COMPAT_BASE_URL.",
            )
        )

        self.assertIsNone(_provider_api_key_needed(ctx))

    def test_model_command_switches_model_on_current_provider(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="lmstudio", model="old-model"))

        with (
            patch("agent.main._resolve_local_model_name", return_value=("new-model", None)),
            patch("agent.main.LLMClient", return_value=SimpleNamespace(provider="lmstudio", model="new-model")),
        ):
            result = _handle_local_command(ctx, "/model new-model")

        self.assertEqual(ctx.llm.model, "new-model")
        self.assertEqual(result, "LM Studio · new-model")

    def test_model_command_switches_known_local_model_to_ollama(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="anthropic", model="claude-3-5-sonnet-latest"))

        with (
            patch("agent.main._resolve_local_model_name", return_value=("llama3.1", None)),
            patch("agent.main.LLMClient", return_value=SimpleNamespace(provider="ollama", model="llama3.1", configuration_error=None)) as client,
        ):
            result = _handle_local_command(ctx, "/model llama3.1")

        client.assert_called_once_with(model="llama3.1", provider="ollama")
        self.assertEqual(ctx.llm.provider, "ollama")
        self.assertEqual(result, "Ollama · llama3.1")

    def test_missing_ollama_model_offers_install_action_without_login(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with patch(
            "agent.main._discover_provider_models",
            return_value=(["qwen3:latest"], None),
        ):
            result = _handle_local_command(ctx, "/model ollama llama3.3")

        self.assertIn("Status: model not installed", result)
        self.assertIn("/install ollama llama3.3", result)
        self.assertNotIn("login", result.casefold())

    def test_missing_cataloged_vllm_model_offers_local_install_action(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with patch(
            "agent.main._discover_provider_models",
            return_value=([], None),
        ):
            result = _handle_local_command(
                ctx,
                "/model vllm Qwen/Qwen2.5-Coder-32B-Instruct",
            )

        self.assertIn("Status: model not installed", result)
        self.assertIn(
            "/install vllm Qwen/Qwen2.5-Coder-32B-Instruct",
            result,
        )

    def test_offline_ollama_runtime_is_reported_as_unavailable(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with patch(
            "agent.main._discover_provider_models",
            return_value=([], "connection refused"),
        ):
            result = _handle_local_command(ctx, "/model ollama llama3.3")

        self.assertIn("Status: runtime unavailable", result)
        self.assertIn("https://ollama.com/download", result)
        self.assertIn("/install ollama llama3.3", result)

    def test_offline_cataloged_llamacpp_model_points_to_install_preview(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with patch(
            "agent.main._discover_provider_models",
            return_value=([], "connection refused"),
        ):
            result = _handle_local_command(
                ctx,
                "/model llamacpp qwen2.5-coder-7b-instruct",
            )

        self.assertIn("Status: runtime unavailable", result)
        self.assertIn(
            "Preview/install locally: /install llamacpp qwen2.5-coder-7b-instruct",
            result,
        )
        self.assertNotIn("Automatic installation is not available", result)

    def test_qwen_coder_install_preview_has_concrete_ollama_size(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        result = _handle_local_command(ctx, "/install ollama qwen2.5-coder")

        self.assertIn("Local model install preview", result)
        self.assertIn("Exact artifact: qwen2.5-coder:7b", result)
        self.assertIn("Parameters: 7B", result)
        self.assertIn("Download: ~4.7 GB", result)
        self.assertIn("Confirm download: /install ollama qwen2.5-coder --yes", result)

    def test_unknown_localai_install_does_not_preview_unknown_size_download(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        result = _handle_local_command(ctx, "/install localai qwen2.5-coder")

        self.assertIn("Status: model is not in the install catalog", result)
        self.assertIn("Agent will not preview or start an unknown-size local download.", result)
        self.assertNotIn("Parameters: varies", result)
        self.assertNotIn("Confirm download:", result)

    def test_install_ollama_model_pulls_then_selects_it(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))
        process = SimpleNamespace(
            stdout=io.BytesIO(b"pulling manifest\nsuccess\n"),
            wait=Mock(return_value=0),
        )
        progress: list[str] = []

        with (
            patch("agent.main.shutil.which", return_value="/usr/bin/ollama"),
            patch("agent.main._ensure_ollama_running", return_value=None),
            patch("agent.main.subprocess.Popen", return_value=process) as popen,
            patch("agent.main._verify_local_model_ready", return_value=None),
            patch("agent.main._switch_model", return_value="Ollama · llama3.3") as switch,
        ):
            result = _handle_local_command(
                ctx,
                "/install ollama llama3.3 --yes",
                install_progress=progress.append,
            )

        popen.assert_called_once_with(
            ["ollama", "pull", "llama3.3:70b"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        process.wait.assert_called_once_with()
        switch.assert_called_once_with(ctx, model="llama3.3", provider="ollama")
        self.assertIn("Installed `llama3.3`", result)
        self.assertIn("Ollama · checking local runtime", progress)
        self.assertTrue(any("pulling manifest" in item for item in progress))
        self.assertIn("Ollama · installed llama3.3; selecting model", progress)

    def test_install_does_not_claim_success_when_ollama_pull_cannot_be_verified(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))
        process = SimpleNamespace(stdout=io.BytesIO(b"success\n"), wait=Mock(return_value=0))

        with (
            patch("agent.main.shutil.which", return_value="/usr/bin/ollama"),
            patch("agent.main._ensure_ollama_running", return_value=None),
            patch("agent.main.subprocess.Popen", return_value=process),
            patch(
                "agent.main._verify_local_model_ready",
                return_value="Local API models: none.",
            ),
            patch("agent.main._switch_model") as switch,
        ):
            result = _handle_local_command(ctx, "/install ollama llama3.3 --yes")

        self.assertIn("Status: install could not be verified", result)
        self.assertNotIn("Installed `llama3.3`", result)
        switch.assert_not_called()

    def test_install_local_model_without_runtime_shows_provider_install_path(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with patch("agent.main.shutil.which", return_value=None):
            result = _handle_local_command(ctx, "/install llamacpp gemma-3-1b-it --yes")

        self.assertIn("Status: runtime not installed", result)
        self.assertIn("llama.cpp is not installed", result)
        self.assertIn("ggml-org/llama.cpp/releases", result)
        self.assertNotIn("login", result.casefold())

    def test_non_ollama_install_uses_selected_provider_backend(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))
        cases = (
            (
                "lmstudio",
                "gpt-oss-20b",
                ["lms", "get", "openai/gpt-oss-20b"],
            ),
            (
                "llamacpp",
                "gemma-3-1b-it",
                [
                    "/usr/bin/llama-cli",
                    "-hf",
                    "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M",
                    "-p",
                    "",
                    "-n",
                    "1",
                ],
            ),
            (
                "vllm",
                "qwen2.5-1.5b-instruct",
                ["/usr/bin/hf", "download", "Qwen/Qwen2.5-1.5B-Instruct"],
            ),
            (
                "localai",
                "llama-3.2-1b-instruct",
                [
                    "local-ai",
                    "models",
                    "install",
                    "llama-3.2-1b-instruct:q4_k_m",
                ],
            ),
        )

        for provider, model, expected_command in cases:
            with self.subTest(provider=provider):
                def which(name: str) -> str | None:
                    paths = {
                        "lms": "/usr/bin/lms",
                        "llama-cli": "/usr/bin/llama-cli",
                        "vllm": "/usr/bin/vllm",
                        "hf": "/usr/bin/hf",
                        "local-ai": "/usr/bin/local-ai",
                    }
                    return paths.get(name)

                with (
                    patch("agent.main.shutil.which", side_effect=which),
                    patch(
                        "agent.main._run_local_install_command",
                        return_value=(True, "complete"),
                    ) as run_install,
                    patch("agent.main._activate_local_runtime", return_value=None),
                    patch("agent.main._verify_local_model_ready", return_value=None),
                    patch(
                        "agent.main._switch_model",
                        return_value=f"{provider} · {model}",
                    ),
                ):
                    result = _handle_local_command(
                        ctx,
                        f"/install {provider} {model} --yes",
                    )

                self.assertEqual(run_install.call_args.args[0], expected_command)
                self.assertIn("Installed", result)

    def test_install_starts_local_ollama_runtime_when_it_is_offline(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))
        progress: list[str] = []

        with (
            patch(
                "agent.main._discover_provider_models",
                side_effect=[
                    ([], "connection refused"),
                    ([], "connection refused"),
                    (["qwen3:latest"], None),
                ],
            ),
            patch("agent.main._provider_base_url", return_value="http://localhost:11434"),
            patch("agent.main.time.sleep"),
            patch("agent.main.subprocess.Popen") as popen,
        ):
            result = _ensure_ollama_running(ctx, progress=progress.append)

        self.assertIsNone(result)
        popen.assert_called_once_with(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.assertIn("Ollama · starting local runtime", progress)
        self.assertIn("Ollama · local runtime started", progress)

    def test_vllm_install_activation_starts_selected_model_locally(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with (
            patch("agent.main._provider_base_url", return_value="http://localhost:8000/v1"),
            patch("agent.main._discover_provider_models", return_value=(["qwen-local"], None)),
            patch("agent.main.subprocess.Popen") as popen,
        ):
            result = _activate_local_runtime(
                ctx,
                provider="vllm",
                model="qwen-local",
                install_id="Qwen/Qwen2.5-1.5B-Instruct",
                progress=None,
            )

        self.assertIsNone(result)
        popen.assert_called_once_with(
            [
                "vllm",
                "serve",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--served-model-name",
                "qwen-local",
                "--port",
                "8000",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def test_local_runtime_activation_waits_for_selected_model(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with (
            patch("agent.main._provider_base_url", return_value="http://localhost:8000/v1"),
            patch(
                "agent.main._discover_provider_models",
                side_effect=[(["other-model"], None), (["qwen-local"], None)],
            ) as discover,
            patch("agent.main.subprocess.Popen"),
            patch("agent.main.time.sleep"),
        ):
            result = _activate_local_runtime(
                ctx,
                provider="vllm",
                model="qwen-local",
                install_id="Qwen/Qwen2.5-1.5B-Instruct",
                progress=None,
            )

        self.assertIsNone(result)
        self.assertEqual(discover.call_count, 2)

    def test_model_command_lists_provider_model_hints(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="deepseek", model="deepseek-chat"))

        result = _handle_local_command(ctx, "/model")

        self.assertIn("Model source: DeepSeek", result)
        self.assertIn("deepseek-v4-flash", result)
        self.assertIn("DeepSeek - sign in or API key", result)
        self.assertIn("qwen3", result)
        self.assertIn("open source · local runtime/install · no login", result)
        self.assertIn("/model <source> <model>", result)
        self.assertIn("installed on this computer", result)
        self.assertIn("/install <provider> <model>", result)
        self.assertIn("Ollama, LM Studio, llama.cpp, vLLM, and LocalAI", result)

    def test_status_command_shows_session_and_context_only(self) -> None:
        ctx = SimpleNamespace(
            session_id="abc123",
            workspace_root="/workspace",
            llm=SimpleNamespace(
                provider="openai",
                model="gpt-4o",
                mode="hosted",
                endpoint="OpenAI",
                configuration_error="OPENAI_API_KEY is missing",
            ),
            session=SimpleNamespace(pending_approvals=[{"status": "pending"}]),
        )

        result = _handle_local_command(ctx, "/status")

        self.assertIn("Status", result)
        self.assertIn("Session:\nabc123", result)
        self.assertIn("Context:", result)
        self.assertIn("left (", result)
        self.assertIn("█", result)
        self.assertNotIn("Configuration:", result)
        self.assertNotIn("Pending approvals:", result)

    def test_connect_command_shows_a_short_guided_setup(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        result = _handle_local_command(ctx, "/connect")

        self.assertIn("Connect a model", result)
        self.assertIn("Ollama", result)
        self.assertNotIn("OPENAI_API_KEY", result)
        self.assertNotIn("AGENT_LLAMACPP_BASE_URL", result)

    def test_setup_palette_uses_guided_provider_choices(self) -> None:
        entries = _slash_palette_entries("/setup ")

        self.assertTrue(any(entry.value == "ollama" for entry in entries))
        self.assertTrue(any(entry.value == "status" for entry in entries))
        self.assertFalse(any(entry.value == "/tools" for entry in entries))

    def test_local_open_source_provider_never_opens_login(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="llamacpp", model="local-model"))

        with patch("agent.main.webbrowser.open") as open_browser:
            result = _handle_local_command(ctx, "/login llamacpp")

        open_browser.assert_not_called()
        self.assertIn("needs no login", result)

    def test_slash_exit_aliases_are_exit_commands(self) -> None:
        self.assertTrue(_is_exit_command("/exit"))
        self.assertTrue(_is_exit_command("/q"))
        self.assertFalse(_is_exit_command("quit"))
        self.assertFalse(_is_exit_command("exit"))
