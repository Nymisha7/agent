from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nym_agent.main import (
    LiveTurnState,
    _approval_panel_lines,
    _compact_usage_text,
    _complete_slash_command,
    _handle_local_command,
    _is_exit_command,
    _queue_status,
    _render_tui_transcript,
    _slash_command_lines,
    _slash_palette_entries,
    _usage_panel_lines,
    run_tui,
)
from nym_agent.session_store import TokenUsage


class TuiExitTests(unittest.TestCase):
    def test_run_tui_handles_ctrl_c_without_error(self) -> None:
        with (
            patch("nym_agent.main.sys.stdin.isatty", return_value=True),
            patch("nym_agent.main.sys.stdout.isatty", return_value=True),
            patch("nym_agent.main.curses.wrapper", side_effect=KeyboardInterrupt),
        ):
            self.assertEqual(run_tui(SimpleNamespace()), 0)

    def test_queue_status_shows_pending_prompt_count(self) -> None:
        self.assertEqual(_queue_status("Thinking...", 0), "Thinking...")
        self.assertEqual(_queue_status("Thinking...", 2), "Thinking... | queued 2")


class TuiRenderingTests(unittest.TestCase):
    def test_render_tui_transcript_shows_historical_messages(self) -> None:
        messages = [
            SimpleNamespace(role="user", content="hi", created_at="now"),
            SimpleNamespace(role="assistant", content="hello", created_at="now"),
        ]
        live_turn = {"phase": "idle", "active": False, "feed": [], "error": None}

        rendered = _render_tui_transcript(messages, live_turn, 80)

        self.assertIn("You  ", rendered[0])
        self.assertIn("  hi", rendered)
        self.assertTrue(any("Nym" in line for line in rendered))
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

    def test_live_turn_does_not_render_raw_reasoning_delta(self) -> None:
        live_turn = LiveTurnState()
        live_turn.start("inspect the repo")
        live_turn.update({
            "kind": "reasoning_delta",
            "delta": "private detailed chain of thought",
        })

        rendered = _render_tui_transcript([], live_turn.snapshot(), 80)
        text = "\n".join(rendered)

        self.assertIn("Thinking through the next step", text)
        self.assertNotIn("private detailed chain of thought", text)

    def test_usage_panel_shows_tokens_context_and_cost(self) -> None:
        session = SimpleNamespace(
            tokens=TokenUsage(input=1000, output=500, reasoning=200, cache_read=300),
            cost_usd=0.0123,
        )

        lines = _usage_panel_lines(session, "gpt-4o", "openai", 30)
        text = "\n".join(lines)

        self.assertIn("Usage", text)
        self.assertIn("Provider   openai", text)
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
                    "reason": "external_path_requires_approval",
                }
            ],
            "selected_index": 0,
        }

        lines = _approval_panel_lines(approvals, 48)
        text = "\n".join(lines)

        self.assertIn("Approvals", text)
        self.assertIn("delete_path", text)
        self.assertIn("/tmp/external.txt", text)
        self.assertIn("external_path_requires_approval", text)

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
        self.assertIn("/providers", text)
        self.assertIn("/provider", text)
        self.assertIn("/status", text)

    def test_slash_command_lines_filter_by_prefix(self) -> None:
        lines = _slash_command_lines("/mo", 80)
        text = "\n".join(lines)

        self.assertIn("/model", text)
        self.assertIn("/models", text)
        self.assertNotIn("/providers", text)

    def test_tab_completion_completes_single_slash_command(self) -> None:
        self.assertEqual(_complete_slash_command("/sta"), "/status ")
        self.assertIsNone(_complete_slash_command("/pro"))

    def test_provider_palette_filters_and_completes_provider(self) -> None:
        entries = _slash_palette_entries("/provider de")

        self.assertEqual(entries[0].value, "deepseek")
        self.assertTrue(entries[0].execute)
        self.assertEqual(_complete_slash_command("/provider de"), "/provider deepseek")

    def test_provider_palette_does_not_override_explicit_model(self) -> None:
        entries = _slash_palette_entries("/provider ollama llama3.1")

        self.assertEqual(entries[0].value, "/provider")
        self.assertFalse(entries[0].execute)


class LocalCommandTests(unittest.TestCase):
    def test_provider_command_lists_available_providers(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o", mode="hosted", endpoint="OpenAI", configuration_error=None))

        result = _handle_local_command(ctx, "/providers")

        self.assertIsNotNone(result)
        self.assertIn("Active provider: openai", result)
        self.assertIn("Configuration: ready", result)
        self.assertIn("ollama", result)
        self.assertIn("deepseek", result)
        self.assertIn("glm", result)

    def test_provider_command_switches_provider_and_model(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with patch("nym_agent.main.LLMClient", return_value=SimpleNamespace(provider="ollama", model="llama3.1", configuration_error=None)):
            result = _handle_local_command(ctx, "/provider ollama llama3.1")

        self.assertEqual(ctx.llm.provider, "ollama")
        self.assertEqual(ctx.llm.model, "llama3.1")
        self.assertIn("Provider switched to ollama", result)
        self.assertIn("Configuration: ready", result)

    def test_provider_command_uses_provider_default_model_when_unspecified(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        with patch("nym_agent.main.LLMClient", return_value=SimpleNamespace(provider="deepseek", model="deepseek-chat", configuration_error=None)) as client:
            result = _handle_local_command(ctx, "/provider deepseek")

        client.assert_called_once_with(model=None, provider="deepseek")
        self.assertEqual(ctx.llm.provider, "deepseek")
        self.assertEqual(ctx.llm.model, "deepseek-chat")
        self.assertIn("Provider switched to deepseek", result)

    def test_provider_command_surfaces_configuration_error(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="ollama", model="llama3.1"))

        with patch(
            "nym_agent.main.LLMClient",
            return_value=SimpleNamespace(
                provider="openai",
                model="gpt-4o",
                configuration_error="OpenAI is not configured. Set OPENAI_API_KEY.",
            ),
        ):
            result = _handle_local_command(ctx, "/provider openai")

        self.assertIn("Provider switched to openai", result)
        self.assertIn("OPENAI_API_KEY", result)

    def test_model_command_switches_model_on_current_provider(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="lmstudio", model="old-model"))

        with patch("nym_agent.main.LLMClient", return_value=SimpleNamespace(provider="lmstudio", model="new-model")):
            result = _handle_local_command(ctx, "/model new-model")

        self.assertEqual(ctx.llm.model, "new-model")
        self.assertIn("Model switched to new-model", result)

    def test_models_command_lists_provider_model_hints(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="deepseek", model="deepseek-chat"))

        result = _handle_local_command(ctx, "/models")

        self.assertIn("Active provider: deepseek", result)
        self.assertIn("deepseek-chat", result)
        self.assertIn("/provider <provider> <model>", result)

    def test_status_command_shows_configuration_and_approvals(self) -> None:
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

        self.assertIn("Session: abc123", result)
        self.assertIn("Configuration: OPENAI_API_KEY is missing", result)
        self.assertIn("Pending approvals: 1", result)

    def test_connect_command_shows_setup_paths(self) -> None:
        ctx = SimpleNamespace(llm=SimpleNamespace(provider="openai", model="gpt-4o"))

        result = _handle_local_command(ctx, "/connect")

        self.assertIn("OPENAI_API_KEY", result)
        self.assertIn("Ollama local", result)

    def test_slash_exit_aliases_are_exit_commands(self) -> None:
        self.assertTrue(_is_exit_command("/exit"))
        self.assertTrue(_is_exit_command("/q"))
        self.assertTrue(_is_exit_command("quit"))
