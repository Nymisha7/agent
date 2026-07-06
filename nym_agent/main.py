from __future__ import annotations

import argparse
import curses
import os
import sys
import uuid
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .context_builder import build_stored_context
from .language_servers import LanguageServerManager
from .llm import LLMClient, SUPPORTED_PROVIDERS
from .planner import (
    AgentSession,
    agent_session_from_dict,
    agent_session_to_dict,
    run_agent,
)
from .rust_tools import RustTools
from .session_store import SessionInfo, SessionStore, TokenUsage


SESSION_LIST_LIMIT = 10
USAGE_PANEL_WIDTH = 30
USAGE_PANEL_MIN_TERMINAL_WIDTH = 105
DEFAULT_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
}
LOCAL_COMMANDS = (
    ("/help", "Show local command help"),
    ("/status", "Show session, provider, and configuration status"),
    ("/providers", "Show active provider and available providers"),
    ("/provider", "Switch provider: /provider <provider> [model]"),
    ("/models", "Show model suggestions for each provider"),
    ("/model", "Switch model on the active provider: /model <model>"),
    ("/connect", "Show setup instructions for hosted and local providers"),
    ("/exit", "Exit the TUI"),
)
PROVIDER_MODEL_HINTS = {
    "openai": ("gpt-4o", "gpt-4o-mini", "gpt-4.1"),
    "anthropic": ("claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"),
    "ollama": ("llama3.1", "qwen2.5-coder", "deepseek-r1"),
    "lmstudio": ("local-model",),
    "openai-compatible": ("local-model",),
    "deepseek": ("deepseek-chat", "deepseek-reasoner"),
    "glm": ("glm-4", "glm-4.5"),
}
COLOR_HEADER = 1
COLOR_USER = 2
COLOR_ASSISTANT = 3
COLOR_THINKING = 4
COLOR_GUARDRAIL = 5
COLOR_MUTED = 6
COLOR_ERROR = 7


@dataclass
class AppContext:
    workspace_root: Path
    search_roots: list[Path]
    rust: RustTools
    llm: LLMClient
    language_servers: LanguageServerManager
    session: AgentSession
    session_id: str
    store: SessionStore
    stored_context: str | None = None
    debug: bool = False


@dataclass(frozen=True)
class PaletteEntry:
    value: str
    label: str
    description: str
    complete_to: str
    execute: bool = False


@dataclass
class ApprovalQueueState:
    ctx: AppContext
    lock: threading.Condition = field(default_factory=threading.Condition, init=False, repr=False)
    selected_index: int = 0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            pending = self.pending_items()
            selected = min(self.selected_index, max(0, len(pending) - 1))
            return {
                "pending": pending,
                "selected_index": selected,
            }

    def pending_items(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.ctx.session.pending_approvals
            if isinstance(item, dict) and item.get("status") == "pending"
        ]

    def request(self, request: dict[str, Any]) -> str:
        normalized = self._normalize_request(request)
        with self.lock:
            self._upsert_request(normalized)
            self.selected_index = len(self.pending_items()) - 1
            persist_agent_state(self.ctx)
            while self._request_status(normalized["id"]) == "pending":
                self.lock.wait()
            return self._request_decision(normalized["id"]) or "denied"

    def approve_selected(self) -> bool:
        return self._decide_selected("approved")

    def deny_selected(self) -> bool:
        return self._decide_selected("denied")

    def next_item(self) -> None:
        with self.lock:
            pending = self.pending_items()
            if pending:
                self.selected_index = min(len(pending) - 1, self.selected_index + 1)

    def previous_item(self) -> None:
        with self.lock:
            pending = self.pending_items()
            if pending:
                self.selected_index = max(0, self.selected_index - 1)

    def _decide_selected(self, decision: str) -> bool:
        with self.lock:
            pending = self.pending_items()
            if not pending:
                return False
            self.selected_index = min(self.selected_index, len(pending) - 1)
            request = pending[self.selected_index]
            request["status"] = "approved" if decision == "approved" else "denied"
            request["decision"] = decision
            request["decision_at"] = datetime.now(timezone.utc).isoformat()
            persist_agent_state(self.ctx)
            self.lock.notify_all()
            return True

    def _upsert_request(self, request: dict[str, Any]) -> None:
        request_id = str(request.get("id") or uuid.uuid4().hex)
        request["id"] = request_id
        request["status"] = "pending"
        request.setdefault("decision", None)
        request.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        approvals = [item for item in self.ctx.session.pending_approvals if _approval_item_id(item) != request_id]
        approvals.append(request)
        self.ctx.session.pending_approvals = approvals

    def _request_status(self, request_id: str) -> str:
        request = self._request_by_id(request_id)
        if request is None:
            return "denied"
        return str(request.get("status") or "denied")

    def _request_decision(self, request_id: str) -> str | None:
        request = self._request_by_id(request_id)
        if request is None:
            return None
        decision = request.get("decision")
        return str(decision) if isinstance(decision, str) and decision else None

    def _request_by_id(self, request_id: str) -> dict[str, Any] | None:
        for item in self.ctx.session.pending_approvals:
            if _approval_item_id(item) == request_id:
                return item
        return None

    def _normalize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(request)
        normalized.setdefault("id", uuid.uuid4().hex)
        normalized.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        normalized.setdefault("status", "pending")
        return normalized


def _approval_item_id(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("id")
        if isinstance(value, str):
            return value
    return ""


@dataclass
class LiveTurnState:
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    active: bool = False
    phase: str = "idle"
    prompt: str = ""
    feed: list[tuple[str, str]] = field(default_factory=list)
    _reasoning_buf: str = field(default="", init=False, repr=False)
    _text_buf: str = field(default="", init=False, repr=False)
    _current_tool: str | None = field(default=None, init=False, repr=False)
    error: str | None = None

    def start(self, prompt: str) -> None:
        with self.lock:
            self.active = True
            self.phase = "thinking"
            self.prompt = prompt
            self.feed = []
            self._reasoning_buf = ""
            self._text_buf = ""
            self._current_tool = None
            self.error = None

    def _flush_reasoning(self) -> None:
        text = self._reasoning_buf.strip()
        if text:
            if not self.feed or self.feed[-1] != ("thinking", "Thinking through the next step"):
                self.feed.append(("thinking", "Thinking through the next step"))
        self._reasoning_buf = ""

    def _flush_text(self) -> None:
        text = self._text_buf.strip()
        if text:
            self.feed.append(("text", text))
        self._text_buf = ""

    def update(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        delta = event.get("delta")
        with self.lock:
            if kind == "reasoning_delta" and isinstance(delta, str):
                self._flush_text()
                self._reasoning_buf += delta
                self.phase = "reasoning"
            elif kind == "text_delta" and isinstance(delta, str):
                self._flush_reasoning()
                self._text_buf += delta
                self.phase = "responding"
            elif kind == "tool_call_started":
                self._flush_reasoning()
                self._flush_text()
                name = event.get("name", "")
                self._current_tool = name if isinstance(name, str) else ""
                self.phase = "tool_call"
            elif kind == "tool_call_arguments_done":
                args = event.get("arguments") or ""
                label = f"Tool: {self._current_tool}({truncate(args, 72)})"
                self.feed.append(("tool", label))
                self._current_tool = None
                self.phase = "tool_call"
            elif kind == "tool_result":
                self._flush_reasoning()
                self._flush_text()
                self.feed.append(_live_tool_result_feed_item(event))
                self.phase = "observing"
            elif kind == "approval_request":
                self._flush_reasoning()
                self._flush_text()
                summary = event.get("summary")
                if isinstance(summary, str) and summary:
                    self.feed.append(("guardrail", summary))
                self.phase = "observing"
            elif kind == "approval_decision":
                self._flush_reasoning()
                self._flush_text()
                summary = event.get("summary")
                if isinstance(summary, str) and summary:
                    self.feed.append(("guardrail", summary))
                self.phase = "observing"
            elif kind == "response_completed":
                self._flush_reasoning()
                self._flush_text()
                self.active = False
                self.phase = "completed"

    def finish(self, error: str | None = None) -> None:
        with self.lock:
            self._flush_reasoning()
            self._flush_text()
            self.active = False
            self.phase = "error" if error else "completed"
            self.error = error

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            feed_snapshot = list(self.feed)
            # append live buffers as streaming (not yet flushed) entries
            if self._reasoning_buf:
                feed_snapshot.append(("thinking", "Thinking through the next step"))
            if self._text_buf:
                feed_snapshot.append(("text", self._text_buf))
            if self._current_tool:
                feed_snapshot.append(("tool", f"Tool: {self._current_tool}(...)"))
            return {
                "active": self.active,
                "phase": self.phase,
                "prompt": self.prompt,
                "feed": feed_snapshot,
                "error": self.error,
            }


def _live_tool_result_feed_item(event: dict[str, Any]) -> tuple[str, str]:
    name = event.get("name")
    summary = event.get("summary")
    observation = event.get("observation")
    label = str(name) if isinstance(name, str) and name else "tool"

    if isinstance(observation, dict):
        if observation.get("blocked"):
            reason = observation.get("reason")
            guidance = observation.get("guidance") or observation.get("error")
            detail = guidance if isinstance(guidance, str) and guidance else summary
            if isinstance(reason, str) and reason:
                return ("guardrail", f"{reason}: {detail or label}")
            return ("guardrail", str(detail or f"{label} blocked"))
        if observation.get("ok") is False:
            error = observation.get("error")
            return ("tool_error", str(error or summary or f"{label} failed"))

    return ("tool_result", str(summary or f"{label} completed"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nym",
        description="Nym CLI coding agent",
    )

    parser.add_argument(
        "--root",
        default=None,
        help="Workspace root. Defaults to current directory.",
    )

    parser.add_argument(
        "--rust-bin",
        default=None,
        help="Path to nym-rust binary. Defaults to the built binary in the repository.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model to use for the selected provider. Defaults to the provider's configured model.",
    )

    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "openai-compatible", "ollama", "lmstudio", "anthropic", "deepseek", "glm"],
        help=(
            "LLM provider. Defaults to NYM_LLM_PROVIDER or openai. "
            "Use ollama/lmstudio/deepseek/glm for local no-login providers when no hosted API key is set."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information.",
    )

    parser.add_argument(
        "--tui",
        action="store_true",
        help="Start the terminal UI instead of the line prompt.",
    )

    parser.add_argument(
        "session_command",
        nargs="?",
        help="Use 'resume' or pass a session id/prefix to resume.",
    )

    parser.add_argument(
        "session_id",
        nargs="?",
        help="Session id/prefix when using 'resume'.",
    )

    return parser


def resolve_rust_bin(
    args: argparse.Namespace,
    workspace_root: Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    if getattr(args, "rust_bin", None):
        rust_bin = Path(args.rust_bin).expanduser()
        if not rust_bin.is_absolute():
            rust_bin = workspace_root / rust_bin
        return rust_bin.resolve()

    repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    candidate_roots = [
        repo_root,
        workspace_root.resolve(),
        workspace_root.resolve().parent,
    ]

    for candidate_root in candidate_roots:
        for candidate in (
            candidate_root / "nym-rust" / "target" / "debug" / "nym-rust",
            candidate_root / "nym-rust" / "target" / "release" / "nym-rust",
            candidate_root / "target" / "debug" / "nym-rust",
            candidate_root / "target" / "release" / "nym-rust",
        ):
            if candidate.exists():
                return candidate.resolve()

    return (repo_root / "nym-rust" / "target" / "debug" / "nym-rust").resolve()


def build_context(
    args: argparse.Namespace,
    *,
    store: SessionStore,
    session_info: SessionInfo,
) -> AppContext:
    workspace_root = Path(session_info.workspace_root).expanduser().resolve()
    rust_bin = resolve_rust_bin(
        args,
        workspace_root,
        repo_root=Path(__file__).resolve().parents[1],
    )
    session = load_agent_session(session_info)
    stored_context = build_stored_context(
        store=store,
        session=session_info,
    ).text

    return AppContext(
        workspace_root=workspace_root,
        search_roots=parse_search_roots(workspace_root),
        rust=RustTools(rust_bin=rust_bin),
        llm=LLMClient(model=args.model, provider=args.provider),
        language_servers=LanguageServerManager(),
        session=session,
        session_id=session_info.id,
        store=store,
        stored_context=stored_context,
        debug=args.debug,
    )


def default_workspace_root(args: argparse.Namespace) -> Path:
    if args.root:
        return Path(args.root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if cwd == Path("/"):
        return Path.home().resolve()

    if cwd.name == "nym":
        repo_candidates = [cwd.parent, cwd]
    elif cwd.name == "nym_agent":
        repo_candidates = [cwd.parent.parent, cwd.parent, cwd]
    else:
        repo_candidates = [cwd]

    for candidate in repo_candidates:
        if (candidate / "nym-rust").exists() and (candidate / "README.md").exists():
            return candidate.resolve()

    return cwd


def parse_search_roots(_workspace_root: Path) -> list[Path]:
    return []


def load_session_messages(store: SessionStore, session_id: str) -> list[dict[str, str]]:
    messages = store.list_messages(session_id, limit=None)
    return [
        {"role": message.role, "content": message.content}
        for message in messages
        if message.role in {"user", "assistant"}
    ]


def handle_prompt(
    ctx: AppContext,
    prompt: str,
    *,
    stream_event: Callable[[dict[str, Any]], None] | None = None,
    approval_requester: Callable[[dict[str, Any]], str] | None = None,
) -> str:
    ctx.store.update_last_prompt(ctx.session_id, prompt)
    ctx.store.add_message(ctx.session_id, "user", prompt)
    ctx.store.add_event(
        ctx.session_id,
        event_type="turn_started",
        summary=f"User prompt: {truncate(prompt, 260)}",
        data={"prompt": prompt},
    )
    ctx.store.add_event(
        ctx.session_id,
        event_type="assistant_stream_started",
        summary="Assistant stream started",
        data={"prompt": prompt},
    )

    conversation_history = load_session_messages(ctx.store, ctx.session_id)

    ctx.llm.reset_turn_usage()
    try:
        answer = run_agent(
            llm=ctx.llm,
            rust=ctx.rust,
            workspace_root=str(ctx.workspace_root),
            search_roots=[str(root) for root in ctx.search_roots],
            user_prompt=prompt,
            session=ctx.session,
            stored_context=ctx.stored_context,
            conversation_history=conversation_history,
            record_event=lambda **kwargs: ctx.store.add_event(ctx.session_id, **kwargs),
            stream_event=stream_event,
            approval_requester=approval_requester,
            language_servers=ctx.language_servers,
            debug=ctx.debug,
        )
    finally:
        usage = ctx.llm.consume_turn_usage()
        ctx.store.add_usage(
            ctx.session_id,
            tokens=TokenUsage(
                input=usage.get("input", 0),
                output=usage.get("output", 0),
                reasoning=usage.get("reasoning", 0),
                cache_read=usage.get("cache_read", 0),
                cache_write=usage.get("cache_write", 0),
            ),
            cost_usd=ctx.llm.estimate_cost_usd(usage),
        )

    ctx.store.add_event(
        ctx.session_id,
        event_type="assistant_answer",
        summary=f"Assistant answer: {truncate(answer, 260)}",
        data={"answer": answer},
    )
    ctx.store.add_event(
        ctx.session_id,
        event_type="assistant_stream_completed",
        summary="Assistant stream completed",
        data={"answer": answer},
    )
    ctx.store.add_message(ctx.session_id, "assistant", answer)
    persist_agent_state(ctx)
    ctx.stored_context = None
    return answer


def repl(ctx: AppContext) -> int:
    print("Nym agent started.")
    print(f"Session: {ctx.session_id}")
    print("Type 'exit' to quit.")
    print()

    try:
        while True:
            try:
                user_input = input("nym> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not user_input:
                continue

            if _is_exit_command(user_input):
                return 0

            try:
                answer = _handle_local_command(ctx, user_input)
                if answer is None:
                    answer = handle_prompt(ctx, user_input)
                print(answer)
            except Exception as exc:
                if ctx.debug:
                    raise
                print(f"Error: {exc}")
    finally:
        _stop_language_servers(ctx)

    return 0


def create_new_session(args: argparse.Namespace, store: SessionStore) -> SessionInfo:
    workspace_root = default_workspace_root(args)
    return store.create_session(
        workspace_root=workspace_root,
        model=args.model,
    )


def choose_session(store: SessionStore) -> SessionInfo | None:
    sessions = store.list_sessions(limit=SESSION_LIST_LIMIT)
    if not sessions:
        print("No sessions found.")
        return None

    print("Resume a previous session")
    print()

    for index, item in enumerate(sessions, start=1):
        title = item.last_prompt or item.title
        print(f"{index:>2}. {item.id}  {format_age(item.updated_at):>8}  {truncate(title, 96)}")

    print()
    choice = input("Select session number/id, or press Enter to cancel: ").strip()
    if not choice:
        return None

    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(sessions):
            return sessions[index - 1]
        raise ValueError(f"Invalid session number: {choice}")

    session_id = store.resolve_session_id(choice)
    return store.get_session(session_id)


def load_existing_session(command: str, store: SessionStore) -> SessionInfo:
    session_id = store.resolve_session_id(command)
    return store.get_session(session_id)


def load_agent_session(session_info: SessionInfo) -> AgentSession:
    return agent_session_from_dict(session_info.state)


def persist_agent_state(ctx: AppContext) -> None:
    ctx.store.save_agent_state(
        ctx.session_id,
        agent_session_to_dict(ctx.session),
    )


def _handle_local_command(ctx: AppContext, user_input: str) -> str | None:
    text = user_input.strip()
    if not text.startswith("/"):
        return None
    if text == "/":
        return _slash_help_text()
    parts = text.split()
    command = parts[0].casefold()

    if command in {"/providers", "/provider"} and len(parts) == 1:
        return _providers_text(ctx)

    if command == "/provider":
        provider = parts[1] if len(parts) >= 2 else ""
        model = parts[2] if len(parts) >= 3 else None
        if not provider:
            return "Usage: /provider <provider> [model]"
        try:
            ctx.llm = LLMClient(model=model, provider=provider)
        except Exception as exc:
            return f"Could not switch provider: {exc}"
        return (
            f"Provider switched to {_active_provider(ctx)} with model {ctx.llm.model}.\n"
            f"Configuration: {_llm_configuration(ctx)}"
        )

    if command in {"/models", "/model"} and len(parts) == 1:
        return _models_text(ctx)

    if command == "/model":
        if len(parts) < 2:
            return _models_text(ctx)
        model = parts[1]
        try:
            ctx.llm = LLMClient(model=model, provider=_active_provider(ctx))
        except Exception as exc:
            return f"Could not switch model: {exc}"
        return f"Model switched to {ctx.llm.model} on provider {_active_provider(ctx)}."

    if command == "/status":
        return _status_text(ctx)

    if command == "/connect":
        return _connect_text()

    if command == "/help":
        return _slash_help_text()

    if command in {"/exit", "/quit", "/q"}:
        return "Exiting Nym."

    return f"Unknown local command: {parts[0]}"


def _slash_help_text() -> str:
    lines = ["Local commands:"]
    lines.extend(f"{name} - {description}" for name, description in LOCAL_COMMANDS)
    lines.append("Ctrl+P - open command palette")
    lines.append("Tab - complete a unique slash command")
    return "\n".join(lines)


def _providers_text(ctx: AppContext) -> str:
    provider = _active_provider(ctx)
    providers = ", ".join(sorted(SUPPORTED_PROVIDERS))
    return (
        f"Active provider: {provider}\n"
        f"Active model: {ctx.llm.model}\n"
        f"Mode: {_llm_mode(ctx)}\n"
        f"Endpoint: {_llm_endpoint(ctx)}\n"
        f"Configuration: {_llm_configuration(ctx)}\n"
        f"Available providers: {providers}\n"
        "Switch with: /provider <provider> [model]"
    )


def _models_text(ctx: AppContext) -> str:
    active_provider = _active_provider(ctx)
    lines = [
        f"Active provider: {active_provider}",
        f"Active model: {ctx.llm.model}",
        "",
        "Suggested models:",
    ]
    for provider in sorted(SUPPORTED_PROVIDERS):
        hints = ", ".join(PROVIDER_MODEL_HINTS.get(provider, ("custom-model",)))
        marker = "*" if provider == active_provider else " "
        lines.append(f"{marker} {provider}: {hints}")
    lines.extend([
        "",
        "Switch model: /model <model>",
        "Switch provider and model: /provider <provider> <model>",
    ])
    return "\n".join(lines)


def _status_text(ctx: AppContext) -> str:
    pending_approvals = [
        item
        for item in getattr(ctx.session, "pending_approvals", [])
        if isinstance(item, dict) and item.get("status") == "pending"
    ]
    return "\n".join([
        f"Session: {ctx.session_id}",
        f"Root: {ctx.workspace_root}",
        f"Provider: {_active_provider(ctx)}",
        f"Model: {ctx.llm.model}",
        f"Mode: {_llm_mode(ctx)}",
        f"Endpoint: {_llm_endpoint(ctx)}",
        f"Configuration: {_llm_configuration(ctx)}",
        f"Pending approvals: {len(pending_approvals)}",
    ])


def _connect_text() -> str:
    return "\n".join([
        "Provider setup:",
        "OpenAI: set OPENAI_API_KEY, then /provider openai [model]",
        "Anthropic: set ANTHROPIC_API_KEY, then /provider anthropic [model]",
        "DeepSeek hosted: set DEEPSEEK_API_KEY, then /provider deepseek [model]",
        "GLM hosted: set GLM_API_KEY or ZAI_API_KEY, then /provider glm [model]",
        "Ollama local: start Ollama, pull a model, then /provider ollama <model>",
        "LM Studio local: start the local server, then /provider lmstudio <model>",
    ])


def _llm_mode(ctx: AppContext) -> str:
    value = getattr(getattr(ctx, "llm", None), "mode", "")
    return str(value or "unknown")


def _llm_endpoint(ctx: AppContext) -> str:
    value = getattr(getattr(ctx, "llm", None), "endpoint", "")
    return str(value or "not configured")


def _llm_configuration(ctx: AppContext) -> str:
    error = getattr(getattr(ctx, "llm", None), "configuration_error", None)
    return str(error) if error else "ready"


def _is_exit_command(value: str) -> bool:
    return value.strip().casefold() in {"exit", "quit", "/exit", "/quit", "/q"}


def _active_provider(ctx: Any) -> str:
    provider = getattr(getattr(ctx, "llm", None), "provider", None)
    return str(provider or "openai")


def print_session_history(store: SessionStore, session_id: str) -> None:
    messages = store.list_messages(session_id, limit=None)
    if not messages:
        return

    print("Loaded session history")
    print()
    for message in messages:
        print(f"{message.role}:")
        print(message.content)
        print()


def list_sessions(store: SessionStore) -> int:
    sessions = store.list_sessions(limit=SESSION_LIST_LIMIT)
    if not sessions:
        print("No sessions found.")
        return 0

    print(f"{'ID':12}  {'UPDATED':>10}  TITLE")
    for item in sessions:
        title = item.last_prompt or item.title
        print(f"{item.id:12}  {format_age(item.updated_at):>10}  {truncate(title, 96)}")

    return 0


def format_age(timestamp: str) -> str:
    try:
        updated = datetime.fromisoformat(timestamp)
    except ValueError:
        return "unknown"

    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)

    seconds = max(0, int((datetime.now(timezone.utc) - updated).total_seconds()))

    if seconds < 60:
        return "now"

    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"

    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"

    days = hours // 24
    if days < 30:
        return f"{days}d ago"

    return updated.date().isoformat()


def truncate(value: str | None, limit: int) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = SessionStore.default()
    command = args.session_command.strip() if args.session_command else None
    session_arg = args.session_id.strip() if args.session_id else None

    if command and command.lower() == "list":
        if session_arg:
            parser.error("'list' does not accept a session id.")
        return list_sessions(store)

    use_tui = args.tui or (command and command.lower() == "tui")

    if command and command.lower() == "tui":
        command = None

    if command and command.lower() == "resume":
        if session_arg:
            session_info = load_existing_session(session_arg, store)
        else:
            session_info = choose_session(store)
            if session_info is None:
                return 0
    else:
        if session_arg:
            parser.error(f"unrecognized arguments: {session_arg}")

        session_info = (
            load_existing_session(command, store)
            if command
            else create_new_session(args, store)
        )

    ctx = build_context(args, store=store, session_info=session_info)

    if use_tui:
        return run_tui(ctx)

    print_session_history(store, session_info.id)

    if ctx.debug:
        print(f"context={ctx}")

    return repl(ctx)


def run_tui(ctx: AppContext) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("TUI requires an interactive terminal.")
        return 1

    try:
        curses.wrapper(_run_tui, ctx)
    except KeyboardInterrupt:
        print()
        return 0
    except curses.error as exc:
        print(f"Failed to start TUI: {exc}")
        return 1
    finally:
        _stop_language_servers(ctx)

    return 0


def _stop_language_servers(ctx: Any) -> None:
    manager = getattr(ctx, "language_servers", None)
    stop_all = getattr(manager, "stop_all", None)
    if callable(stop_all):
        stop_all()


def _run_tui(stdscr: Any, ctx: AppContext) -> None:
    curses.curs_set(1)
    stdscr.keypad(True)
    stdscr.nodelay(True)
    _setup_tui_colors()

    prompt = ""
    prompt_history: list[str] = []
    history_index: int | None = None
    transcript_scroll = 0
    transcript_at_bottom = True
    event_scroll = 0
    status = "Ready"
    live_turn = LiveTurnState()
    approval_queue = ApprovalQueueState(ctx)
    worker: threading.Thread | None = None
    worker_error: str | None = None
    active_prompt: str | None = None
    queued_prompts: deque[str] = deque()
    palette_index = 0

    def launch_turn(user_prompt: str, *, queue_if_busy: bool = True) -> None:
        nonlocal worker, worker_error, active_prompt, status

        if worker is not None and worker.is_alive():
            if queue_if_busy:
                queued_prompts.append(user_prompt)
                status = f"Queued {len(queued_prompts)} prompt(s)"
            else:
                status = "Agent is still working"
            return

        active_prompt = user_prompt
        worker_error = None
        status = "Thinking..."
        live_turn.start(user_prompt)

        def _target() -> None:
            nonlocal worker_error, active_prompt, status
            try:
                handle_prompt(
                    ctx,
                    user_prompt,
                    stream_event=live_turn.update,
                    approval_requester=approval_queue.request,
                )
                status = "Ready"
                live_turn.finish()
            except Exception as exc:  # pragma: no cover - surfaced in UI
                worker_error = str(exc)
                status = "Error"
                live_turn.finish(str(exc))
            finally:
                active_prompt = None

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()

    while True:
        height, width = stdscr.getmaxyx()
        messages = ctx.store.list_messages(ctx.session_id, limit=None)
        session_info = ctx.store.get_session(ctx.session_id)
        show_usage_panel = width >= USAGE_PANEL_MIN_TERMINAL_WIDTH
        panel_width = USAGE_PANEL_WIDTH if show_usage_panel else 0
        panel_gap = 1 if show_usage_panel else 0
        content_width = max(20, width - panel_width - panel_gap)
        approval_snapshot = approval_queue.snapshot()

        if worker is not None and not worker.is_alive() and active_prompt is None and status == "Thinking...":
            status = "Ready"

        if worker is not None and not worker.is_alive() and active_prompt is None and queued_prompts:
            launch_turn(queued_prompts.popleft(), queue_if_busy=False)
            continue

        rendered_messages = _render_tui_transcript(
            messages,
            live_turn.snapshot(),
            max(20, content_width - 2),
        )
        palette_entries = _slash_palette_entries(prompt)
        if palette_entries:
            palette_index = min(palette_index, len(palette_entries) - 1)
        else:
            palette_index = 0
        command_palette = _slash_command_lines(prompt, max(20, content_width - 2), selected_index=palette_index)
        palette_height = min(5, len(command_palette))

        header_height = 3
        status_y = max(header_height + 1, height - 3)
        input_y = max(header_height + 2, height - 2)
        palette_y = max(header_height + 1, status_y - palette_height)
        transcript_height = max(5, (palette_y if palette_height else status_y) - header_height)

        max_transcript_scroll = max(0, len(rendered_messages) - transcript_height)
        if transcript_at_bottom:
            transcript_scroll = 0
        else:
            transcript_scroll = min(transcript_scroll, max_transcript_scroll)
            transcript_scroll = max(0, transcript_scroll)

        stdscr.erase()
        _draw_header(stdscr, ctx, width)
        _draw_box_title(stdscr, 2, "Conversation", content_width)
        _draw_lines(
            stdscr,
            header_height,
            transcript_height,
            rendered_messages,
            transcript_scroll,
            content_width,
        )
        if palette_height:
            _draw_command_palette(stdscr, palette_y, command_palette[:palette_height], content_width)
        if show_usage_panel:
            _draw_usage_panel(
                stdscr,
                x=content_width,
                y=2,
                height=max(3, input_y - 2),
                width=panel_width,
                session=session_info,
                model=ctx.llm.model,
                provider=_active_provider(ctx),
                approvals=approval_snapshot,
            )
        _draw_status_line(
            stdscr,
            status_y,
            width,
            status=_queue_status(status, len(queued_prompts), len(approval_snapshot.get("pending", []))),
            live_turn=live_turn.snapshot(),
            worker_alive=worker is not None and worker.is_alive(),
            error=worker_error,
            usage_summary=None if show_usage_panel else _compact_usage_text(session_info, ctx.llm.model, _active_provider(ctx)),
        )
        _draw_input_line(
            stdscr,
            input_y,
            width,
            prompt,
        )
        stdscr.refresh()

        key = stdscr.getch()
        if key == -1:
            time.sleep(0.05)
            continue

        if key == 3:  # Ctrl+C
            if worker is not None and worker.is_alive():
                canceled = ctx.rust.cancel_active()
                status = "Cancelling..." if canceled else "Cancellation requested"
                continue
            break

        if key == 17:  # Ctrl+Q
            break

        if key == 1:  # Ctrl+A
            if approval_queue.approve_selected():
                status = "Approval granted"
            continue

        if key == 4:  # Ctrl+D
            if approval_queue.deny_selected():
                status = "Approval denied"
            continue

        if key == 14:  # Ctrl+N
            approval_queue.next_item()
            continue

        if key == 16:  # Ctrl+P
            if approval_snapshot.get("pending"):
                approval_queue.previous_item()
            elif not prompt:
                prompt = "/"
                palette_index = 0
            continue

        if key in (10, 13):  # Enter
            candidate = prompt.strip()
            if candidate:
                selected = _selected_palette_entry(prompt, palette_index)
                if selected is not None:
                    if selected.execute:
                        candidate = selected.complete_to.strip()
                    else:
                        prompt = selected.complete_to
                        palette_index = 0
                        continue
                if _is_exit_command(candidate):
                    break
                prompt_history.append(candidate)
                history_index = None
                prompt = ""
                transcript_at_bottom = True
                transcript_scroll = 0
                local_answer = _handle_local_command(ctx, candidate)
                if local_answer is not None:
                    ctx.store.update_last_prompt(ctx.session_id, candidate)
                    ctx.store.add_message(ctx.session_id, "user", candidate)
                    ctx.store.add_message(ctx.session_id, "assistant", local_answer)
                    status = truncate(local_answer, 80)
                else:
                    launch_turn(candidate)
            continue

        if key in (curses.KEY_BACKSPACE, 127, 8):
            prompt = prompt[:-1]
            palette_index = 0
            continue

        if key == 9:  # Tab
            completed = _complete_slash_command(prompt)
            if completed is not None:
                prompt = completed
                palette_index = 0
            continue

        if key == curses.KEY_UP:
            if palette_entries:
                palette_index = max(0, palette_index - 1)
            elif prompt:
                if prompt_history:
                    if history_index is None:
                        history_index = len(prompt_history) - 1
                    else:
                        history_index = max(0, history_index - 1)
                    prompt = prompt_history[history_index]
            else:
                step = max(1, transcript_height // 2)
                if transcript_at_bottom:
                    transcript_scroll = step
                    transcript_at_bottom = False
                else:
                    transcript_scroll = min(max_transcript_scroll, transcript_scroll + step)
            continue

        if key == curses.KEY_DOWN:
            if palette_entries:
                palette_index = min(len(palette_entries) - 1, palette_index + 1)
            elif prompt:
                if prompt_history and history_index is not None:
                    if history_index >= len(prompt_history) - 1:
                        history_index = None
                        prompt = ""
                    else:
                        history_index += 1
                        prompt = prompt_history[history_index]
            else:
                step = max(1, transcript_height // 2)
                transcript_scroll = max(0, transcript_scroll - step)
                if transcript_scroll == 0:
                    transcript_at_bottom = True
            continue

        if key == curses.KEY_PPAGE:
            step = max(1, transcript_height // 2)
            if transcript_at_bottom:
                transcript_scroll = step
                transcript_at_bottom = False
            else:
                transcript_scroll = min(max_transcript_scroll, transcript_scroll + step)
            continue

        if key == curses.KEY_NPAGE:
            step = max(1, transcript_height // 2)
            transcript_scroll = max(0, transcript_scroll - step)
            if transcript_scroll == 0:
                transcript_at_bottom = True
            continue

        if 32 <= key <= 126:
            prompt += chr(key)
            palette_index = 0
            continue


def _setup_tui_colors() -> None:
    try:
        curses.use_default_colors()
    except curses.error:
        return
    if not _has_tui_colors():
        return
    curses.start_color()
    pairs = [
        (COLOR_HEADER, curses.COLOR_CYAN, -1),
        (COLOR_USER, curses.COLOR_GREEN, -1),
        (COLOR_ASSISTANT, curses.COLOR_WHITE, -1),
        (COLOR_THINKING, curses.COLOR_MAGENTA, -1),
        (COLOR_GUARDRAIL, curses.COLOR_YELLOW, -1),
        (COLOR_MUTED, curses.COLOR_BLUE, -1),
        (COLOR_ERROR, curses.COLOR_RED, -1),
    ]
    for pair, foreground, background in pairs:
        try:
            curses.init_pair(pair, foreground, background)
        except curses.error:
            continue


def _tui_attr(pair: int, *flags: int) -> int:
    attr = curses.color_pair(pair) if _has_tui_colors() else 0
    for flag in flags:
        attr |= flag
    return attr


def _has_tui_colors() -> bool:
    try:
        return curses.has_colors()
    except curses.error:
        return False


def _draw_header(stdscr: Any, ctx: AppContext, width: int) -> None:
    root_budget = max(12, width - 58)
    title = " nym "
    detail = (
        f"session {ctx.session_id}  provider {_active_provider(ctx)}  model {ctx.llm.model}  "
        f"root {truncate(str(ctx.workspace_root), root_budget)}"
    )
    stdscr.addnstr(0, 0, title, width - 1, _tui_attr(COLOR_HEADER, curses.A_BOLD))
    if width > len(title):
        stdscr.addnstr(
            0,
            len(title),
            detail.ljust(max(0, width - len(title))),
            max(0, width - len(title) - 1),
            _tui_attr(COLOR_MUTED),
        )
    stdscr.addnstr(1, 0, ("-" * max(0, width - 1)), width - 1, _tui_attr(COLOR_MUTED))


def _queue_status(status: str, queued_count: int, approval_count: int = 0) -> str:
    if queued_count <= 0 and approval_count <= 0:
        return status
    parts = [status]
    if queued_count > 0:
        parts.append(f"queued {queued_count}")
    if approval_count > 0:
        parts.append(f"approvals {approval_count}")
    return " | ".join(parts)


def _draw_box_title(stdscr: Any, y: int, title: str, width: int) -> None:
    line = f" {title} "
    stdscr.addnstr(y, 0, line.ljust(width), width - 1, _tui_attr(COLOR_HEADER, curses.A_BOLD))


def _draw_lines(stdscr: Any, start_y: int, height: int, lines: list[str], scroll: int, width: int) -> None:
    if scroll <= 0:
        visible = lines[-height:]
    else:
        end = max(0, len(lines) - scroll)
        start = max(0, end - height)
        visible = lines[start:end]
    for offset, line in enumerate(visible):
        stdscr.addnstr(start_y + offset, 0, line.ljust(width), width - 1, _line_attr(line))
    for offset in range(len(visible), height):
        stdscr.addnstr(start_y + offset, 0, " ".ljust(width), width - 1)


def _slash_command_lines(prompt: str, width: int, *, selected_index: int = 0) -> list[str]:
    entries = _slash_palette_entries(prompt)
    if not entries:
        return []
    title = _slash_palette_title(prompt)
    lines = [_clip_line(title, width)]
    for index, entry in enumerate(entries[:4]):
        marker = ">" if index == selected_index else " "
        lines.append(_clip_line(f"{marker} {entry.label:<16} {entry.description}", width))
    return lines


def _slash_palette_entries(prompt: str) -> list[PaletteEntry]:
    if not prompt.startswith("/"):
        return []

    stripped = prompt.strip()
    parts = stripped.split()
    if _is_provider_palette_prompt(prompt, parts):
        return _provider_palette_entries(parts[1] if len(parts) >= 2 else "")
    if _is_model_palette_prompt(prompt, parts):
        return _model_palette_entries(parts[1] if len(parts) >= 2 else "")

    query = parts[0].casefold() if parts else "/"
    matches = [
        PaletteEntry(
            value=name,
            label=name,
            description=description,
            complete_to=f"{name} " if name in {"/provider", "/model"} else name,
            execute=name not in {"/provider", "/model"},
        )
        for name, description in LOCAL_COMMANDS
        if name.casefold().startswith(query)
    ]
    matches.sort(key=lambda entry: entry.value.casefold() != query)
    if not matches:
        matches = [
            PaletteEntry(
                value=name,
                label=name,
                description=description,
                complete_to=f"{name} " if name in {"/provider", "/model"} else name,
                execute=name not in {"/provider", "/model"},
            )
            for name, description in LOCAL_COMMANDS
        ]
    return matches


def _provider_palette_entries(query: str) -> list[PaletteEntry]:
    normalized = query.casefold()
    providers = sorted(SUPPORTED_PROVIDERS)
    matches = [provider for provider in providers if provider.casefold().startswith(normalized)]
    if not matches:
        matches = providers
    return [
        PaletteEntry(
            value=provider,
            label=provider,
            description=f"default model: {PROVIDER_MODEL_HINTS.get(provider, ('custom-model',))[0]}",
            complete_to=f"/provider {provider}",
            execute=True,
        )
        for provider in matches
    ]


def _model_palette_entries(query: str) -> list[PaletteEntry]:
    normalized = query.casefold()
    models = sorted({model for hints in PROVIDER_MODEL_HINTS.values() for model in hints})
    matches = [model for model in models if model.casefold().startswith(normalized)]
    if not matches:
        matches = models
    return [
        PaletteEntry(
            value=model,
            label=model,
            description="switch active provider to this model",
            complete_to=f"/model {model}",
            execute=True,
        )
        for model in matches
    ]


def _slash_palette_title(prompt: str) -> str:
    stripped = prompt.strip()
    parts = stripped.split()
    if _is_provider_palette_prompt(prompt, parts):
        return "Providers"
    if _is_model_palette_prompt(prompt, parts):
        return "Models"
    return "Commands"


def _is_provider_palette_prompt(prompt: str, parts: list[str]) -> bool:
    return (
        len(parts) <= 2
        and (prompt.startswith("/provider ") or (len(parts) >= 1 and parts[0].casefold() == "/provider" and prompt.endswith(" ")))
    )


def _is_model_palette_prompt(prompt: str, parts: list[str]) -> bool:
    return (
        len(parts) <= 2
        and (prompt.startswith("/model ") or (len(parts) >= 1 and parts[0].casefold() == "/model" and prompt.endswith(" ")))
    )


def _selected_palette_entry(prompt: str, selected_index: int) -> PaletteEntry | None:
    entries = _slash_palette_entries(prompt)
    if not entries:
        return None
    return entries[min(max(0, selected_index), len(entries) - 1)]


def _complete_slash_command(prompt: str) -> str | None:
    selected = _selected_palette_entry(prompt, 0)
    if selected is not None and (prompt.startswith("/provider ") or prompt.startswith("/model ")):
        return selected.complete_to
    if not prompt.startswith("/") or " " in prompt:
        return None
    query = prompt.casefold()
    matches = [name for name, _description in LOCAL_COMMANDS if name.casefold().startswith(query)]
    if len(matches) == 1:
        return f"{matches[0]} "
    return None


def _draw_command_palette(stdscr: Any, y: int, lines: list[str], width: int) -> None:
    for offset, line in enumerate(lines):
        attr = _tui_attr(COLOR_HEADER, curses.A_BOLD) if offset == 0 else _tui_attr(COLOR_MUTED)
        stdscr.addnstr(y + offset, 0, line.ljust(width), width - 1, attr)


def _line_attr(line: str) -> int:
    stripped = line.strip()
    if stripped.startswith("You"):
        return _tui_attr(COLOR_USER, curses.A_BOLD)
    if stripped.startswith("Nym"):
        return _tui_attr(COLOR_ASSISTANT, curses.A_BOLD)
    if stripped.startswith("Thinking") or stripped.startswith("Tool") or stripped.startswith("Result"):
        return _tui_attr(COLOR_THINKING)
    if stripped.startswith("Guardrail"):
        return _tui_attr(COLOR_GUARDRAIL, curses.A_BOLD)
    if stripped.startswith("Error"):
        return _tui_attr(COLOR_ERROR, curses.A_BOLD)
    if stripped.startswith("Activity"):
        return _tui_attr(COLOR_MUTED, curses.A_BOLD)
    return 0


def _draw_usage_panel(
    stdscr: Any,
    *,
    x: int,
    y: int,
    height: int,
    width: int,
    session: SessionInfo,
    model: str,
    provider: str,
    approvals: dict[str, Any] | None = None,
) -> None:
    if width <= 3 or height <= 0:
        return
    separator_attr = _tui_attr(COLOR_MUTED)
    for offset in range(height):
        stdscr.addnstr(y + offset, x, "|", 1, separator_attr)

    panel_x = x + 2
    panel_width = max(1, width - 2)
    lines = _usage_panel_lines(session, model, provider, panel_width)
    approval_lines = _approval_panel_lines(approvals or {"pending": [], "selected_index": 0}, panel_width)
    lines.extend([""] + approval_lines)
    for offset in range(height):
        text = lines[offset] if offset < len(lines) else ""
        attr = _panel_line_attr(text)
        stdscr.addnstr(y + offset, panel_x, text.ljust(panel_width), panel_width - 1, attr)


def _panel_line_attr(line: str) -> int:
    if line.startswith("Usage"):
        return _tui_attr(COLOR_HEADER, curses.A_BOLD)
    if line.startswith("Guardrails"):
        return _tui_attr(COLOR_GUARDRAIL, curses.A_BOLD)
    if line.startswith("Approvals"):
        return _tui_attr(COLOR_HEADER, curses.A_BOLD)
    if line.startswith("> "):
        return _tui_attr(COLOR_HEADER, curses.A_BOLD)
    if line.startswith("Provider") or line.startswith("Model"):
        return _tui_attr(COLOR_ASSISTANT, curses.A_BOLD)
    if line.startswith("Context") or line.startswith("Tokens") or line.startswith("Cost"):
        return _tui_attr(COLOR_ASSISTANT, curses.A_BOLD)
    if line.startswith("Approve ") or line.startswith("Deny ") or line.startswith("Pending"):
        return _tui_attr(COLOR_GUARDRAIL)
    return _tui_attr(COLOR_MUTED)


def _usage_panel_lines(session: SessionInfo, model: str, provider: str, width: int) -> list[str]:
    usage = session.tokens
    total_tokens = _billable_token_total(usage)
    context_limit = _context_window_for_model(model)
    percent = _usage_percent(total_tokens, context_limit)
    lines = [
        "Usage",
        "",
        f"Provider   {provider}",
        f"Model      {model}",
        "",
        f"Tokens     {_format_count(total_tokens)}",
        f"Context    {_format_percent(percent)}",
        f"Cost       {_format_cost(session.cost_usd)}",
        "",
        f"Input      {_format_count(usage.input)}",
        f"Output     {_format_count(usage.output)}",
        f"Reasoning  {_format_count(usage.reasoning)}",
        f"Cached     {_format_count(usage.cache_read)}",
        "",
        "Guardrails",
        "Visible when tools are",
        "blocked or need approval.",
    ]
    if context_limit is None:
        lines.insert(5, "Context limit unknown")
    return [_clip_line(line, width) for line in lines]


def _approval_panel_lines(approvals: dict[str, Any], width: int) -> list[str]:
    pending = approvals.get("pending") if isinstance(approvals, dict) else []
    selected_index = approvals.get("selected_index") if isinstance(approvals, dict) else 0
    pending_items = pending if isinstance(pending, list) else []
    if not pending_items:
        return [_clip_line("Approvals", width), "", _clip_line("None pending", width)]

    lines = [_clip_line("Approvals", width), ""]
    for index, item in enumerate(pending_items[:6], start=1):
        tool = _approval_text(item.get("tool"))
        path = _approval_text(item.get("translated_path") or item.get("resolved_path") or item.get("requested_path"))
        reason = _approval_text(item.get("reason"))
        prefix = ">" if index - 1 == selected_index else " "
        lines.append(_clip_line(f"{prefix} {index}. {tool}", width))
        if path:
            lines.append(_clip_line(f"   {path}", width))
        if reason:
            lines.append(_clip_line(f"   {reason}", width))
    return lines


def _approval_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def _compact_usage_text(session: SessionInfo, model: str, provider: str = "openai") -> str:
    total_tokens = _billable_token_total(session.tokens)
    percent = _usage_percent(total_tokens, _context_window_for_model(model))
    return (
        f"{provider}/{model}"
        f" tokens {_format_count(total_tokens)}"
        f" ({_format_percent(percent)})"
        f" cost {_format_cost(session.cost_usd)}"
    )


def _billable_token_total(usage: TokenUsage) -> int:
    return max(0, usage.input) + max(0, usage.output)


def _context_window_for_model(model: str) -> int | None:
    env_value = os.environ.get("NYM_CONTEXT_WINDOW_TOKENS", "").strip()
    if env_value:
        try:
            parsed = int(env_value.replace("_", ""))
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    normalized = model.strip().casefold()
    if normalized in DEFAULT_CONTEXT_WINDOWS:
        return DEFAULT_CONTEXT_WINDOWS[normalized]
    for prefix, limit in DEFAULT_CONTEXT_WINDOWS.items():
        if normalized.startswith(prefix.casefold()):
            return limit
    return None


def _usage_percent(tokens: int, context_limit: int | None) -> float | None:
    if context_limit is None or context_limit <= 0:
        return None
    return max(0.0, (tokens / context_limit) * 100.0)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.1:
        return "<0.1%"
    if value < 10:
        return f"{value:.1f}%"
    return f"{value:.0f}%"


def _format_count(value: int) -> str:
    value = max(0, int(value))
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def _format_cost(value: float) -> str:
    value = max(0.0, float(value))
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _clip_line(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return f"{value[: width - 1]}..."


def _draw_status_line(
    stdscr: Any,
    y: int,
    width: int,
    *,
    status: str,
    live_turn: dict[str, Any],
    worker_alive: bool,
    error: str | None,
    usage_summary: str | None = None,
) -> None:
    footer = f" {status}"
    if worker_alive:
        footer = f"{footer}  working"
    phase = live_turn.get("phase")
    if isinstance(phase, str) and phase not in {"idle", "completed"}:
        footer = f"{footer}  {phase}"
    if error:
        footer = f"{footer}  {truncate(error, max(0, width - 18))}"
    if usage_summary:
        footer = f"{footer}  {usage_summary}"
    help_text = "Ctrl+C cancel/exit  PgUp/PgDn scroll  Ctrl+A approve  Ctrl+D deny  Ctrl+N/P select"
    gap = max(1, width - len(footer) - len(help_text) - 1)
    line = f"{footer}{' ' * gap}{help_text}"
    attr = _tui_attr(COLOR_ERROR, curses.A_BOLD) if error else _tui_attr(COLOR_MUTED)
    stdscr.addnstr(y, 0, line.ljust(width), width - 1, attr)


def _draw_input_line(
    stdscr: Any,
    y: int,
    width: int,
    prompt: str,
) -> None:
    label = " nym> "
    body_width = max(0, width - len(label))
    visible_prompt = prompt[-body_width:]
    line = f"{label}{visible_prompt}"
    stdscr.addnstr(y, 0, line.ljust(width), width - 1, _tui_attr(COLOR_ASSISTANT, curses.A_REVERSE))
    cursor_x = min(width - 1, len(label) + len(visible_prompt))
    stdscr.move(y, cursor_x)


def _render_tui_transcript(messages: list[Any], live_turn: dict[str, Any], width: int) -> list[str]:
    active = bool(live_turn.get("active"))
    error = live_turn.get("error")
    prompt = live_turn.get("prompt", "")
    visible_messages = list(messages)

    if active and prompt and visible_messages:
        last = visible_messages[-1]
        if (
            getattr(last, "role", None) == "user"
            and getattr(last, "content", "").strip() == str(prompt).strip()
        ):
            visible_messages = visible_messages[:-1]

    lines = _render_messages(visible_messages, width) if visible_messages else []
    live_lines = _render_live_turn(live_turn, width) if active or error else []

    if lines and live_lines:
        return [*lines, *live_lines]
    if lines:
        return lines
    if live_lines:
        return live_lines
    return ["No messages yet."]


def _render_messages(messages: list[Any], width: int) -> list[str]:
    lines: list[str] = []
    for message in messages:
        speaker = "You" if message.role == "user" else "Nym" if message.role == "assistant" else message.role.title()
        lines.append(f"{speaker}  {message.created_at}")
        body = message.content.strip() or "<empty>"
        for paragraph in body.splitlines() or [""]:
            wrapped = textwrap.wrap(
                paragraph,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            if wrapped:
                lines.extend(f"  {line}" for line in wrapped)
            else:
                lines.append("")
        lines.append("")
    return lines or ["No messages yet."]


def _render_live_turn(live_turn: dict[str, Any], width: int) -> list[str]:
    phase = live_turn.get("phase")
    feed = live_turn.get("feed", [])
    active = live_turn.get("active", False)

    if not active and phase in {"idle", None} and not live_turn.get("error"):
        return []

    lines: list[str] = []
    prompt = live_turn.get("prompt", "")
    if prompt:
        lines.append(f"You")
        lines.extend(_wrap_lines(prompt, width, indent="  "))
        lines.append("")
        lines.append("Nym")

    prev_kind: str | None = None
    for kind, content in feed:
        if kind in {"thinking", "reasoning"}:
            if prev_kind not in {"thinking", "reasoning"}:
                lines.append("  Activity")
            lines.extend(_wrap_lines(str(content), width, indent="    Thinking: "))
        elif kind == "text":
            for para in content.splitlines() or [""]:
                pieces = textwrap.wrap(para, width=max(1, width - 2), break_long_words=True) or [""]
                lines.extend(f"  {p}" for p in pieces)
        elif kind == "tool":
            if prev_kind != "tool":
                lines.append("  Activity")
            lines.extend(_wrap_lines(str(content), width, indent="    "))
        elif kind == "tool_result":
            lines.extend(_wrap_lines(str(content), width, indent="    Result: "))
        elif kind == "guardrail":
            lines.extend(_wrap_lines(str(content), width, indent="    Guardrail: "))
        elif kind == "tool_error":
            lines.extend(_wrap_lines(str(content), width, indent="    Error: "))
        prev_kind = kind

    if active and not feed:
        lines.append("  Activity")
        lines.append("    Thinking: starting")

    error = live_turn.get("error")
    if error:
        lines.append(f"  Error: {truncate(error, width - 9)}")

    lines.append("")
    return lines


def _wrap_lines(text: str, width: int, *, indent: str = "") -> list[str]:
    wrapped: list[str] = []
    for paragraph in text.splitlines() or [""]:
        pieces = textwrap.wrap(
            paragraph,
            width=max(1, width - len(indent)),
            break_long_words=False,
            break_on_hyphens=False,
        )
        if pieces:
            wrapped.extend(f"{indent}{piece}" for piece in pieces)
        else:
            wrapped.append(indent.rstrip())
    return wrapped
