from __future__ import annotations

import argparse
import curses
import os
import subprocess
import sys
import uuid
import textwrap
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from enum import Enum
import json
import urllib.error
import urllib.request

from .context_builder import build_stored_context
from .language_servers import LanguageServerManager
from .llm import LLMClient, SUPPORTED_PROVIDERS, _normalize_provider
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
    "gpt-5.5": 1_000_000,
    "gpt-5.5-mini": 400_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-3-5-sonnet-latest": 200_000,
    "claude-3-5-haiku-latest": 200_000,
    "llama3.1": 128_000,
    "llama3.3": 128_000,
    "qwen2.5-coder": 128_000,
    "qwen3": 128_000,
    "qwen3.5": 128_000,
    "qwen3.6": 256_000,
    "qwen3-coder": 256_000,
    "qwen3-coder-next": 256_000,
    "deepseek-r1": 128_000,
    "deepseek-v3": 128_000,
    "deepseek-v3.2": 128_000,
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    "deepseek-v4-flash": 128_000,
    "deepseek-v4-pro": 128_000,
    "glm-4": 128_000,
    "glm-4.5": 128_000,
    "glm-4.7": 128_000,
    "glm-4.7-flash": 128_000,
    "glm-5": 128_000,
    "glm-5.1": 128_000,
    "gemma3": 128_000,
    "gemma4": 128_000,
    "codestral": 128_000,
    "codellama": 128_000,
    "gpt-oss": 128_000,
    "starcoder2": 128_000,
}


LOCAL_COMMANDS = (
    ("/model", "Open model picker or switch model"),
    ("/status", "Show session and model status"),
    ("/connect", "Set up hosted or local models"),
    ("/help", "Show commands and shortcuts"),
    ("/exit", "Exit Nym"),
)

PROVIDER_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "glm": "GLM_API_KEY",
    "openai-compatible": "NYM_OPENAI_COMPAT_API_KEY",
}

PROVIDER_LOGIN_URLS = {
    "copilot": "https://github.com/login/device",
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "gemini": "https://aistudio.google.com/app/apikey",
    "groq": "https://console.groq.com/keys",
    "openrouter": "https://openrouter.ai/settings/keys",
    "bedrock": "https://console.aws.amazon.com/bedrock/home",
    "azure": "https://ai.azure.com",
    "vertexai": "https://console.cloud.google.com/vertex-ai",
    "deepseek": "https://platform.deepseek.com/api_keys",
    "glm": "https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
    "openai-compatible": "https://platform.openai.com/api-keys",
}
PROVIDER_DISPLAY_NAMES = {
    "copilot": "GitHub Copilot",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "bedrock": "AWS Bedrock",
    "azure": "Azure OpenAI",
    "vertexai": "Google Cloud Vertex AI",
    "deepseek": "DeepSeek",
    "glm": "GLM",
    "openai-compatible": "Custom OpenAI-compatible",
    "ollama": "Ollama",
    "lmstudio": "LM Studio",
}
PROVIDER_ARGUMENT_COMMANDS = {"/provider", "/login", "/auth", "/apikey", "/key"}
LOCAL_PROVIDERS = {"ollama", "lmstudio"}
PROVIDER_SORT_ORDER = {
    "copilot": 0,
    "anthropic": 1,
    "openai": 2,
    "gemini": 3,
    "groq": 4,
    "openrouter": 5,
    "bedrock": 6,
    "azure": 7,
    "vertexai": 8,
    "deepseek": 9,
    "glm": 10,
    "ollama": 11,
    "lmstudio": 12,
    "openai-compatible": 13,
}
PROVIDER_MODEL_HINTS = {
    "copilot": (
        "gpt-4.1",
        "gpt-5.4-mini",
        "claude-sonnet-4.5",
        "gemini-2.5-pro",
    ),
    "openai": (
        "gpt-5.5",
        "gpt-5.5-mini",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    ),
    "anthropic": (
        "claude-sonnet-4.5",
        "claude-opus-4.1",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ),
    "gemini": (
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
    ),
    "groq": (
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct-0905",
    ),
    "openrouter": (
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-5.4-mini",
        "google/gemini-2.5-pro",
        "meta-llama/llama-3.3-70b-instruct",
    ),
    "bedrock": (
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-opus-4-1-20250805-v1:0",
        "amazon.nova-pro-v1:0",
        "meta.llama3-3-70b-instruct-v1:0",
    ),
    "azure": (
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ),
    "vertexai": (
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "claude-sonnet-4.5",
        "claude-opus-4.1",
    ),
    "ollama": (
        "qwen3.6",
        "qwen3-coder",
        "qwen3-coder-next",
        "qwen3.5",
        "qwen3",
        "qwen2.5-coder",
        "qwen2.5",
        "deepseek-r1",
        "deepseek-v3.2",
        "deepseek-v3",
        "deepseek-coder-v2",
        "deepseek-coder",
        "llama3.3",
        "llama3.1",
        "llama3.2",
        "llama4",
        "gemma4",
        "gemma3",
        "gemma3n",
        "mistral-small3.2",
        "mistral-small",
        "mistral-nemo",
        "mistral",
        "mixtral",
        "codestral",
        "codegemma",
        "codellama",
        "gpt-oss",
        "phi4",
        "phi4-reasoning",
        "phi4-mini",
        "granite-code",
        "starcoder2",
        "glm-5.1",
        "glm-4.7-flash",
    ),
    "lmstudio": ("local-model",),
    "openai-compatible": ("local-model",),
    "deepseek": (
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-chat",
        "deepseek-reasoner",
    ),
    "glm": ("glm-5.1", "glm-5", "glm-4.7", "glm-4.7-flash", "glm-4", "glm-4.5"),
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

    pending_provider: str | None = None
    pending_model: str | None = None

@dataclass(frozen=True)
class PaletteEntry:
    value: str
    label: str
    description: str
    complete_to: str
    execute: bool = False

class AccessMode(Enum):
    NO_AUTH = "no_auth"
    API_KEY = "api_key"
    BROWSER_LOGIN = "browser_login"
    OPTIONAL_API_KEY = "optional_api_key"


@dataclass(frozen=True, slots=True)
class ModelRecord:
    id: str
    display_name: str
    provider_id: str
    access: "AccessMode"
    state: "ModelState"
    capabilities: frozenset[str]
    context_window: int | None
    local: bool
    installed: bool | None
    selectable: bool
    action_label: str | None = None

class ModelState(Enum):
    READY = "ready"
    AUTH_REQUIRED = "auth_required"
    LOGIN_REQUIRED = "login_required"
    SERVER_OFFLINE = "server_offline"
    MODEL_NOT_INSTALLED = "model_not_installed"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"

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
        self._reasoning_buf = ""

    def _flush_text(self) -> None:
        text = self._text_buf.strip()
        if text:
            self.feed.append(("text", text))
        self._text_buf = ""

    def _drop_reasoning(self) -> None:
        self.feed = [(kind, content) for kind, content in self.feed if kind not in {"thinking", "reasoning"}]

    def update(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        delta = event.get("delta")
        with self.lock:
            if kind == "reasoning_delta" and isinstance(delta, str):
                self._reasoning_buf += delta
                self.phase = "reasoning"
            elif kind == "text_delta" and isinstance(delta, str):
                self._drop_reasoning()
                self._flush_reasoning()
                self._text_buf += delta
                self.phase = "responding"
            elif kind == "tool_call_started":
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
            self._drop_reasoning()
            self.active = False
            self.phase = "error" if error else "completed"
            self.error = error

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            feed_snapshot = list(self.feed)
            if self._reasoning_buf and not self._text_buf:
                feed_snapshot.append(("reasoning", self._reasoning_buf))
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
        choices=sorted(SUPPORTED_PROVIDERS),
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
        "--tui-bridge",
        choices=("snapshot", "submit", "stream-submit", "complete", "approve", "deny"),
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--bridge-session-id",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--bridge-prompt",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--bridge-request-id",
        default=None,
        help=argparse.SUPPRESS,
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
    provider = getattr(args, "provider", None) or session_info.provider
    model = _selected_model(args, session_info)
    llm = LLMClient(model=model, provider=provider)
    if session_info.provider != llm.provider or session_info.model != llm.model:
        store.update_llm_config(session_info.id, provider=llm.provider, model=llm.model)

    return AppContext(
        workspace_root=workspace_root,
        search_roots=parse_search_roots(workspace_root),
        rust=RustTools(rust_bin=rust_bin),
        llm=llm,
        language_servers=LanguageServerManager(),
        session=session,
        session_id=session_info.id,
        store=store,
        stored_context=stored_context,
        debug=args.debug,
    )


def _selected_model(args: argparse.Namespace, session_info: SessionInfo) -> str | None:
    if getattr(args, "model", None):
        return args.model
    provider_override = getattr(args, "provider", None)
    if provider_override and provider_override != session_info.provider:
        return None
    return session_info.model


def default_workspace_root(args: argparse.Namespace) -> Path:
    if args.root:
        return Path(args.root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if cwd == Path("/") and (repo_root / "nym-rust").exists() and (repo_root / "README.md").exists():
        return repo_root.resolve()

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
    messages = store.list_messages(session_id, limit=20)
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
    conversation_history = load_session_messages(ctx.store, ctx.session_id)
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
        provider=args.provider,
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
    text = _normalized_command_prompt(user_input.strip())
    if not text.startswith("/"):
        return None
    if text == "/":
        return _slash_help_text()
    parts = text.split()
    command = parts[0].casefold()

    if command in {"/login", "/auth"}:
        provider = parts[1] if len(parts) >= 2 else _active_provider(ctx)
        return _login_provider(ctx, provider)

    if command in {"/apikey", "/key"}:
        provider = parts[1] if len(parts) >= 2 else ""
        api_key = parts[2] if len(parts) >= 3 else ""
        if not provider:
            return "Usage: /apikey <provider> [api-key]"
        if not api_key:
            return (
                f"Paste the key using the hidden TUI prompt: /apikey {provider}\n"
                f"Non-TUI fallback: /apikey {provider} <api-key>"
            )
        return _set_provider_api_key(ctx, provider, api_key)

    if command in {"/models", "/model"} and len(parts) == 1:
        return _models_text(ctx)

    if command == "/model":
        if len(parts) < 2:
            return _models_text(ctx)
        provider: str | None = None
        if len(parts) >= 3:
            try:
                provider = _normalize_provider(parts[1])
                model = parts[2]
            except ValueError:
                provider = None
                model = parts[1]
        else:
            model = parts[1]
        return _switch_model(ctx, model=model, provider=provider)

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


def _provider_switch_text(ctx: AppContext) -> str:
    provider = _active_provider(ctx)
    configuration = _llm_configuration(ctx)
    lines = [f"Model source switched to {_model_source_label(provider)} with model {ctx.llm.model}."]
    if configuration == "ready":
        lines.append("Configuration: ready")
        return "\n".join(lines)

    env_name = PROVIDER_API_KEY_ENVS.get(provider)
    display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider)
    if provider == "openai-compatible" and "BASE_URL" in configuration:
        lines.extend([
            "",
            "OpenAI-compatible models need an endpoint before requests can run.",
            "Set environment variable: NYM_OPENAI_COMPAT_BASE_URL",
            "Optional API key: /apikey openai-compatible",
        ])
        return "\n".join(lines)

    if env_name:
        lines.extend([
            "",
            f"{display_name} needs an API key before requests can run.",
            f"Set key: /apikey {provider}",
        ])
        if provider in PROVIDER_LOGIN_URLS:
            lines.append(f"Open account/API keys: /login {provider}")
        lines.extend([
            f"Environment variable: {env_name}",
            "Keys loaded with /apikey are used for this Nym process and are not written to session history.",
        ])
        return "\n".join(lines)

    lines.append(f"Configuration: {configuration}")
    return "\n".join(lines)


def _providers_text(ctx: AppContext) -> str:
    provider = _active_provider(ctx)
    providers = ", ".join(sorted(SUPPORTED_PROVIDERS))
    return (
        f"Active model: {ctx.llm.model}\n"
        f"Model source: {_model_source_label(provider)}\n"
        f"Mode: {_llm_mode(ctx)}\n"
        f"Endpoint: {_llm_endpoint(ctx)}\n"
        f"Configuration: {_llm_configuration(ctx)}\n"
        f"Available model sources: {providers}\n"
        "Switch with: /model <source> <model>"
    )


def _models_text(ctx: AppContext) -> str:
    active_provider = _active_provider(ctx)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for option in _model_options_for_display(ctx):
        grouped.setdefault(option["provider"], []).append(option)

    lines = [
        f"Active model: {ctx.llm.model}",
        f"Model source: {_model_source_label(active_provider)}",
        "",
        "Models:",
    ]

    for provider in sorted(
        grouped,
        key=lambda item: PROVIDER_SORT_ORDER.get(item, 99),
    ):
        lines.append("")
        lines.append(
            f"{_model_source_label(provider)} - {_provider_access_label(provider)}"
        )
        for option in grouped[provider]:
            marker = (
                "*"
                if (
                    option["provider"] == active_provider
                    and option["model"] == ctx.llm.model
                )
                else " "
            )
            lines.append(
                f"{marker} {option['model']}  "
                f"{_model_state_label(option)}"
            )

    lines.extend([
        "",
        "Switch model: /model <model>",
        "Choose exact source/model: /model <source> <model>",
        "Hosted models open auth automatically when a key is missing.",
        "Local models are not installed in this workspace.",
        "Install them in Ollama or LM Studio first, then switch to them here.",
    ])

    return "\n".join(lines)

def _resolve_local_model_name(
    ctx: Any,
    provider: str,
    requested_model: str,
) -> tuple[str | None, str | None]:
    discovered, discovery_error = _discover_provider_models(
        ctx,
        provider,
    )

    if discovery_error:
        return None, _local_model_setup_error(provider, requested_model, discovery_error)

    # Exact installed model name.
    if requested_model in discovered:
        return requested_model, None

    requested_lower = requested_model.casefold()

    # Allow an untagged alias only when it resolves to one model.
    matches = [
        model
        for model in discovered
        if model.casefold() == requested_lower
        or model.casefold().startswith(
            f"{requested_lower}:"
        )
    ]

    if len(matches) == 1:
        return matches[0], None

    if len(matches) > 1:
        choices = ", ".join(matches)

        return None, (
            f"`{requested_model}` matches multiple installed models: "
            f"{choices}. Choose the exact model name."
        )

    installed = ", ".join(discovered) or "none"

    message = (
        f"Local model `{requested_model}` is not installed. "
        f"Installed models: {installed}."
    )
    if provider == "ollama":
        message = f"{message} Install it with: ollama pull {requested_model}"
    elif provider == "lmstudio":
        message = f"{message} Download or load it in LM Studio first."
    return None, message


def _local_model_setup_error(provider: str, model: str, detail: str) -> str:
    source = _model_source_label(provider)
    if provider == "ollama":
        return (
            f"{source} is not ready for `{model}`.\n"
            "Install and start Ollama, then pull the model first:\n"
            f"ollama pull {model}\n"
            f"Details: {detail}"
        )
    if provider == "lmstudio":
        return (
            f"{source} is not ready for `{model}`.\n"
            "Install LM Studio, load the model, and start its local server first.\n"
            f"Details: {detail}"
        )
    return detail


def _switch_model(
    ctx: Any,
    *,
    model: str,
    provider: str | None = None,
) -> str:
    resolved_provider = provider or _provider_for_model(
        model,
        _active_provider(ctx),
    )

    try:
        candidate = LLMClient(
            model=model,
            provider=resolved_provider,
        )
    except Exception as exc:
        return f"Could not use model `{model}`: {exc}"

    configuration_error = getattr(
        candidate,
        "configuration_error",
        None,
    )

    if configuration_error:
        # Local models should show server/install instructions,
        # not an API-key prompt.
        if resolved_provider in LOCAL_PROVIDERS:
            return _handle_model_setup(candidate)

        # Remember what the user selected while authentication happens.
        ctx.pending_provider = resolved_provider
        ctx.pending_model = model

        lines = [
            f"{model} needs authentication.\n"
            f"Set key: /apikey {resolved_provider}",
        ]
        if resolved_provider in PROVIDER_LOGIN_URLS:
            lines.append(f"Open account/API keys: /login {resolved_provider}")
        lines.append(f"Complete the secure {_model_source_label(resolved_provider)} key prompt to continue.")
        return "\n".join(lines)

    previous_llm = ctx.llm

    try:
        ctx.llm = candidate
        ctx.pending_provider = None
        ctx.pending_model = None
        _persist_llm_config(ctx)
    except Exception as exc:
        ctx.llm = previous_llm
        return f"Could not save model `{model}`: {exc}"

    return _model_switch_text(ctx)


def _normalize_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")

    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    for suffix in (
        "/chat/completions",
        "/responses",
        "/models",
    ):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")

    return url


def _strip_api_suffix(base_url: str) -> str:
    url = _normalize_base_url(base_url)

    if url.endswith("/v1"):
        return url[:-3].rstrip("/")

    return url


def _openai_models_url(base_url: str) -> str:
    url = _normalize_base_url(base_url)

    if not url:
        return ""

    if not url.endswith("/v1"):
        url = f"{url}/v1"

    return f"{url}/models"


def _ollama_tags_url(base_url: str) -> str:
    root = _strip_api_suffix(base_url)

    if not root:
        return ""

    return f"{root}/api/tags"


def _get_json(
    url: str,
    *,
    api_key: str | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    if not url:
        raise RuntimeError("Endpoint is not configured.")

    headers = {
        "Accept": "application/json",
        "User-Agent": "Nym/1.0",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url,
        headers=headers,
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw = response.read().decode(
                "utf-8",
                errors="replace",
            )

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        if exc.code in {401, 403}:
            raise RuntimeError(
                "Authentication failed."
            ) from exc

        if exc.code == 404:
            raise RuntimeError(
                f"Model-list endpoint was not found: {url}"
            ) from exc

        raise RuntimeError(
            f"Endpoint returned HTTP {exc.code}: {body}"
        ) from exc

    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)

        raise RuntimeError(
            f"Could not connect to {url}: {reason}"
        ) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Endpoint returned invalid JSON: {url}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Endpoint returned an unexpected response: {url}"
        )

    return payload


def _discover_ollama_models(
    base_url: str,
) -> list[str]:
    payload = _get_json(
        _ollama_tags_url(base_url),
    )

    items = payload.get("models", [])

    if not isinstance(items, list):
        return []

    models: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        model = item.get("model") or item.get("name")

        if isinstance(model, str) and model:
            models.append(model)

    return sorted(set(models))


def _discover_openai_compatible_models(
    base_url: str,
    *,
    api_key: str | None = None,
) -> list[str]:
    payload = _get_json(
        _openai_models_url(base_url),
        api_key=api_key,
    )

    items = payload.get("data", [])

    if not isinstance(items, list):
        return []

    models: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        model = item.get("id")

        if isinstance(model, str) and model:
            models.append(model)

    return sorted(set(models))


def _provider_base_url(
    ctx: Any,
    provider: str,
) -> str:
    # First use the endpoint already resolved by LLMClient.
    active_llm = getattr(ctx, "llm", None)

    if (
        active_llm is not None
        and getattr(active_llm, "provider", None) == provider
    ):
        endpoint = getattr(active_llm, "endpoint", None)

        if isinstance(endpoint, str) and endpoint.strip():
            return endpoint.strip()

    # Then use provider-specific environment configuration.
    if provider == "ollama":
        return os.environ.get(
            "NYM_OLLAMA_BASE_URL",
            os.environ.get(
                "OLLAMA_HOST",
                "http://localhost:11434",
            ),
        )

    if provider == "lmstudio":
        return os.environ.get(
            "NYM_LMSTUDIO_BASE_URL",
            "http://localhost:1234/v1",
        )

    if provider == "openai-compatible":
        return os.environ.get(
            "NYM_OPENAI_COMPAT_BASE_URL",
            "",
        )

    if provider == "openai":
        return "https://api.openai.com/v1"

    if provider == "groq":
        return os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    if provider == "openrouter":
        return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    return ""



def _discover_provider_models(
    ctx: Any,
    provider: str,
) -> tuple[list[str], str | None]:
    base_url = _provider_base_url(ctx, provider)

    try:
        if provider == "ollama":
            return _discover_ollama_models(base_url), None

        if provider == "lmstudio":
            return _discover_openai_compatible_models(base_url), None

        if provider == "openai-compatible":
            if not base_url:
                return [], "Endpoint is not configured."

            api_key = os.environ.get(
                "NYM_OPENAI_COMPAT_API_KEY",
            )

            return (
                _discover_openai_compatible_models(
                    base_url,
                    api_key=api_key,
                ),
                None,
            )

        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")

            if not api_key:
                return [], "OpenAI API key is required."

            return (
                _discover_openai_compatible_models(
                    "https://api.openai.com/v1",
                    api_key=api_key,
                ),
                None,
            )

        if provider in {"groq", "openrouter"}:
            api_key = os.environ.get(PROVIDER_API_KEY_ENVS.get(provider, ""))

            if not api_key:
                return [], f"{_model_source_label(provider)} API key is required."

            return (
                _discover_openai_compatible_models(
                    base_url,
                    api_key=api_key,
                ),
                None,
            )

        return list(PROVIDER_MODEL_HINTS.get(provider, ())), None

    except RuntimeError as exc:
        return [], str(exc)


def _model_state_label(option: dict[str, Any]) -> str:
    state = option.get("state")

    labels = {
        ModelState.READY: "Ready",
        ModelState.AUTH_REQUIRED: "Add API key",
        ModelState.LOGIN_REQUIRED: "Sign in",
        ModelState.SERVER_OFFLINE: "Server offline",
        ModelState.MODEL_NOT_INSTALLED: "Not installed",
        ModelState.DOWNLOADING: "Downloading",
        ModelState.LOADING: "Loading",
        ModelState.UNAVAILABLE: "Unavailable",
        ModelState.INCOMPATIBLE: "Incompatible",
        ModelState.UNKNOWN: "Setup required",
    }

    return labels.get(state, "Unknown")


def _handle_model_setup(candidate: Any) -> str:
    provider = str(getattr(candidate, "provider", "") or "")
    model = str(getattr(candidate, "model", "") or "")
    error = str(
        getattr(candidate, "configuration_error", "")
        or "Model is not ready."
    )

    if provider == "ollama":
        return (
            f"`{model}` is not ready locally.\n"
            f"Start Ollama, then install it with: ollama pull {model}"
        )

    if provider == "lmstudio":
        return (
            f"`{model}` is not ready locally.\n"
            "Start the LM Studio server and load the model first."
        )

    if provider in PROVIDER_API_KEY_ENVS:
        return (
            f"`{model}` needs an API key.\n"
            f"Details: {error}"
        )

    return f"Could not prepare `{model}`: {error}"    

def _model_switch_text(ctx: Any) -> str:
    provider = _active_provider(ctx)
    model = getattr(getattr(ctx, "llm", None), "model", None)
    configuration = _llm_configuration(ctx)
    lines = [
        f"Model switched to {model}.",
        f"Source: {_model_source_label(provider)}",
        f"Access: {_provider_access_label(provider)}",
    ]
    if configuration == "ready":
        lines.append("Configuration: ready")
        return "\n".join(lines)
    lines.append("")
    lines.append(_provider_switch_text(ctx))
    return "\n".join(lines)


def _provider_for_model(model: str, active_provider: str) -> str:
    providers = _providers_for_model(model)
    if active_provider in providers:
        return active_provider
    if providers:
        local_matches = [provider for provider in providers if provider in LOCAL_PROVIDERS]
        if local_matches:
            return local_matches[0]
        return providers[0]
    return active_provider


def _providers_for_model(model: str) -> list[str]:
    normalized = model.casefold()
    providers: list[str] = []
    for provider in sorted(
        SUPPORTED_PROVIDERS,
        key=lambda item: PROVIDER_SORT_ORDER.get(item, 99),
    ):
        hints = PROVIDER_MODEL_HINTS.get(provider, ())
        if any(hint.casefold() == normalized for hint in hints):
            providers.append(provider)
    return providers


def _discovered_model_options(ctx: Any,) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []

    for provider in sorted(
        SUPPORTED_PROVIDERS,
        key=lambda item: PROVIDER_SORT_ORDER.get(item, 99),
    ):
        discovered, discovery_error = _discover_provider_models(
            ctx,
            provider,
        )

        discovered_set = set(discovered)
        suggestions = PROVIDER_MODEL_HINTS.get(provider, ())

        for model in discovered:
            options.append({
                "provider": provider,
                "model": model,
                "state": ModelState.READY,
                "selectable": True,
                "error": None,
            })

        for model in suggestions:
            if model in discovered_set:
                continue

            if provider in LOCAL_PROVIDERS:
                state = (
                    ModelState.SERVER_OFFLINE
                    if discovery_error
                    else ModelState.MODEL_NOT_INSTALLED
                )
            elif discovery_error and "key" in discovery_error.casefold():
                state = ModelState.AUTH_REQUIRED
            elif discovery_error:
                state = ModelState.UNAVAILABLE
            else:
                state = ModelState.UNKNOWN

            options.append({
                "provider": provider,
                "model": model,
                "state": state,
                "selectable": state in {
                    ModelState.READY,
                    ModelState.AUTH_REQUIRED,
                },
                "error": discovery_error,
            })

    return options

def _model_options() -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []

    for provider in sorted(
        SUPPORTED_PROVIDERS,
        key=lambda item: PROVIDER_SORT_ORDER.get(
            item,
            99,
        ),
    ):
        for model in PROVIDER_MODEL_HINTS.get(
            provider,
            (),
        ):
            options.append({
                "provider": provider,
                "model": model,
                "state": ModelState.UNKNOWN,
                "selectable": True,
                "error": None,
            })

    return options


def _model_options_for_display(ctx: Any) -> list[dict[str, Any]]:
    active_provider = _active_provider(ctx)
    active_model = str(getattr(getattr(ctx, "llm", None), "model", "") or "")
    options = _model_options()

    for option in options:
        provider = option["provider"]
        model = option["model"]
        option["state"] = _hinted_model_state(ctx, provider, model, active_provider, active_model)
        option["selectable"] = True
        option["error"] = None

    return options


def _hinted_model_state(
    ctx: Any,
    provider: str,
    model: str,
    active_provider: str,
    active_model: str,
) -> ModelState:
    if provider == active_provider and model == active_model and _llm_configuration(ctx) == "ready":
        return ModelState.READY

    if provider in LOCAL_PROVIDERS:
        return ModelState.UNKNOWN

    if provider == "openai-compatible":
        return ModelState.UNKNOWN

    env_name = PROVIDER_API_KEY_ENVS.get(provider)
    if env_name and not os.environ.get(env_name):
        return ModelState.AUTH_REQUIRED

    return ModelState.UNKNOWN


def _provider_access_label(provider: str) -> str:
    if provider in LOCAL_PROVIDERS:
        return "local app, install outside workspace"
    if provider == "openai-compatible":
        return "endpoint required"
    if provider == "copilot":
        return "GitHub sign-in"
    if provider in {"bedrock", "vertexai"}:
        return "cloud credentials"
    if provider == "azure":
        return "Azure endpoint and API key"
    return "sign in or API key"


def _model_source_label(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider)


def _status_text(ctx: AppContext) -> str:
    pending_approvals = [
        item
        for item in getattr(ctx.session, "pending_approvals", [])
        if isinstance(item, dict) and item.get("status") == "pending"
    ]
    lines = [
        f"Session: {ctx.session_id}",
        f"Root: {ctx.workspace_root}",
        f"Model: {ctx.llm.model}",
        f"Model source: {_model_source_label(_active_provider(ctx))}",
        f"Mode: {_llm_mode(ctx)}",
        f"Configuration: {_llm_configuration(ctx)}",
    ]
    lines.extend(_status_context_lines(ctx))
    lines.append(f"Pending approvals: {len(pending_approvals)}")
    return "\n".join(lines)


def _status_context_lines(ctx: Any) -> list[str]:
    model = str(getattr(getattr(ctx, "llm", None), "model", "") or "")
    context_limit = _context_window_for_model(model)
    session_info = _ctx_session_info(ctx)
    if session_info is None:
        total_tokens = 0
        cost_usd = 0.0
    else:
        total_tokens = _billable_token_total(session_info.tokens)
        cost_usd = session_info.cost_usd

    lines = [f"Tokens used: {_format_count(total_tokens)}"]
    if context_limit is None:
        lines.append("Context left: n/a")
    else:
        context_left = max(0, context_limit - total_tokens)
        lines.append(f"Context left: {_format_count(context_left)} / {_format_count(context_limit)}")
        lines.append(f"Context used: {_format_percent(_usage_percent(total_tokens, context_limit))}")
    lines.append(f"Cost: {_format_cost(cost_usd)}")
    return lines


def _ctx_session_info(ctx: Any) -> SessionInfo | None:
    store = getattr(ctx, "store", None)
    session_id = getattr(ctx, "session_id", None)
    if store is None or not session_id or not hasattr(store, "get_session"):
        return None
    try:
        return store.get_session(session_id)
    except Exception:
        return None


def _connect_text() -> str:
    return "\n".join([
        "Model setup:",
        "GitHub Copilot: /login copilot, then /model copilot <model>",
        "OpenAI: /login openai, then /apikey openai (OPENAI_API_KEY)",
        "Anthropic: /login anthropic, then /apikey anthropic (ANTHROPIC_API_KEY)",
        "Google Gemini: /login gemini, then /apikey gemini (GOOGLE_API_KEY)",
        "Groq: /login groq, then /apikey groq (GROQ_API_KEY)",
        "OpenRouter: /login openrouter, then /apikey openrouter (OPENROUTER_API_KEY)",
        "AWS Bedrock: /login bedrock, configure AWS_PROFILE or AWS credentials, then /model bedrock <model>",
        "Azure OpenAI: /login azure, set AZURE_OPENAI_ENDPOINT and /apikey azure (AZURE_OPENAI_API_KEY)",
        "Google Cloud Vertex AI: /login vertexai, set GOOGLE_CLOUD_PROJECT and run gcloud application-default auth",
        "DeepSeek hosted: /login deepseek, then /apikey deepseek (DEEPSEEK_API_KEY)",
        "GLM hosted: /login glm, then /apikey glm (GLM_API_KEY)",
        "Ollama local: install Ollama, run `ollama pull <model>`, then /model ollama <model>",
        "LM Studio local: install LM Studio, load a model, start the local server, then /model lmstudio <model>",
        "Local models live in the local app, not in this workspace.",
        "Nym does not download or bundle local models; Ollama or LM Studio owns install, storage, and runtime.",
        "Persistent keys: export the model source environment variable before starting nym.",
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


def _persist_llm_config(ctx: Any) -> None:
    store = getattr(ctx, "store", None)
    session_id = getattr(ctx, "session_id", None)
    if store is None or not session_id or not hasattr(store, "update_llm_config"):
        return
    store.update_llm_config(
        session_id,
        provider=getattr(getattr(ctx, "llm", None), "provider", None),
        model=getattr(getattr(ctx, "llm", None), "model", None),
    )


def _set_provider_api_key(ctx: Any, provider: str, api_key: str) -> str:
    try:
        normalized = _normalize_provider(provider)
    except ValueError as exc:
        return str(exc)

    env_name = PROVIDER_API_KEY_ENVS.get(normalized)
    if env_name is None:
        return f"{_model_source_label(normalized)} does not use an API key."

    os.environ[env_name] = api_key
    active_provider = _active_provider(ctx)
    pending_provider = getattr(ctx, "pending_provider", None)
    pending_model = getattr(ctx, "pending_model", None)
    reload_provider = pending_provider if pending_provider == normalized else active_provider

    if reload_provider == normalized:
        current_model = getattr(getattr(ctx, "llm", None), "model", None)
        reload_model = pending_model if pending_provider == normalized else current_model
        try:
            ctx.llm = LLMClient(model=reload_model, provider=reload_provider)
            ctx.pending_provider = None
            ctx.pending_model = None
            _persist_llm_config(ctx)
        except Exception as exc:
            return f"{_model_source_label(normalized)} key loaded, but reload failed: {exc}"

    return (
        f"{_model_source_label(normalized)} key loaded.\n"
        f"Configuration: {_llm_configuration(ctx)}"
    )


def _provider_api_key_needed(
    ctx: Any,
) -> str | None:
    pending_provider = getattr(
        ctx,
        "pending_provider",
        None,
    )

    if pending_provider:
        if pending_provider in LOCAL_PROVIDERS:
            return None

        if pending_provider in PROVIDER_API_KEY_ENVS:
            return pending_provider

        return None

    provider = _active_provider(ctx)

    if provider in LOCAL_PROVIDERS:
        return None

    if _llm_configuration(ctx) == "ready":
        return None

    if provider == "openai-compatible":
        return None

    if provider in PROVIDER_API_KEY_ENVS:
        return provider

    return None


def _login_provider(ctx: Any, provider: str) -> str:
    try:
        normalized = _normalize_provider(provider)
    except ValueError as exc:
        return str(exc)

    url = PROVIDER_LOGIN_URLS.get(normalized)
    display_name = PROVIDER_DISPLAY_NAMES.get(normalized, normalized)
    if url is None:
        if normalized in {"ollama", "lmstudio"}:
            return f"{display_name} is local. Start its local server, then use /model {normalized} <model>."
        return f"No login URL is configured for {normalized}."

    opened = False
    try:
        opened = bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception:
        opened = False

    status = "Opened" if opened else "Open"
    active_hint = ""
    if normalized == _active_provider(ctx):
        active_hint = f"\nAfter creating a key, return here and run: /apikey {normalized}"
    return f"{status} {display_name} account/API-key page:\n{url}{active_hint}"


def _api_key_prompt_provider(text: str) -> str | None:
    parts = text.strip().split()
    if len(parts) != 2 or parts[0].casefold() not in {"/apikey", "/key"}:
        return None
    try:
        return _normalize_provider(parts[1])
    except ValueError:
        return None




def _starts_auth_candidate(text: str) -> bool:
    normalized = text.strip().casefold()
    return normalized.startswith("/provider ") or normalized.startswith("/model ")


def _redact_local_command(text: str) -> str:
    parts = text.strip().split()
    if len(parts) >= 3 and parts[0].casefold() in {"/apikey", "/key"}:
        return f"{parts[0]} {parts[1]} <redacted>"
    return text


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

    if args.tui_bridge:
        return _run_tui_bridge(args, store)

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

    repo_root = Path(__file__).resolve().parents[1]
    command = [
        str(ctx.rust.rust_bin),
        "tui",
        "--python",
        sys.executable,
        "--repo-root",
        str(repo_root),
        "--session-id",
        ctx.session_id,
    ]
    env = os.environ.copy()
    store_db_path = getattr(getattr(ctx, "store", None), "db_path", None)
    if store_db_path is not None:
        env["NYM_SESSION_DB"] = str(store_db_path)

    try:
        completed = subprocess.run(command, check=False, env=env)
    except OSError as exc:
        print(f"Failed to start Ratatui UI: {exc}")
        return 1
    finally:
        _stop_language_servers(ctx)

    return int(completed.returncode)


def _run_tui_bridge(args: argparse.Namespace, store: SessionStore) -> int:
    session_id = (args.bridge_session_id or "").strip()
    if not session_id:
        print(json.dumps({"ok": False, "error": "Missing bridge session id."}))
        return 1

    ctx: AppContext | None = None
    try:
        session_info = store.get_session(session_id)
        ctx = build_context(args, store=store, session_info=session_info)
        if args.tui_bridge == "snapshot":
            payload = {"ok": True, "snapshot": _tui_bridge_snapshot(ctx)}
        elif args.tui_bridge == "complete":
            payload = {
                "ok": True,
                "completions": _tui_bridge_completions(args.bridge_prompt or ""),
            }
        elif args.tui_bridge in {"approve", "deny"}:
            request_id = (args.bridge_request_id or "").strip()
            if not request_id:
                payload = {"ok": False, "error": "Missing approval request id.", "snapshot": _tui_bridge_snapshot(ctx)}
            else:
                decision = "approved" if args.tui_bridge == "approve" else "denied"
                payload = _tui_bridge_apply_approval_decision(ctx, request_id, decision)
        elif args.tui_bridge == "stream-submit":
            prompt = (args.bridge_prompt or "").strip()
            if not prompt:
                _bridge_emit({"kind": "final", "ok": False, "error": "Prompt cannot be empty.", "snapshot": _tui_bridge_snapshot(ctx)})
                return 1
            _bridge_emit({"kind": "submitted", "prompt": prompt, "snapshot": _tui_bridge_snapshot(ctx)})
            try:
                answer = _handle_local_command(ctx, prompt)
                if answer is None:
                    answer = handle_prompt(
                        ctx,
                        prompt,
                        stream_event=lambda event: _tui_bridge_stream_event(ctx, event),
                        approval_requester=lambda request: _tui_bridge_wait_for_approval(ctx, request),
                    )
                else:
                    logged_prompt = _redact_local_command(prompt)
                    ctx.store.update_last_prompt(ctx.session_id, logged_prompt)
                    ctx.store.add_message(ctx.session_id, "user", logged_prompt)
                    ctx.store.add_message(ctx.session_id, "assistant", answer)
                _bridge_emit({
                    "kind": "final",
                    "ok": True,
                    "answer": answer,
                    "snapshot": _tui_bridge_snapshot(ctx),
                })
                return 0
            except Exception as exc:
                _bridge_emit({
                    "kind": "final",
                    "ok": False,
                    "error": str(exc),
                    "snapshot": _tui_bridge_snapshot(ctx),
                })
                return 1
        else:
            prompt = (args.bridge_prompt or "").strip()
            if not prompt:
                payload = {"ok": False, "error": "Prompt cannot be empty.", "snapshot": _tui_bridge_snapshot(ctx)}
            else:
                try:
                    answer = _handle_local_command(ctx, prompt)
                    if answer is None:
                        answer = handle_prompt(ctx, prompt)
                    else:
                        logged_prompt = _redact_local_command(prompt)
                        ctx.store.update_last_prompt(ctx.session_id, logged_prompt)
                        ctx.store.add_message(ctx.session_id, "user", logged_prompt)
                        ctx.store.add_message(ctx.session_id, "assistant", answer)
                    payload = {
                        "ok": True,
                        "answer": answer,
                        "snapshot": _tui_bridge_snapshot(ctx),
                    }
                except Exception as exc:
                    payload = {"ok": False, "error": str(exc), "snapshot": _tui_bridge_snapshot(ctx)}
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    finally:
        try:
            if ctx is not None:
                _stop_language_servers(ctx)
        except Exception:
            pass


def _tui_bridge_snapshot(ctx: AppContext) -> dict[str, Any]:
    session = ctx.store.get_session(ctx.session_id)
    messages = ctx.store.list_messages(ctx.session_id, limit=None)
    return {
        "session": {
            "id": session.id,
            "title": session.title,
            "workspace_root": session.workspace_root,
            "updated_at": session.updated_at,
            "provider": _active_provider(ctx),
            "model": ctx.llm.model,
            "mode": _llm_mode(ctx),
            "configuration": _llm_configuration(ctx),
            "cost_usd": session.cost_usd,
            "tokens": {
                "input": session.tokens.input,
                "output": session.tokens.output,
                "reasoning": session.tokens.reasoning,
                "cache_read": session.tokens.cache_read,
                "cache_write": session.tokens.cache_write,
            },
        },
        "approvals": [
            dict(item)
            for item in ctx.session.pending_approvals
            if isinstance(item, dict) and item.get("status") == "pending"
        ],
        "messages": [
            {
                "role": item.role,
                "content": item.content,
                "created_at": item.created_at,
            }
            for item in messages
        ],
    }


def _tui_bridge_completions(prompt: str) -> dict[str, Any]:
    entries = _slash_palette_entries(prompt)
    return {
        "title": _slash_palette_title(prompt) if entries else "",
        "entries": [
            {
                "value": entry.value,
                "label": entry.label,
                "description": entry.description,
                "complete_to": entry.complete_to,
                "execute": entry.execute,
            }
            for entry in entries[:12]
        ],
    }


def _bridge_emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _tui_bridge_stream_event(ctx: AppContext, event: dict[str, Any]) -> None:
    persist_agent_state(ctx)
    _bridge_emit({"kind": "stream_event", "event": event, "snapshot": _tui_bridge_snapshot(ctx)})


def _tui_bridge_wait_for_approval(ctx: AppContext, request: dict[str, Any], *, poll_interval: float = 0.1, timeout: float = 1800.0) -> str:
    request_id = str(request.get("id") or "")
    if not request_id:
        return "denied"

    persist_agent_state(ctx)
    deadline = time.time() + timeout
    while time.time() < deadline:
        session_info = ctx.store.get_session(ctx.session_id)
        refreshed = agent_session_from_dict(session_info.state)
        ctx.session.pending_approvals = refreshed.pending_approvals
        for item in ctx.session.pending_approvals:
            if not isinstance(item, dict) or item.get("id") != request_id:
                continue
            status = str(item.get("status") or "pending")
            decision = str(item.get("decision") or "")
            if status != "pending" and decision in {"approved", "denied"}:
                persist_agent_state(ctx)
                return decision
        time.sleep(poll_interval)
    return "denied"


def _tui_bridge_apply_approval_decision(ctx: AppContext, request_id: str, decision: str) -> dict[str, Any]:
    matched = False
    for item in ctx.session.pending_approvals:
        if not isinstance(item, dict) or item.get("id") != request_id:
            continue
        item["status"] = decision
        item["decision"] = decision
        item["decision_at"] = datetime.now(timezone.utc).isoformat()
        matched = True
        break

    if not matched:
        session_info = ctx.store.get_session(ctx.session_id)
        refreshed = agent_session_from_dict(session_info.state)
        ctx.session.pending_approvals = refreshed.pending_approvals
        for item in ctx.session.pending_approvals:
            if not isinstance(item, dict) or item.get("id") != request_id:
                continue
            item["status"] = decision
            item["decision"] = decision
            item["decision_at"] = datetime.now(timezone.utc).isoformat()
            matched = True
            break

    if not matched:
        return {"ok": False, "error": f"Approval request not found: {request_id}", "snapshot": _tui_bridge_snapshot(ctx)}

    persist_agent_state(ctx)
    return {"ok": True, "snapshot": _tui_bridge_snapshot(ctx)}


def _stop_language_servers(ctx: Any) -> None:
    manager = getattr(ctx, "language_servers", None)
    stop_all = getattr(manager, "stop_all", None)
    if callable(stop_all):
        stop_all()


def _run_tui(stdscr: Any, ctx: AppContext) -> None:
    curses.curs_set(1)
    stdscr.keypad(True)
    stdscr.nodelay(True)
    _setup_tui_colors(stdscr)

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
    secret_provider: str | None = None
    secret_value = ""

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
        palette_entries = [] if secret_provider else _slash_palette_entries(prompt)
        if palette_entries:
            palette_index = min(palette_index, len(palette_entries) - 1)
        else:
            palette_index = 0
        command_palette = _slash_command_lines(prompt, max(20, content_width - 2), selected_index=palette_index)
        palette_height = min(9, len(command_palette))

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
            auth_active=secret_provider is not None,
        )
        _draw_input_line(
            stdscr,
            input_y,
            width,
            "*" * len(secret_value) if secret_provider else prompt,
            label=f" {secret_provider} key> " if secret_provider else " nym> ",
        )
        stdscr.refresh()

        key = stdscr.getch()
        if key == -1:
            time.sleep(0.05)
            continue

        if key == 3:  # Ctrl+C
            if secret_provider:
                secret_provider = None
                secret_value = ""
                status = "API key entry cancelled"
                continue
            if worker is not None and worker.is_alive():
                canceled = ctx.rust.cancel_active()
                status = "Cancelling..." if canceled else "Cancellation requested"
                continue
            break

        if key == 17:  # Ctrl+Q
            break

        if key == 15:  # Ctrl+O
            if secret_provider:
                login_answer = _login_provider(ctx, secret_provider)
                status = truncate(login_answer, 80)
            continue

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
            if secret_provider:
                provider = secret_provider
                api_key = secret_value.strip()
                secret_provider = None
                secret_value = ""
                if not api_key:
                    status = "API key entry cancelled"
                    continue
                local_answer = _set_provider_api_key(ctx, provider, api_key)
                logged_candidate = f"/apikey {provider} <redacted>"
                prompt_history.append(logged_candidate)
                history_index = None
                transcript_at_bottom = True
                transcript_scroll = 0
                ctx.store.update_last_prompt(ctx.session_id, logged_candidate)
                ctx.store.add_message(ctx.session_id, "user", logged_candidate)
                ctx.store.add_message(ctx.session_id, "assistant", local_answer)
                status = truncate(local_answer, 80)
                continue

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
                prompt = ""
                transcript_at_bottom = True
                transcript_scroll = 0
                api_key_provider = _api_key_prompt_provider(candidate)
                if api_key_provider:
                    secret_provider = api_key_provider
                    secret_value = ""
                    prompt_history.append(candidate)
                    history_index = None
                    status = f"Paste {api_key_provider} API key. Input is hidden."
                    continue
                local_answer = _handle_local_command(ctx, candidate)
                if local_answer is not None:
                    logged_candidate = _redact_local_command(candidate)
                    prompt_history.append(logged_candidate)
                    history_index = None
                    ctx.store.update_last_prompt(ctx.session_id, logged_candidate)
                    ctx.store.add_message(ctx.session_id, "user", logged_candidate)
                    ctx.store.add_message(ctx.session_id, "assistant", local_answer)
                    auth_provider = _provider_api_key_needed(ctx)
                    if auth_provider and _starts_auth_candidate(candidate):
                        secret_provider = auth_provider
                        secret_value = ""
                        status = f"{auth_provider} needs a key. Paste it here; Ctrl+O opens account/API keys."
                    else:
                        status = truncate(local_answer, 80)
                else:
                    prompt_history.append(candidate)
                    history_index = None
                    launch_turn(candidate)
            continue

        if key in (curses.KEY_BACKSPACE, 127, 8):
            if secret_provider:
                secret_value = secret_value[:-1]
            else:
                prompt = prompt[:-1]
            palette_index = 0
            continue

        if key == 9:  # Tab
            if secret_provider:
                continue
            completed = _complete_slash_command(prompt)
            if completed is not None:
                prompt = completed
                palette_index = 0
            continue

        if key == curses.KEY_UP:
            if secret_provider:
                continue
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
            if secret_provider:
                continue
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
            if secret_provider:
                secret_value += chr(key)
            else:
                char = chr(key)
                prompt += "/" if char == "\\" and not prompt else char
            palette_index = 0
            continue


def _setup_tui_colors(stdscr: Any | None = None) -> None:
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    if not _has_tui_colors():
        return
    curses.start_color()
    theme = os.environ.get("NYM_TUI_THEME", "").strip().casefold()
    light_theme = theme == "light"
    background = curses.COLOR_WHITE if light_theme else -1
    text = curses.COLOR_BLACK if light_theme else curses.COLOR_WHITE
    muted = curses.COLOR_BLUE if light_theme else curses.COLOR_WHITE
    pairs = [
        (COLOR_HEADER, curses.COLOR_BLUE if light_theme else curses.COLOR_CYAN, background),
        (COLOR_USER, text, background),
        (COLOR_ASSISTANT, text, background),
        (COLOR_THINKING, curses.COLOR_MAGENTA if light_theme else curses.COLOR_YELLOW, background),
        (COLOR_GUARDRAIL, curses.COLOR_MAGENTA if light_theme else curses.COLOR_YELLOW, background),
        (COLOR_MUTED, muted, background),
        (COLOR_ERROR, curses.COLOR_RED, background),
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
        f"session {ctx.session_id}  source {_model_source_label(_active_provider(ctx))}  model {ctx.llm.model}  "
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
    visible_count = 8
    selected_index = min(max(0, selected_index), len(entries) - 1)
    start = min(
        max(0, selected_index - visible_count + 1),
        max(0, len(entries) - visible_count),
    )
    for index, entry in enumerate(entries[start : start + visible_count], start=start):
        marker = ">" if index == selected_index else " "
        lines.append(_clip_line(f"{marker} {entry.label:<16} {entry.description}", width))
    return lines


def _slash_palette_entries(prompt: str) -> list[PaletteEntry]:
    prompt = _normalized_command_prompt(prompt)
    if not prompt.startswith("/"):
        return []

    stripped = prompt.strip()
    parts = stripped.split()
    provider_command = _provider_argument_command(prompt, parts)
    if provider_command is not None:
        return _provider_palette_entries(provider_command, parts[1] if len(parts) >= 2 else "")
    if _is_model_palette_prompt(prompt, parts):
        return _model_palette_entries(parts[1] if len(parts) >= 2 else "")

    query = parts[0].casefold() if parts else "/"
    matches = [
        PaletteEntry(
            value=name,
            label=name,
            description=description,
            complete_to=_slash_command_complete_to(name),
            execute=_slash_command_executes(name),
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
                complete_to=_slash_command_complete_to(name),
                execute=_slash_command_executes(name),
            )
            for name, description in LOCAL_COMMANDS
        ]
    return matches


def _slash_command_complete_to(name: str) -> str:
    if name in PROVIDER_ARGUMENT_COMMANDS | {"/model"}:
        return f"{name} "
    return name


def _slash_command_executes(name: str) -> bool:
    return name not in PROVIDER_ARGUMENT_COMMANDS | {"/model"}


def _provider_palette_entries(command: str, query: str) -> list[PaletteEntry]:
    normalized = query.casefold()
    providers = sorted(
        SUPPORTED_PROVIDERS,
        key=lambda item: PROVIDER_SORT_ORDER.get(item, 99),
    )
    matches = [provider for provider in providers if provider.casefold().startswith(normalized)]
    if not matches:
        matches = providers
    return [
        PaletteEntry(
            value=provider,
            label=_model_source_label(provider),
            description=_provider_palette_description(command, provider),
            complete_to=f"{command} {provider}",
            execute=True,
        )
        for provider in matches
    ]


def _provider_palette_description(command: str, provider: str) -> str:
    if command in {"/apikey", "/key"}:
        env_name = PROVIDER_API_KEY_ENVS.get(provider)
        return f"load {env_name}" if env_name else "no API key"
    if command in {"/login", "/auth"}:
        return "open account/API keys" if provider in PROVIDER_LOGIN_URLS else "local app"
    return f"default model: {PROVIDER_MODEL_HINTS.get(provider, ('custom-model',))[0]}"


def _model_palette_entries(query: str) -> list[PaletteEntry]:
    normalized = query.casefold()
    options = _model_options()
    matches = [
        option for option in options
        if option["model"].casefold().startswith(normalized)
        or option["provider"].casefold().startswith(normalized)
    ]
    if not matches:
        matches = options
    return [
        PaletteEntry(
            value=f"{option['provider']}/{option['model']}",
            label=option["model"],
            description=f"{_model_source_label(option['provider'])}: {_provider_access_label(option['provider'])}",
            complete_to=f"/model {option['provider']} {option['model']}",
            execute=True,
        )
        for option in matches
    ]


def _slash_palette_title(prompt: str) -> str:
    prompt = _normalized_command_prompt(prompt)
    stripped = prompt.strip()
    parts = stripped.split()
    if _provider_argument_command(prompt, parts) is not None:
        return "Model sources"
    if _is_model_palette_prompt(prompt, parts):
        return "Models"
    return "Commands"


def _provider_argument_command(prompt: str, parts: list[str]) -> str | None:
    if not parts:
        return None
    command = parts[0].casefold()
    if command not in PROVIDER_ARGUMENT_COMMANDS or len(parts) > 2:
        return None
    if prompt.startswith(f"{command} ") or prompt.endswith(" "):
        return command
    return None


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
    prompt = _normalized_command_prompt(prompt)
    selected = _selected_palette_entry(prompt, 0)
    parts = prompt.strip().split()
    if selected is not None and (_provider_argument_command(prompt, parts) is not None or prompt.startswith("/model ")):
        return selected.complete_to
    if not prompt.startswith("/") or " " in prompt:
        return None
    query = prompt.casefold()
    matches = [name for name, _description in LOCAL_COMMANDS if name.casefold().startswith(query)]
    if len(matches) == 1:
        return f"{matches[0]} "
    return None


def _normalized_command_prompt(prompt: str) -> str:
    if prompt.startswith("\\"):
        return f"/{prompt[1:]}"
    return prompt


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
    if line.startswith("Source") or line.startswith("Provider") or line.startswith("Model"):
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
        f"Source     {_model_source_label(provider)}",
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
    auth_active: bool = False,
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
    help_text = (
        "Enter save key  Ctrl+O open account  Ctrl+C cancel"
        if auth_active
        else "Ctrl+C cancel/exit  PgUp/PgDn scroll  Ctrl+A approve  Ctrl+D deny"
    )
    gap = max(1, width - len(footer) - len(help_text) - 1)
    line = f"{footer}{' ' * gap}{help_text}"
    attr = _tui_attr(COLOR_ERROR, curses.A_BOLD) if error else _tui_attr(COLOR_MUTED)
    stdscr.addnstr(y, 0, line.ljust(width), width - 1, attr)


def _draw_input_line(
    stdscr: Any,
    y: int,
    width: int,
    prompt: str,
    *,
    label: str = " nym> ",
) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
