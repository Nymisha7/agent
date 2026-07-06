from __future__ import annotations

import json
import re
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .language_servers import LanguageServerManager
from .llm import LLMClient
from .policy import PolicyEngine
from .prompt_loader import load_system_prompt
from .project_identity import identity_text, resolve_workspace_alias
from .rust_tools import RustTools
from .tools import ToolContext, build_tool_registry


@dataclass
class ModelToolCall:
    name: str
    call_id: str
    arguments: dict[str, Any]


@dataclass
class AgentSession:
    active_root: str | None = None
    focus_paths: list[str] = field(default_factory=list)
    last_candidates: list[dict[str, str]] = field(default_factory=list)
    recent_files: list[dict[str, str]] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    approved_external_read_roots: list[str] = field(default_factory=list)
    approved_external_write_roots: list[str] = field(default_factory=list)
    approved_external_delete_roots: list[str] = field(default_factory=list)
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RequestFrame:
    intent: str
    named_scope: str | None
    resolved_scope: Path | None
    requires_target_resolution: bool


def run_agent(
    *,
    llm: LLMClient,
    rust: RustTools,
    workspace_root: str,
    search_roots: list[str] | None = None,
    user_prompt: str,
    session: AgentSession | None = None,
    stored_context: str | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    record_event: Callable[..., None] | None = None,
    stream_event: Callable[[dict[str, Any]], None] | None = None,
    approval_requester: Callable[[dict[str, Any]], str] | None = None,
    language_servers: LanguageServerManager | None = None,
    max_steps: int = 20,
    debug: bool = False,
) -> str:
    effective_search_roots = search_roots if search_roots is not None else []
    system_prompt = load_system_prompt()
    active_session = session or AgentSession()
    tool_ctx = ToolContext(
        rust=rust,
        workspace_root=Path(workspace_root),
        search_roots=[Path(root) for root in effective_search_roots],
        approved_external_read_roots=[Path(root) for root in active_session.approved_external_read_roots],
        approved_external_write_roots=[Path(root) for root in active_session.approved_external_write_roots],
        approved_external_delete_roots=[Path(root) for root in active_session.approved_external_delete_roots],
        language_servers=language_servers,
    )
    tool_registry = build_tool_registry(tool_ctx)
    tools = tool_registry.schemas()
    policy = PolicyEngine()

    context_text = stored_context.strip() if stored_context else ""
    msg_history = list(_build_initial_messages(
        workspace_root=workspace_root,
        context_text=context_text,
        session=active_session,
        user_prompt=user_prompt,
        conversation_history=conversation_history,
    ))
    require_tool_use = _requires_tool_use(user_prompt)

    for step in range(max_steps):
        response = llm.respond(
            instructions=system_prompt,
            messages=msg_history,
            tools=tools,
            previous_response_id=None,
            tool_choice="required" if step == 0 and require_tool_use else None,
            stream=stream_event is not None,
            event_handler=(
                lambda event: _handle_stream_event(event, stream_event, debug=debug)
            )
            if stream_event is not None
            else None,
        )
        tool_calls = _tool_calls(response)
        if not tool_calls:
            return policy.redact_text(_response_text(response))

        tool_outputs: list[dict[str, Any]] = []
        for call in tool_calls:
            if debug:
                _debug_event("tool-call", {"name": call.name, "arguments": call.arguments})
            observation = _preflight_tool_call(
                call,
                tool_ctx=tool_ctx,
                session=session,
                user_prompt=user_prompt,
            )
            if observation is None:
                try:
                    observation = tool_registry.execute(call.name, call.arguments, tool_ctx)
                except Exception as exc:
                    observation = {
                        "ok": False,
                        "tool": call.name,
                        "args": call.arguments,
                        "error": str(exc),
                    }
            approval_request = _approval_request_from_observation(
                call,
                observation,
                user_prompt=user_prompt,
                workspace_root=Path(workspace_root),
            )
            if approval_request is not None:
                _record_pending_approval(active_session, approval_request)
                if record_event:
                    record_event(
                        event_type="approval_requested",
                        tool=call.name,
                        summary=_summarize_approval_request(approval_request),
                        path=approval_request.get("requested_path") or approval_request.get("translated_path"),
                        data={"request": approval_request},
                    )
                if stream_event:
                    stream_event({
                        "kind": "approval_request",
                        "tool": call.name,
                        "summary": _summarize_approval_request(approval_request),
                        "request": approval_request,
                    })
                if approval_requester is not None:
                    decision = approval_requester(approval_request)
                    normalized_decision = (decision or "").strip().casefold()
                    if normalized_decision == "approved":
                        _apply_approval(active_session, tool_ctx, approval_request)
                        if record_event:
                            record_event(
                                event_type="approval_decided",
                                tool=call.name,
                                summary=_summarize_approval_decision(approval_request, "approved"),
                                path=approval_request.get("requested_path") or approval_request.get("translated_path"),
                                data={"request": approval_request, "decision": "approved"},
                            )
                        if stream_event:
                            stream_event({
                                "kind": "approval_decision",
                                "tool": call.name,
                                "approved": True,
                                "summary": _summarize_approval_decision(approval_request, "approved"),
                                "request": approval_request,
                            })
                        observation = tool_registry.execute(call.name, call.arguments, tool_ctx)
                    else:
                        observation = _approval_denied_observation(call.name, call.arguments, approval_request)
                        if record_event:
                            record_event(
                                event_type="approval_decided",
                                tool=call.name,
                                summary=_summarize_approval_decision(approval_request, "denied"),
                                path=approval_request.get("requested_path") or approval_request.get("translated_path"),
                                data={"request": approval_request, "decision": "denied"},
                            )
                        if stream_event:
                            stream_event({
                                "kind": "approval_decision",
                                "tool": call.name,
                                "approved": False,
                                "summary": _summarize_approval_decision(approval_request, "denied"),
                                "request": approval_request,
                            })
            observation = _annotate_tool_observation(
                call.name,
                call.arguments,
                observation,
            )
            sanitized_observation = policy.sanitize_observation(observation)
            if debug:
                _debug_event("tool-result", {"name": call.name, "observation": sanitized_observation})
            if session is not None:
                _update_session_from_tool_result(
                    session,
                    tool=call.name,
                    args=call.arguments,
                    observation=observation,
                    workspace_root=Path(workspace_root),
                    user_prompt=user_prompt,
                )
            if record_event:
                record_event(
                    event_type="tool_result",
                    tool=call.name,
                    summary=_summarize_tool_result(call.name, call.arguments, sanitized_observation),
                    path=_extract_path(sanitized_observation, call.arguments),
                    data={
                        "args": call.arguments,
                        "observation": sanitized_observation,
                    },
                )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": _prepare_tool_output(sanitized_observation),
                }
            )

        msg_history.extend(_response_output_items(response))
        msg_history.extend(tool_outputs)

    final_response = llm.respond(
        instructions=system_prompt,
        messages=[
            *msg_history,
            {
                "role": "user",
                "content": (
                    "Stop using tools now. Answer the user's request from the evidence "
                    "already gathered. Be explicit about any gaps caused by the tool "
                    "budget being exhausted."
                ),
            },
        ],
        tools=[],
        previous_response_id=None,
    )
    return policy.redact_text(_response_text(final_response))


def _build_initial_messages(
    *,
    workspace_root: str,
    context_text: str,
    session: AgentSession,
    user_prompt: str,
    conversation_history: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    session_context = _session_context_text(session)
    workspace_identity = identity_text(Path(workspace_root))
    if conversation_history:
        parts = [f"Workspace root: {workspace_root}"]
        if workspace_identity:
            parts += ["", workspace_identity]
        if context_text:
            parts += ["", context_text]
        if session_context:
            parts += ["", session_context]
        parts += ["", "Resumed conversation history follows."]
        return (
            [{"role": "user", "content": "\n".join(parts)}]
            + _normalize_history(conversation_history)
        )

    parts = [f"Workspace root: {workspace_root}"]
    if workspace_identity:
        parts += ["", workspace_identity]
    if context_text:
        parts += ["", context_text]
    if session_context:
        parts += ["", session_context]
    parts += ["", f"User request: {user_prompt}"]
    return [{"role": "user", "content": "\n".join(parts)}]


_WORKSPACE_EVIDENCE_RE = re.compile(
    r"\b("
    r"project|repo|repository|codebase|workspace|app|application|"
    r"architecture|structure|overview|summari[sz]e|explain|walk\s+through|"
    r"file|files|folder|directory|module|package|component|entry\s*point"
    r")\b",
    re.IGNORECASE,
)

_LOCAL_REFERENCE_RE = re.compile(
    r"\b("
    r"this|these|here|local|current|existing|the|my|our|about|inside|under"
    r")\b",
    re.IGNORECASE,
)

_PATHLIKE_RE = re.compile(
    r"(^|[\s'\"`])"
    r"("
    r"\.?\.?/[\w./-]+|"
    r"~/?[\w./-]*|"
    r"[\w.-]+\.(py|js|jsx|ts|tsx|rs|go|java|rb|php|sh|md|toml|json|ya?ml|css|html|sql)"
    r")"
    r"($|[\s'\"`,:;?.!])",
    re.IGNORECASE,
)


def _requires_tool_use(user_prompt: str) -> bool:
    prompt = " ".join(user_prompt.split())
    if not prompt:
        return False

    create_patterns = [
        r"\b(create|create a|make|add|write|save|update|edit|modify|append|touch)\b",
        r"\b(file|files|script|module|config|note|notes)\b",
    ]
    if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in create_patterns):
        if re.search(r"\b(new|named|with content|contents?|into|to file|file named)\b", prompt, re.IGNORECASE):
            return True
        if re.search(r"\b(create|make|write|save|update|edit|modify|append|touch)\b", prompt, re.IGNORECASE):
            return True

    if _PATHLIKE_RE.search(prompt):
        return True
    if re.search(
        r"\b(language\s*server|lsp|pyright|clangd|jdtls|eclipse\.jdt\.ls|tsserver|typescript-language-server|gopls|rust-analyzer|go\s+to\s+definition|find\s+references|workspace\s+symbols?|document\s+symbols?)\b",
        prompt,
        re.IGNORECASE,
    ):
        return True
    if not _WORKSPACE_EVIDENCE_RE.search(prompt):
        return False
    return bool(_LOCAL_REFERENCE_RE.search(prompt))


def agent_session_from_dict(value: dict[str, Any] | None) -> AgentSession:
    if not isinstance(value, dict):
        return AgentSession()
    return AgentSession(
        active_root=_optional_str(value.get("active_root")),
        focus_paths=_string_list(value.get("focus_paths")),
        last_candidates=_candidate_dicts(value.get("last_candidates")),
        recent_files=_recent_file_dicts(value.get("recent_files")),
        pending_action=_pending_action_dict(value.get("pending_action")),
        approved_external_read_roots=_string_list(value.get("approved_external_read_roots")),
        approved_external_write_roots=_string_list(value.get("approved_external_write_roots")),
        approved_external_delete_roots=_string_list(value.get("approved_external_delete_roots")),
        pending_approvals=_pending_approval_dicts(value.get("pending_approvals")),
    )


def agent_session_to_dict(session: AgentSession) -> dict[str, Any]:
    return {
        "active_root": session.active_root,
        "focus_paths": session.focus_paths,
        "last_candidates": session.last_candidates,
        "recent_files": session.recent_files,
        "pending_action": session.pending_action,
        "approved_external_read_roots": session.approved_external_read_roots,
        "approved_external_write_roots": session.approved_external_write_roots,
        "approved_external_delete_roots": session.approved_external_delete_roots,
        "pending_approvals": session.pending_approvals,
    }


def _requires_workspace_evidence(user_prompt: str) -> bool:
    return _requires_tool_use(user_prompt)


def _is_edit_intent(user_prompt: str) -> bool:
    return bool(re.search(r"\b(edit|modify|change|update|append|add\s+to)\b", user_prompt, re.IGNORECASE))


def _is_create_intent(user_prompt: str) -> bool:
    return bool(re.search(r"\b(create|make|new\s+file|write\s+(?:a\s+)?(?:new\s+)?file|save\s+as)\b", user_prompt, re.IGNORECASE))


def _is_delete_intent(user_prompt: str) -> bool:
    return bool(re.search(r"\b(delete|remove|rm|trash|unlink)\b", user_prompt, re.IGNORECASE))


def _is_mutation_intent(user_prompt: str) -> bool:
    return _is_delete_intent(user_prompt) or _is_edit_intent(user_prompt) or _is_create_intent(user_prompt)


def _mutation_action(user_prompt: str) -> str:
    if _is_delete_intent(user_prompt):
        return "delete"
    if _is_edit_intent(user_prompt):
        return "edit"
    if _is_create_intent(user_prompt):
        return "create"
    return "mutate"


def _requested_action(user_prompt: str) -> str:
    if _is_mutation_intent(user_prompt):
        return _mutation_action(user_prompt)
    return "answer"


def _request_frame(user_prompt: str, workspace_root: Path | None = None) -> RequestFrame:
    named_scope = _named_project_scope(user_prompt, workspace_root=workspace_root)
    resolved_scope = None
    if workspace_root is not None and named_scope:
        resolved_scope = resolve_workspace_alias(named_scope, workspace_root)
    return RequestFrame(
        intent=_request_intent(user_prompt),
        named_scope=named_scope,
        resolved_scope=resolved_scope,
        requires_target_resolution=bool(named_scope and resolved_scope is None),
    )


def _request_intent(user_prompt: str) -> str:
    if re.search(r"\b(api\s*key|secret|secrets|credential|credentials|token|password|private\s*key)\b", user_prompt, re.IGNORECASE):
        return "secret_audit"
    if _is_delete_intent(user_prompt):
        return "delete"
    if _is_edit_intent(user_prompt):
        return "edit"
    if _is_create_intent(user_prompt):
        return "create"
    if re.search(r"\b(search|find|grep|look\s+for|locate|check)\b", user_prompt, re.IGNORECASE):
        return "search"
    if re.search(
        r"\b(tell\s+me\s+about|explain|summari[sz]e|(?:the\s+)?(?:overview|rundown)|details?\s+(?:of|on)|inside|list|show|inspect|walk\s+through)\b",
        user_prompt,
        re.IGNORECASE,
    ):
        return "inspect"
    return "answer"


def _path_from_args(args: dict[str, Any], workspace_root: Path) -> Path | None:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve(strict=False)


def _recent_file_matches(session: AgentSession, path: Path) -> list[dict[str, str]]:
    path_text = str(path)
    name = path.name
    matches: list[dict[str, str]] = []
    for item in session.recent_files:
        item_path = item.get("path")
        if not item_path:
            continue
        if item_path == path_text or Path(item_path).name == name:
            matches.append(item)
    return matches


def _pending_approval_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    approvals: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            approvals.append(dict(item))
    return approvals


def _record_pending_approval(session: AgentSession, request: dict[str, Any]) -> None:
    approval_id = _optional_str(request.get("id"))
    pending = [item for item in session.pending_approvals if _optional_str(item.get("id")) != approval_id]
    pending.append(dict(request))
    session.pending_approvals = pending


def _apply_approval(session: AgentSession, tool_ctx: ToolContext, request: dict[str, Any]) -> None:
    target_path = _approval_path(request)
    if not target_path:
        return
    operation = _optional_str(request.get("operation")) or "read"
    roots = {
        "read": session.approved_external_read_roots,
        "write": session.approved_external_write_roots,
        "delete": session.approved_external_delete_roots,
    }.get(operation, session.approved_external_read_roots)
    if target_path not in roots:
        roots.append(target_path)
    _sync_tool_context_approvals(tool_ctx, session)


def _sync_tool_context_approvals(tool_ctx: ToolContext, session: AgentSession) -> None:
    tool_ctx.approved_external_read_roots = [Path(path) for path in session.approved_external_read_roots]
    tool_ctx.approved_external_write_roots = [Path(path) for path in session.approved_external_write_roots]
    tool_ctx.approved_external_delete_roots = [Path(path) for path in session.approved_external_delete_roots]


def _approval_path(request: dict[str, Any]) -> str | None:
    for key in ("translated_path", "resolved_path", "requested_path"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _approval_request_from_observation(
    call: ModelToolCall,
    observation: Any,
    *,
    user_prompt: str,
    workspace_root: Path,
) -> dict[str, Any] | None:
    if not isinstance(observation, dict) or observation.get("blocked") is not True:
        return None
    if observation.get("recoverable") is not True:
        return None

    reason = _optional_str(observation.get("reason")) or ""
    if reason not in {
        "external_path_requires_approval",
        "external_delete_requires_confirmation",
        "external_windows_path_requires_approval",
    }:
        return None

    operation = _optional_str(observation.get("operation")) or _tool_operation(call.name)
    request = {
        "id": uuid.uuid4().hex,
        "status": "pending",
        "tool": call.name,
        "operation": operation,
        "reason": reason,
        "guidance": _optional_str(observation.get("guidance")) or "",
        "requested_path": _optional_str(observation.get("requested_path")) or _optional_str(call.arguments.get("path")),
        "resolved_path": _optional_str(observation.get("resolved_path")),
        "translated_path": _optional_str(observation.get("translated_path")),
        "broad_path": bool(observation.get("broad_path")),
        "prompt": user_prompt,
        "args": dict(call.arguments),
        "workspace_root": str(workspace_root),
    }
    if request["broad_path"] or not _approval_path(request):
        return None
    if request["reason"] == "external_windows_path_requires_approval" and request["translated_path"] is None:
        return None
    return request


def _tool_operation(tool: str) -> str:
    return {
        "read_path": "read",
        "list_path": "read",
        "inspect_target": "read",
        "inspect_tree": "read",
        "glob": "read",
        "grep": "read",
        "secret_scan": "read",
        "write_file": "write",
        "edit_file": "write",
        "delete_path": "delete",
    }.get(tool, "read")


def _summarize_approval_request(request: dict[str, Any]) -> str:
    tool = _optional_str(request.get("tool")) or "tool"
    path = _approval_path(request) or "target"
    reason = _optional_str(request.get("reason")) or "approval_required"
    return f"Approval required for {tool} on {path} ({reason})"


def _summarize_approval_decision(request: dict[str, Any], decision: str) -> str:
    tool = _optional_str(request.get("tool")) or "tool"
    path = _approval_path(request) or "target"
    return f"{decision.title()} {tool} on {path}"


def _approval_denied_observation(
    tool: str,
    args: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool,
        "args": args,
        "blocked": True,
        "recoverable": True,
        "reason": "approval_denied",
        "operation": request.get("operation"),
        "requested_path": request.get("requested_path"),
        "resolved_path": request.get("resolved_path"),
        "translated_path": request.get("translated_path"),
        "guidance": "The user denied access. Stop and ask for a different path or a narrower target.",
    }


def _preflight_tool_call(
    call: ModelToolCall,
    *,
    tool_ctx: ToolContext,
    session: AgentSession | None,
    user_prompt: str,
) -> Any | None:
    pending_block = _preflight_pending_action(call, tool_ctx=tool_ctx, session=session, user_prompt=user_prompt)
    if pending_block is not None:
        return pending_block

    scope_block = _preflight_named_scope(call, tool_ctx=tool_ctx, user_prompt=user_prompt)
    if scope_block is not None:
        return scope_block

    if call.name == "write_file" and _is_edit_intent(user_prompt) and not _is_create_intent(user_prompt):
        path = _path_from_args(call.arguments, tool_ctx.workspace_root)
        if path is not None and not path.exists():
            return {
                "ok": False,
                "tool": call.name,
                "args": call.arguments,
                "blocked": True,
                "reason": "edit_intent_would_create_missing_file",
                "path": str(path),
                "guidance": (
                    "The user asked to edit or modify an existing file, but write_file would create "
                    "a new file at this path. Locate and read the existing target first with glob "
                    "or read_path, then use edit_file. Ask before creating a new file if no target exists."
                ),
            }

    if call.name == "delete_path" and session is not None:
        path = _path_from_args(call.arguments, tool_ctx.workspace_root)
        if path is not None and not path.exists():
            matches = _recent_file_matches(session, path)
            deleted = [item for item in matches if item.get("status") == "deleted"]
            existing = [item for item in matches if item.get("status") != "deleted"]
            if deleted:
                return {
                    "path": deleted[0]["path"],
                    "deleted": False,
                    "already_absent": True,
                    "kind": "file",
                    "note": (
                        "This file was already deleted earlier in the session. "
                        "Do not issue another delete for the same file."
                    ),
                    "recent_matches": deleted,
                }
            if existing:
                return {
                    "ok": False,
                    "tool": call.name,
                    "args": call.arguments,
                    "blocked": True,
                    "recoverable": True,
                    "reason": "delete_path_missed_recent_file",
                    "failed_path": str(path),
                    "recent_matches": existing,
                    "guidance": (
                        "The attempted delete path does not exist, but a recently modified file "
                        "with the same name exists. Use the recent file operation path instead "
                        "of guessing a sibling path."
                    ),
                }

    return None


def _preflight_named_scope(
    call: ModelToolCall,
    *,
    tool_ctx: ToolContext,
    user_prompt: str,
) -> Any | None:
    frame = _request_frame(user_prompt, tool_ctx.workspace_root)
    if not frame.requires_target_resolution or not frame.named_scope:
        return None
    if call.name == "inspect_target":
        return None
    if call.name not in {"inspect_tree", "list_path", "read_path", "grep", "glob"}:
        return None
    if not _call_targets_broad_scope(call, tool_ctx.workspace_root):
        return None
    return {
        "ok": False,
        "tool": call.name,
        "args": call.arguments,
        "blocked": True,
        "recoverable": True,
        "reason": "named_project_scope_not_resolved",
        "intent": frame.intent,
        "named_scope": frame.named_scope,
        "guidance": (
            "The user named a narrower project/folder scope, but this tool call targets a "
            "broad root. Resolve the named scope first with inspect_target using kind=directory, "
            "then retry inside exactly one resolved target."
        ),
    }


def _call_targets_broad_scope(call: ModelToolCall, workspace_root: Path) -> bool:
    raw_path = call.arguments.get("path")
    if raw_path is None and call.name in {"glob", "grep"}:
        return True
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    target = path.resolve(strict=False)

    roots = {
        workspace_root.resolve(strict=False),
        Path.home().resolve(strict=False),
    }
    return target in roots


def _named_project_scope(user_prompt: str, *, workspace_root: Path | None = None) -> str | None:
    patterns = [
        r"\b(?:inside|under|within)\s+(?:my\s+|the\s+|a\s+)?([A-Za-z][\w.-]*)\b",
        r"\babout\s+(?:my\s+|the\s+|a\s+)?([A-Za-z][\w.-]*)\b",
        r"\b(?:rundown|overview|details?)\s+(?:of|on)\s+(?:my\s+|the\s+|a\s+)?([A-Za-z][\w.-]*)\b",
        r"\b(?:from|in)\s+(?:my\s+|the\s+|a\s+)?([A-Za-z][\w.-]*)\s+(?:project|folder|directory|repo|app)\b",
        r"\b([A-Za-z][\w.-]*)\s+(?:project|folder|directory|repo|app)\b",
        r"\b(?:from|in)\s+(?:my\s+|the\s+|a\s+)?([A-Za-z][\w.-]*)\b",
    ]
    aliases = set()
    if workspace_root is not None:
        aliases = {alias.casefold() for alias in _workspace_aliases(workspace_root)}
    for pattern in patterns:
        match = re.search(pattern, user_prompt, re.IGNORECASE)
        if match:
            token = match.group(1)
            token_norm = token.casefold()
            if token_norm in aliases:
                return token
            if token_norm not in {
                "my",
                "the",
                "a",
                "this",
                "that",
                "current",
                "workspace",
                "repo",
                "repository",
                "project",
                "folder",
                "directory",
                "home",
                "root",
                "file",
                "files",
                "it",
                "those",
                "there",
            }:
                return token
    return None


def _workspace_aliases(workspace_root: Path) -> set[str]:
    aliases = {workspace_root.name}
    try:
        aliases.add(workspace_root.resolve(strict=False).name)
    except OSError:
        pass
    return {alias for alias in aliases if alias}


def _preflight_pending_action(
    call: ModelToolCall,
    *,
    tool_ctx: ToolContext,
    session: AgentSession | None,
    user_prompt: str,
) -> Any | None:
    if session is None or not isinstance(session.pending_action, dict):
        return None

    pending = session.pending_action
    if pending.get("status") != "unresolved":
        return None

    if _names_new_scope(user_prompt, pending, tool_ctx.workspace_root):
        session.pending_action = None
        return None

    selected = _selected_pending_candidate(pending, user_prompt)
    if selected is None:
        return {
            "ok": False,
            "tool": call.name,
            "args": call.arguments,
            "blocked": True,
            "reason": "pending_target_clarification_unresolved",
            "pending_action": pending,
            "guidance": (
                "A previous request is blocked on multiple possible targets. "
                "The latest user message did not select exactly one candidate, so do not "
                "continue the request or search all candidates. Answer the current message "
                "or ask for a single target selection."
            ),
        }

    path = _path_from_args(call.arguments, tool_ctx.workspace_root)
    if path is not None and not _path_within_pending_candidate(path, selected, tool_ctx.workspace_root):
        return {
            "ok": False,
            "tool": call.name,
            "args": call.arguments,
            "blocked": True,
            "reason": "tool_path_outside_selected_candidate",
            "selected_candidate": selected,
            "pending_action": pending,
            "guidance": (
                "The user selected one pending candidate. Tool calls must stay inside that "
                "selected target; do not operate on sibling candidates."
            ),
        }

    pending["status"] = "resolved"
    pending["selected"] = selected
    session.focus_paths = [selected["path"]]
    return None


def _selected_pending_candidate(pending: dict[str, Any], user_prompt: str) -> dict[str, str] | None:
    candidates = _candidate_dicts(pending.get("candidates"))
    if not candidates:
        return None

    prompt = user_prompt.strip()
    if not prompt:
        return None

    if _looks_like_meta_question(prompt):
        return None

    ordinal = _ordinal_selection(prompt)
    if ordinal is not None:
        index = ordinal - 1
        if 0 <= index < len(candidates):
            return candidates[index]
        return None

    normalized_prompt = _normalize_selection_text(prompt)
    if not normalized_prompt:
        return None
    for candidate in candidates:
        path = candidate.get("path", "")
        if not path:
            continue
        names = {
            Path(path).name.casefold(),
            Path(path).as_posix().casefold(),
        }
        try:
            names.add(Path(path).resolve(strict=False).name.casefold())
        except OSError:
            pass
        if normalized_prompt in names:
            return candidate
    return None

def _names_new_scope(user_prompt: str, pending: dict[str, Any], workspace_root: Path) -> bool:
    scope = _named_project_scope(user_prompt, workspace_root=workspace_root)
    if not scope:
        return False
    if scope.casefold() == str(pending.get("query", "")).casefold():
        return False
    candidates = _candidate_dicts(pending.get("candidates"))
    for candidate in candidates:
        path = candidate.get("path", "")
        if Path(path).name.casefold() == scope.casefold():
            return False
    return True


def _looks_like_meta_question(prompt: str) -> bool:
    return (
        bool(re.search(r"\b(why|how|what happened|weren'?t|didn'?t|explain|which|where)\b", prompt, re.IGNORECASE))
        or "?" in prompt
        or bool(re.search(r"\b(assistant|user)\s+\d{4}-\d{2}-\d{2}t", prompt, re.IGNORECASE))
    )


def _ordinal_selection(prompt: str) -> int | None:
    stripped = prompt.strip().lower()
    match = re.fullmatch(r"(?:yes|yeah|yup|ok(?:ay)?|sure)?\s*#?(\d{1,2})\.?", stripped)
    if match:
        return int(match.group(1))
    words = {
        "one": 1,
        "first": 1,
        "two": 2,
        "second": 2,
        "three": 3,
        "third": 3,
        "four": 4,
        "fourth": 4,
        "five": 5,
        "fifth": 5,
    }
    return words.get(stripped)


def _normalize_selection_text(prompt: str) -> str:
    stripped = re.sub(r"^(yes|yeah|yup|ok(?:ay)?|sure)\b[\s,.:;-]*", "", prompt.strip(), flags=re.IGNORECASE)
    if re.search(r"\s", stripped):
        return ""
    return stripped.casefold()


def _path_within_pending_candidate(path: Path, candidate: dict[str, str], workspace_root: Path) -> bool:
    raw_candidate = candidate.get("path")
    if not raw_candidate:
        return False
    candidate_path = Path(raw_candidate)
    if not candidate_path.is_absolute():
        candidate_path = workspace_root / candidate_path
    try:
        path.resolve(strict=False).relative_to(candidate_path.resolve(strict=False))
        return True
    except ValueError:
        return False


def _annotate_tool_observation(tool: str, args: dict[str, Any], observation: Any) -> Any:
    if not isinstance(observation, dict) or observation.get("ok") is not False:
        return observation

    error = observation.get("error")
    if not isinstance(error, str):
        return observation

    missing_path = (
        "No such file or directory" in error
        or "Path does not exist" in error
        or "does not exist or cannot be inspected" in error
    )
    if not missing_path or tool not in {"read_path", "list_path", "inspect_tree", "grep", "edit_file", "delete_path"}:
        return observation

    annotated = dict(observation)
    annotated["recoverable"] = True
    annotated["guidance"] = (
        "The attempted path is missing, but it may be an incorrect guess. "
        "Do not conclude the user-requested target is absent from this failure alone. "
        "Resolve the named project/folder/file from the original request with inspect_target or glob, "
        "then retry against the resolved path or matching candidates."
    )
    path = args.get("path")
    if isinstance(path, str) and path:
        annotated["failed_path"] = path
    return annotated


def _session_context_text(session: AgentSession) -> str:
    lines: list[str] = []
    if session.active_root:
        lines.append(f"Active root: {session.active_root}")
    if session.focus_paths:
        lines.append("Focused paths:")
        lines.extend(f"- {path}" for path in session.focus_paths[:8])
    if session.last_candidates:
        lines.append("Last path candidates:")
        for candidate in session.last_candidates[:12]:
            path = candidate.get("path", "")
            kind = candidate.get("kind", "unknown")
            if path:
                lines.append(f"- {path} ({kind})")
    if session.recent_files:
        lines.append("Recent file operations:")
        for item in session.recent_files[:8]:
            path = item.get("path", "")
            action = item.get("action", "touched")
            status = item.get("status", "unknown")
            if path:
                lines.append(f"- {path} ({action}, {status})")
    if isinstance(session.pending_action, dict) and session.pending_action.get("status") == "unresolved":
        action = session.pending_action.get("action", "mutation")
        lines.append(f"Pending unresolved action: {action}")
        candidates = _candidate_dicts(session.pending_action.get("candidates"))
        if candidates:
            lines.append("Pending candidates:")
            for index, candidate in enumerate(candidates[:12], start=1):
                path = candidate.get("path", "")
                kind = candidate.get("kind", "unknown")
                if path:
                    lines.append(f"{index}. {path} ({kind})")
    pending_approvals = [
        item for item in session.pending_approvals
        if isinstance(item, dict) and item.get("status") == "pending"
    ]
    if pending_approvals:
        lines.append("Pending approvals:")
        for index, request in enumerate(pending_approvals[:8], start=1):
            tool = _optional_str(request.get("tool")) or "tool"
            path = _approval_path(request) or _optional_str(request.get("requested_path")) or ""
            reason = _optional_str(request.get("reason")) or "approval_required"
            if path:
                lines.append(f"{index}. {tool} on {path} ({reason})")
    if not lines:
        return ""
    return "\n".join([
        "Session path context:",
        *lines,
        "Use this context to resolve vague follow-ups like 'inside those', 'there', 'that file', or ordinal selections.",
        "For follow-ups about a recently created, edited, or deleted file, prefer the recent file operation path over guessing a sibling path.",
        "If a pending action is unresolved, do not continue it until the user selects exactly one pending candidate.",
        "If the user names a new explicit target or acronym, resolve that target directly instead of expanding it from this context.",
    ])


def _update_session_from_tool_result(
    session: AgentSession,
    *,
    tool: str,
    args: dict[str, Any],
    observation: Any,
    workspace_root: Path,
    user_prompt: str = "",
) -> None:
    if not isinstance(observation, dict) or observation.get("ok") is False:
        return

    if tool == "glob":
        _apply_candidates(session, _candidates_from_items(observation.get("matches")))
        _maybe_set_pending_action(session, tool=tool, observation=observation, user_prompt=user_prompt, workspace_root=workspace_root)
        return

    if tool == "inspect_target":
        _apply_inspect_target(session, observation)
        _maybe_set_pending_action(session, tool=tool, observation=observation, user_prompt=user_prompt, workspace_root=workspace_root)
        return

    if tool == "inspect_tree":
        _apply_inspect_tree(session, observation, workspace_root)
        return

    if tool in {"list_path", "read_path"}:
        _apply_read_path(session, observation)
        return

    if tool in {"write_file", "edit_file", "delete_path"}:
        path = _optional_str(observation.get("path")) or _optional_str(args.get("path"))
        if path:
            session.focus_paths = [path]
            _remember_file_operation(session, tool, path, observation)


def _apply_candidates(session: AgentSession, candidates: list[dict[str, str]]) -> None:
    if not candidates:
        return
    session.last_candidates = candidates[:20]
    session.focus_paths = [candidate["path"] for candidate in candidates[:8] if candidate.get("path")]
    if len(candidates) == 1 and candidates[0].get("kind") == "directory":
        session.active_root = candidates[0]["path"]


def _maybe_set_pending_action(
    session: AgentSession,
    *,
    tool: str,
    observation: dict[str, Any],
    user_prompt: str,
    workspace_root: Path,
) -> None:
    candidates = _pending_candidates(tool, observation, workspace_root)
    if len(candidates) <= 1:
        if session.pending_action and session.pending_action.get("status") == "unresolved":
            session.pending_action = None
        return

    session.pending_action = {
        "status": "unresolved",
        "action": _request_frame(user_prompt, workspace_root).intent,
        "reason": "multiple_target_candidates",
        "query": _optional_str(observation.get("query")) or "",
        "candidates": candidates[:12],
    }


def _pending_candidates(
    tool: str,
    observation: dict[str, Any],
    workspace_root: Path,
) -> list[dict[str, str]]:
    if tool == "inspect_target" and observation.get("status") == "candidates":
        candidates = _candidates_from_items(observation.get("candidates"))
        query = _optional_str(observation.get("query")) or ""
        return _project_level_candidates(candidates, query, workspace_root)
    if tool == "glob":
        candidates = _candidates_from_items(observation.get("matches"))
        return [candidate for candidate in candidates if candidate.get("kind") == "directory"] or candidates
    return []


def _project_level_candidates(
    candidates: list[dict[str, str]],
    query: str,
    workspace_root: Path,
) -> list[dict[str, str]]:
    query_norm = query.casefold()
    by_top_level: dict[str, dict[str, str]] = {}
    fallback: list[dict[str, str]] = []

    for candidate in candidates:
        raw_path = candidate.get("path")
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_absolute():
            try:
                rel = path.resolve(strict=False).relative_to(workspace_root.resolve(strict=False))
            except ValueError:
                rel = path
        else:
            rel = path
        parts = rel.parts
        if not parts:
            continue
        top = parts[0]
        if query_norm and not top.casefold().startswith(query_norm):
            fallback.append(candidate)
            continue
        kind = candidate.get("kind", "unknown")
        top_path = str(workspace_root / top)
        by_top_level.setdefault(top_path, {"path": top_path, "kind": kind})

    if by_top_level:
        return list(by_top_level.values())
    return fallback or candidates


def _apply_inspect_target(session: AgentSession, observation: dict[str, Any]) -> None:
    status = observation.get("status")
    if status == "candidates":
        _apply_candidates(session, _candidates_from_items(observation.get("candidates")))
        return
    if status == "resolved":
        target = observation.get("target")
        if isinstance(target, dict):
            path = _optional_str(target.get("path"))
            kind = _optional_str(target.get("kind")) or "unknown"
            if path:
                session.focus_paths = [path]
                session.last_candidates = [{"path": path, "kind": kind}]
                if kind == "directory":
                    session.active_root = path


def _apply_inspect_tree(session: AgentSession, observation: dict[str, Any], workspace_root: Path) -> None:
    root = _optional_str(observation.get("path"))
    if root:
        session.active_root = root
        session.focus_paths = [root]

    direct_items = observation.get("direct_children")
    if isinstance(direct_items, list):
        direct_children = []
        for item in direct_items:
            if not isinstance(item, dict):
                continue
            path = _optional_str(item.get("path"))
            kind = _optional_str(item.get("kind")) or "unknown"
            if path:
                direct_children.append({"path": _absolute_path_text(path, workspace_root), "kind": kind})
        if direct_children:
            _apply_candidates(session, direct_children)
            if root:
                session.active_root = root
            return

    tree = observation.get("tree")
    if not isinstance(tree, list):
        return

    root_rel = _relative_path_text(root, workspace_root) if root else ""
    direct_children: list[dict[str, str]] = []
    for item in tree:
        if not isinstance(item, dict):
            continue
        path = _optional_str(item.get("path"))
        kind = _optional_str(item.get("kind")) or "unknown"
        if not path:
            continue
        if root_rel and Path(path).parent.as_posix() != root_rel:
            continue
        direct_children.append({"path": _absolute_path_text(path, workspace_root), "kind": kind})

    if direct_children:
        _apply_candidates(session, direct_children)
        if root:
            session.active_root = root


def _apply_read_path(session: AgentSession, observation: dict[str, Any]) -> None:
    path = _optional_str(observation.get("path"))
    if not path:
        return
    detection = observation.get("detection")
    kind = ""
    if isinstance(detection, dict):
        kind = _optional_str(detection.get("kind")) or ""

    if kind == "directory":
        session.active_root = path
        session.focus_paths = [path]
        content = _optional_str(observation.get("content")) or ""
        candidates = []
        for line in content.splitlines():
            name = line.strip()
            if not name:
                continue
            is_dir = name.endswith("/")
            child = str(Path(path) / name.rstrip("/"))
            candidates.append({"path": child, "kind": "directory" if is_dir else "file"})
        _apply_candidates(session, candidates)
        session.active_root = path
    else:
        session.focus_paths = [path]
        session.last_candidates = [{"path": path, "kind": "file"}]


def _remember_file_operation(
    session: AgentSession,
    tool: str,
    path: str,
    observation: dict[str, Any],
) -> None:
    action = {
        "write_file": "write",
        "edit_file": "edit",
        "delete_path": "delete",
    }.get(tool, tool)
    status = "deleted" if tool == "delete_path" else "exists"

    record = {
        "path": path,
        "action": action,
        "status": status,
    }
    resource = _optional_str(observation.get("resource"))
    if resource:
        record["resource"] = resource
    created = observation.get("created")
    if isinstance(created, bool):
        record["created"] = "true" if created else "false"

    session.recent_files = [
        item for item in session.recent_files
        if item.get("path") != path
    ]
    session.recent_files.insert(0, record)
    del session.recent_files[12:]


def _candidates_from_items(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    candidates: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        path = _optional_str(item.get("path"))
        kind = _optional_str(item.get("kind")) or _optional_str(item.get("type")) or "unknown"
        if path:
            candidates.append({"path": path, "kind": kind})
    return candidates


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()][:20]


def _candidate_dicts(value: Any) -> list[dict[str, str]]:
    return _candidates_from_items(value)[:20]


def _recent_file_dicts(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = _optional_str(item.get("path"))
        if not path:
            continue
        record = {"path": path}
        for key in ("action", "status", "resource", "created"):
            text = _optional_str(item.get(key))
            if text:
                record[key] = text
        result.append(record)
    return result[:12]


def _pending_action_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    status = _optional_str(value.get("status"))
    action = _optional_str(value.get("action"))
    if not status or not action:
        return None
    result: dict[str, Any] = {
        "status": status,
        "action": action,
    }
    for key in ("reason", "query"):
        text = _optional_str(value.get(key))
        if text:
            result[key] = text
    candidates = _candidate_dicts(value.get("candidates"))
    if candidates:
        result["candidates"] = candidates
    selected = value.get("selected")
    if isinstance(selected, dict):
        selected_candidates = _candidate_dicts([selected])
        if selected_candidates:
            result["selected"] = selected_candidates[0]
    return result


def _relative_path_text(path: str | None, workspace_root: Path) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _absolute_path_text(path: str, workspace_root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(workspace_root / candidate)


def _normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    policy = PolicyEngine()
    result: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        result.append({"role": role, "content": policy.redact_text(content)})
    return result


def _prepare_tool_output(observation: Any, *, max_bytes: int = 12_000) -> str:
    payload = json.dumps(observation, ensure_ascii=False, default=str)
    if len(payload.encode("utf-8")) <= max_bytes:
        return payload

    if isinstance(observation, dict):
        compact = dict(observation)
        for key in list(compact.keys()):
            value = compact[key]
            if isinstance(value, str) and len(value.encode("utf-8")) > 2_000:
                compact[key] = f"<truncated {len(value)} chars>"
            elif isinstance(value, list) and len(value) > 20:
                compact[key] = value[:20] + [f"<truncated {len(value) - 20} more items>"]
            elif isinstance(value, dict):
                compact[key] = {k: v for k, v in list(value.items())[:8]}
        payload = json.dumps(compact, ensure_ascii=False, default=str)
        if len(payload.encode("utf-8")) <= max_bytes:
            return payload

    preview = payload[: max_bytes - 80]
    return f"{preview}...<truncated {len(payload.encode('utf-8'))} bytes>"


def _summarize_tool_result(tool: str, args: dict[str, Any], observation: Any) -> str:
    if isinstance(observation, dict):
        if observation.get("ok") is False:
            return f"{tool} failed: {observation.get('error')}"
        if "matches" in observation and isinstance(observation["matches"], list):
            return f"{tool} returned {len(observation['matches'])} matches"
        if "entries" in observation and isinstance(observation["entries"], list):
            return f"{tool} returned {len(observation['entries'])} entries"
        if "candidates" in observation and isinstance(observation["candidates"], list):
            return f"{tool} returned {len(observation['candidates'])} candidates"
        if "read_file_count" in observation and "file_count" in observation:
            path = observation.get("path") or args.get("path")
            return (
                f"{tool} inspected {path or 'target'}: "
                f"read {observation.get('read_file_count')} of "
                f"{observation.get('file_count')} files"
            )
        if "content" in observation and isinstance(observation.get("content"), str):
            path = observation.get("path") or args.get("path")
            return f"{tool} read {path or 'content'}"
        if "status" in observation:
            path = observation.get("path") or observation.get("target") or args.get("path")
            return f"{tool} returned status={observation.get('status')} for {path}"
    if isinstance(observation, list):
        return f"{tool} returned {len(observation)} items"
    return f"{tool} returned {type(observation).__name__}"


def _extract_path(observation: Any, args: dict[str, Any]) -> str | None:
    if isinstance(observation, dict):
        for key in ("path", "file", "target"):
            value = observation.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict):
                nested = value.get("path") or value.get("file") or value.get("target")
                if isinstance(nested, str) and nested:
                    return nested
        candidates = observation.get("candidates")
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                value = item.get("path") or item.get("target") or item.get("file")
                if isinstance(value, str) and value:
                    return value
    value = args.get("path")
    if isinstance(value, str) and value:
        return value
    return None


def _tool_calls(response: Any) -> list[ModelToolCall]:
    calls: list[ModelToolCall] = []
    for item in _get(response, "output", []):
        if _get(item, "type") != "function_call":
            continue
        name = _get(item, "name")
        call_id = _get(item, "call_id")
        raw_args = _get(item, "arguments", "{}")
        if not isinstance(name, str) or not name:
            raise ValueError("LLM returned a tool call without a valid name.")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("LLM returned a tool call without a valid call_id.")
        calls.append(ModelToolCall(
            name=name,
            call_id=call_id,
            arguments=_parse_arguments(raw_args),
        ))
    return calls


def _response_output_items(response: Any) -> list[dict[str, Any]]:
    items = _get(response, "output", [])
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
            continue
        if hasattr(item, "model_dump"):
            dumped = item.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                normalized.append(dumped)
                continue
        if hasattr(item, "to_dict"):
            dumped = item.to_dict()
            if isinstance(dumped, dict):
                normalized.append(dumped)
    return normalized


def _parse_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        if not raw_args.strip():
            return {}
        value = json.loads(raw_args)
        if isinstance(value, dict):
            return value
    raise ValueError("LLM tool arguments must be a JSON object.")


def _response_text(response: Any) -> str:
    text = _get(response, "output_text", "")
    if isinstance(text, str) and text.strip():
        return _unwrap_final_text(text.strip())
    return "I could not produce a final answer."


def _unwrap_final_text(text: str) -> str:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(value, dict) and value.get("type") == "final":
        answer = value.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    return text


def _debug_event(label: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > 2_000:
        text = f"{text[:2_000]}..."
    print(f"[{label}] {text}", file=sys.stderr)


def _handle_stream_event(
    event: Any,
    stream_event: Callable[[dict[str, Any]], None] | None,
    *,
    debug: bool,
) -> None:
    payload = _normalize_stream_event(event)
    if payload is None:
        return
    if debug:
        _debug_event("stream-event", payload)
    if stream_event:
        stream_event(payload)


def _normalize_stream_event(event: Any) -> dict[str, Any] | None:
    event_type = _get(event, "type")
    if not isinstance(event_type, str):
        return None

    if event_type == "response.reasoning_text.delta":
        return {
            "kind": "reasoning_delta",
            "delta": _get(event, "delta", ""),
            "item_id": _get(event, "item_id"),
            "sequence_number": _get(event, "sequence_number"),
        }
    if event_type == "response.output_text.delta":
        return {
            "kind": "text_delta",
            "delta": _get(event, "delta", ""),
            "item_id": _get(event, "item_id"),
            "sequence_number": _get(event, "sequence_number"),
        }
    if event_type == "response.output_item.added":
        item = _get(event, "item")
        item_type = _get(item, "type")
        if item_type == "function_call":
            return {
                "kind": "tool_call_started",
                "name": _get(item, "name"),
                "call_id": _get(item, "call_id"),
                "item_id": _get(item, "id"),
                "sequence_number": _get(event, "sequence_number"),
            }
        if item_type == "reasoning":
            return {
                "kind": "reasoning_started",
                "item_id": _get(item, "id"),
                "sequence_number": _get(event, "sequence_number"),
            }
        return {
            "kind": "output_item_added",
            "item_type": item_type,
            "item_id": _get(item, "id"),
            "sequence_number": _get(event, "sequence_number"),
        }
    if event_type == "response.function_call_arguments.delta":
        return {
            "kind": "tool_call_arguments_delta",
            "delta": _get(event, "delta", ""),
            "item_id": _get(event, "item_id"),
            "sequence_number": _get(event, "sequence_number"),
        }
    if event_type == "response.function_call_arguments.done":
        return {
            "kind": "tool_call_arguments_done",
            "arguments": _get(event, "arguments", ""),
            "item_id": _get(event, "item_id"),
            "sequence_number": _get(event, "sequence_number"),
        }
    if event_type == "response.completed":
        return {
            "kind": "response_completed",
            "sequence_number": _get(event, "sequence_number"),
        }
    if event_type == "response.in_progress":
        return {
            "kind": "response_in_progress",
            "sequence_number": _get(event, "sequence_number"),
        }
    return None


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
