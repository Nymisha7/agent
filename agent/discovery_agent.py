from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .llm import LLMClient
from .policy import PolicyEngine
from .rust_tools import RustTools
from .tools import ToolContext, build_tool_registry


DISCOVERY_TOOL_NAMES = {
    "glob",
    "grep",
    "list_path",
    "path_status",
    "inspect_target",
    "inspect_tree",
    "read_path",
}

DISCOVERY_SYSTEM_PROMPT = """You are a read-only discovery subagent. Complete one bounded search or inspection task and then stop.
You may only resolve paths, list/read files, inspect directory trees, glob filenames, grep text, and verify path status. You cannot edit/delete files, run commands, use secret-scanning tools, access host devices, control the desktop, ask for approval, spawn agents, or continue in the background.
Ground every workspace claim in tool results. Resolve named targets before broad inspection. Return concise evidence with exact paths and clearly state ambiguity or missing evidence. Never claim an action or mutation occurred."""

DISCOVERY_FINISH_TOOL = {
    "type": "function",
    "name": "finish_discovery",
    "description": "Return the final discovery report after the bounded read-only task is complete.",
    "parameters": {
        "type": "object",
        "properties": {
            "report": {
                "type": "string",
                "description": "Concise evidence-backed report with relevant paths.",
            }
        },
        "required": ["report"],
    },
}


@dataclass
class DiscoverySubagentRunner:
    parent_llm: LLMClient
    rust_bin: Path
    workspace_root: Path
    max_steps: int = 6
    llm_factory: Callable[[], LLMClient] | None = None
    _run_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def run(self, *, task: str, path: str | None = None) -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            return {
                "ok": False,
                "agent": "discovery_subagent",
                "blocked": True,
                "reason": "subagent_already_running",
                "guidance": "Wait for the active discovery subagent to finish before starting another.",
            }

        child_llm: LLMClient | None = None
        try:
            child_llm = self.llm_factory() if self.llm_factory is not None else self._new_llm()
            result = run_discovery_agent(
                llm=child_llm,
                rust=RustTools(self.rust_bin),
                workspace_root=self.workspace_root,
                task=task,
                path=path,
                max_steps=self.max_steps,
            )
            usage = child_llm.consume_turn_usage()
            _merge_usage(self.parent_llm, usage)
            result["usage"] = usage
            return result
        except Exception as exc:
            if child_llm is not None:
                _merge_usage(self.parent_llm, child_llm.consume_turn_usage())
            return {
                "ok": False,
                "agent": "discovery_subagent",
                "isolated": True,
                "sequential": True,
                "tool_policy": "read_only_discovery",
                "error": str(exc),
            }
        finally:
            self._run_lock.release()

    def _new_llm(self) -> LLMClient:
        child = LLMClient(
            provider=self.parent_llm.provider,
            model=self.parent_llm.model,
        )
        child.reasoning_effort = self.parent_llm.reasoning_effort
        child.reasoning_summary = self.parent_llm.reasoning_summary
        return child


def run_discovery_agent(
    *,
    llm: LLMClient,
    rust: RustTools,
    workspace_root: Path,
    task: str,
    path: str | None = None,
    max_steps: int = 6,
) -> dict[str, Any]:
    isolated_session_id = f"discovery-{uuid.uuid4().hex[:12]}"
    tool_ctx = ToolContext(
        rust=rust,
        workspace_root=workspace_root,
        search_roots=[],
    )
    registry = build_tool_registry(tool_ctx).restricted(DISCOVERY_TOOL_NAMES)
    tools = [*registry.schemas(), DISCOVERY_FINISH_TOOL]
    scope_text = path.strip() if isinstance(path, str) and path.strip() else "."
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": (
            f"Workspace root: {workspace_root}\n"
            f"Requested scope: {scope_text}\n"
            f"Discovery task: {task.strip()}"
        ),
    }]
    policy = PolicyEngine()
    evidence: list[dict[str, Any]] = []

    for _step in range(max_steps):
        response = llm.respond(
            instructions=DISCOVERY_SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
            previous_response_id=None,
            tool_choice=None,
            stream=False,
            event_handler=None,
        )
        calls = _tool_calls(response)
        if not calls:
            return _discovery_result(
                session_id=isolated_session_id,
                report=policy.redact_text(_response_text(response)),
                evidence=evidence,
            )

        outputs: list[dict[str, Any]] = []
        finished_report: str | None = None
        for call in calls:
            name = call["name"]
            arguments = call["arguments"]
            if name == "finish_discovery":
                report = arguments.get("report")
                if isinstance(report, str) and report.strip() and len(calls) == 1:
                    finished_report = report.strip()
                    break
                observation: Any = {
                    "ok": False,
                    "blocked": True,
                    "reason": "finish_discovery_must_be_called_alone",
                }
            elif name not in DISCOVERY_TOOL_NAMES:
                observation = {
                    "ok": False,
                    "blocked": True,
                    "reason": "tool_not_allowed_for_discovery_subagent",
                    "tool": name,
                    "allowed_tools": sorted(DISCOVERY_TOOL_NAMES),
                }
            else:
                try:
                    observation = registry.execute(name, arguments, tool_ctx)
                except Exception as exc:
                    observation = {
                        "ok": False,
                        "tool": name,
                        "blocked": True,
                        "error": str(exc),
                    }

            sanitized = policy.sanitize_observation(observation)
            evidence.append({
                "tool": name,
                "summary": _observation_summary(sanitized),
            })
            outputs.append({
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": _bounded_json(sanitized),
            })

        if finished_report is not None:
            return _discovery_result(
                session_id=isolated_session_id,
                report=policy.redact_text(finished_report),
                evidence=evidence,
            )
        messages.extend(_response_items(response))
        messages.extend(outputs)

    return _discovery_result(
        session_id=isolated_session_id,
        report=(
            "Discovery stopped at its step limit. The parent should use the collected evidence "
            "or run a narrower sequential discovery task."
        ),
        evidence=evidence,
        complete=False,
    )


def _discovery_result(
    *,
    session_id: str,
    report: str,
    evidence: list[dict[str, Any]],
    complete: bool = True,
) -> dict[str, Any]:
    return {
        "ok": True,
        "agent": "discovery_subagent",
        "session_id": session_id,
        "isolated": True,
        "sequential": True,
        "background": False,
        "tool_policy": "read_only_discovery",
        "complete": complete,
        "report": report,
        "evidence": evidence[-20:],
    }


def _tool_calls(response: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _get(response, "output", []) or []:
        if _get(item, "type") != "function_call":
            continue
        name = _get(item, "name")
        call_id = _get(item, "call_id")
        if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
            continue
        raw_arguments = _get(item, "arguments", "{}")
        arguments = _json_object(raw_arguments)
        calls.append({"name": name, "call_id": call_id, "arguments": arguments})
    return calls


def _response_items(response: Any) -> list[dict[str, Any]]:
    items = _get(response, "output", []) or []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        elif hasattr(item, "model_dump"):
            dumped = item.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                normalized.append(dumped)
    return normalized


def _response_text(response: Any) -> str:
    text = _get(response, "output_text", "")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return "Discovery completed without a report."


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value or "{}")
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Discovery tool arguments must be a JSON object.")


def _bounded_json(value: Any, *, max_bytes: int = 6_000) -> str:
    payload = json.dumps(value, ensure_ascii=False, default=str)
    if len(payload.encode("utf-8")) <= max_bytes:
        return payload
    preview = payload[: max_bytes - 100]
    return json.dumps({
        "truncated": True,
        "original_bytes": len(payload.encode("utf-8")),
        "preview": preview,
    }, ensure_ascii=False)


def _observation_summary(observation: Any) -> str:
    if not isinstance(observation, dict):
        return "tool returned a non-object observation"
    if observation.get("ok") is False:
        return str(observation.get("reason") or observation.get("error") or "tool failed")
    for key in ("matches", "entries", "files", "candidates"):
        value = observation.get(key)
        if isinstance(value, list):
            return f"{len(value)} {key}"
    path = observation.get("path")
    return f"observed {path}" if isinstance(path, str) else "observation recorded"


def _merge_usage(parent_llm: LLMClient, usage: dict[str, int]) -> None:
    turn_usage = getattr(parent_llm, "turn_usage", None)
    if not isinstance(turn_usage, dict):
        return
    for key, value in usage.items():
        if isinstance(value, int):
            turn_usage[key] = int(turn_usage.get(key, 0)) + value


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
