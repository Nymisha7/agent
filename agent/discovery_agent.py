from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows falls back to the process-local lock.
    fcntl = None  # type: ignore[assignment]

from .llm import LLMClient
from .policy import PolicyEngine
from .pricing import TokenCost
from .rust_tools import RustTools
from .session_store import utc_now
from .tools import ToolContext, build_tool_registry, verify_mutation_observation


TASK_AGENT_READ_TOOLS = {
    "glob",
    "grep",
    "list_path",
    "path_status",
    "inspect_target",
    "inspect_tree",
    "read_path",
}

TASK_AGENT_WRITE_TOOLS = {"write_file", "edit_file"}

TASK_AGENT_SYSTEM_PROMPT = """You are an independent parallel task agent. Complete one bounded task using your own fresh context and then stop.
You may inspect and read the workspace. When ownership scopes are supplied, you may also create or edit files only inside those exact scopes. Read existing files before changing them and never overwrite sibling work. You cannot delete files, run commands, access the host or desktop, send messages, scan secrets, ask for approval, spawn agents, or continue in the background.
Stay focused on the assigned workstream. Prefer targeted searches and reads over exhaustive repository traversal, and call finish_subagent as soon as the requested work is complete.
Ground claims in tool results. Report exactly what you changed, what remains for the parent, and any cross-scope dependency you could not modify. Never claim a file was changed unless a mutation tool succeeded. Report progress concisely, clearly mark completion, use Markdown, and state ambiguity or missing evidence."""

TASK_AGENT_FINISH_TOOL = {
    "type": "function",
    "name": "finish_subagent",
    "description": "Return the final task report after the independent bounded task is complete.",
    "parameters": {
        "type": "object",
        "properties": {
            "report": {
                "type": "string",
                "description": "Concise evidence-backed report listing completed work and relevant paths.",
            },
            "complete": {
                "type": "boolean",
                "description": "False when a dependency or tool failure prevented completion.",
            },
        },
        "required": ["report", "complete"],
    },
}


MIN_PARALLEL_SUBAGENTS = 2
DEFAULT_MAX_PARALLEL_SUBAGENTS = 4
HARD_MAX_PARALLEL_SUBAGENTS = 8
DEFAULT_SUBAGENT_MAX_STEPS = 8
MIN_SUBAGENT_MAX_STEPS = 2
HARD_MAX_SUBAGENT_MAX_STEPS = 20
DEFAULT_PARALLEL_WORK_FILE = ".agent/parallel-work.md"
PROTECTED_OWNERSHIP_PARTS = {
    ".agent",
    ".git",
    ".hg",
    ".mypy_cache",
    ".packages",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}
_WORK_LOG_LOCK = threading.Lock()


@dataclass
class ParallelSubagentRunner:
    parent_llm: LLMClient
    rust_bin: Path
    workspace_root: Path
    max_steps: int = DEFAULT_SUBAGENT_MAX_STEPS
    max_workers: int = DEFAULT_MAX_PARALLEL_SUBAGENTS
    work_file: str = DEFAULT_PARALLEL_WORK_FILE
    llm_factory: Callable[[], LLMClient] | None = None
    event_handler: Callable[[dict[str, Any]], None] | None = None
    _policy: PolicyEngine = field(default_factory=PolicyEngine, init=False, repr=False)

    @classmethod
    def from_environment(
        cls,
        *,
        parent_llm: LLMClient,
        rust_bin: Path,
        workspace_root: Path,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> ParallelSubagentRunner:
        max_workers = _bounded_env_int(
            "AGENT_MAX_PARALLEL_SUBAGENTS",
            DEFAULT_MAX_PARALLEL_SUBAGENTS,
            MIN_PARALLEL_SUBAGENTS,
            HARD_MAX_PARALLEL_SUBAGENTS,
        )
        max_steps = _bounded_env_int(
            "AGENT_SUBAGENT_MAX_STEPS",
            DEFAULT_SUBAGENT_MAX_STEPS,
            MIN_SUBAGENT_MAX_STEPS,
            HARD_MAX_SUBAGENT_MAX_STEPS,
        )
        work_file = os.getenv("AGENT_PARALLEL_WORK_FILE", DEFAULT_PARALLEL_WORK_FILE).strip()
        return cls(
            parent_llm=parent_llm,
            rust_bin=rust_bin,
            workspace_root=workspace_root,
            max_steps=max_steps,
            max_workers=max_workers,
            work_file=work_file or DEFAULT_PARALLEL_WORK_FILE,
            event_handler=event_handler,
        )

    def run(self, *, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = self._normalize_tasks(tasks)
        if len(normalized) < MIN_PARALLEL_SUBAGENTS:
            raise ValueError(
                "parallel_subagents requires at least two independent tasks; "
                "single subagent execution is intentionally unsupported."
            )
        if len(normalized) > self.max_workers:
            raise ValueError(
                f"parallel_subagents accepts at most {self.max_workers} tasks in this configuration."
            )
        self._prepare_owned_directories(normalized)

        run_id = f"parallel-{uuid.uuid4().hex[:12]}"
        started_at = utc_now()
        work_path = self._resolve_work_path()
        safe_tasks: list[dict[str, Any]] = self._policy.sanitize_observation(normalized)
        self._append_work_log(
            work_path,
            _run_started_markdown(run_id, started_at, normalized, self._policy),
        )
        self._emit({
            "kind": "subagent_run_started",
            "run_id": run_id,
            "total": len(normalized),
            "work_file": str(work_path),
            "tasks": safe_tasks,
            "summary": (
                f"Spawned {len(normalized)} parallel subagents · "
                f"log: {self.work_file}"
            ),
        })

        children = [self.llm_factory() if self.llm_factory else self._new_llm() for _ in normalized]
        results: list[dict[str, Any] | None] = [None] * len(normalized)
        with ThreadPoolExecutor(
            max_workers=len(normalized),
            thread_name_prefix="parallel-subagent",
        ) as executor:
            future_to_index = {}
            for index, task in enumerate(normalized):
                self._emit({
                    "kind": "subagent_task_started",
                    "run_id": run_id,
                    "task_id": safe_tasks[index]["id"],
                    "owned_paths": safe_tasks[index]["owns"],
                    "summary": _bounded_line(safe_tasks[index]["task"]),
                })
            for index, (child, task) in enumerate(zip(children, normalized, strict=True)):
                future = executor.submit(
                    self._run_one,
                    child,
                    task,
                    run_id,
                    safe_tasks[index]["id"],
                )
                future_to_index[future] = index
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                child = children[index]
                task = normalized[index]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "ok": False,
                        "agent": "parallel_subagent",
                        "isolated": True,
                        "execution_mode": "parallel_member",
                        "tool_policy": "scoped_write" if task.get("owns") else "read_only",
                        "owned_paths": list(task.get("owns", [])),
                        "changed_files": [],
                        "complete": False,
                        "error": self._policy.redact_text(str(exc)),
                    }
                consume_cost_metrics = getattr(child, "consume_turn_cost_metrics", None)
                if callable(consume_cost_metrics):
                    usage, cost = consume_cost_metrics()
                    cost_usd = cost.total
                    _merge_cost_breakdown(self.parent_llm, cost)
                else:
                    usage, cost_usd = child.consume_turn_metrics()
                    _merge_cost(self.parent_llm, cost_usd)
                _merge_usage(self.parent_llm, usage)
                result["usage"] = usage
                result["task_id"] = task["id"]
                results[index] = result
                self._append_work_log(
                    work_path,
                    _task_result_markdown(run_id, task, result, self._policy),
                )
                task_status = _result_status(result)
                report = result.get("report") or result.get("error") or "No report returned."
                self._emit({
                    "kind": "subagent_task_completed",
                    "run_id": run_id,
                    "task_id": safe_tasks[index]["id"],
                    "status": task_status,
                    "owned_paths": safe_tasks[index]["owns"],
                    "changed_files": result.get("changed_files", []),
                    "changed_count": len(result.get("changed_files", []))
                    if isinstance(result.get("changed_files"), list)
                    else 0,
                    "summary": _bounded_line(
                        self._policy.redact_text(str(report))
                    ),
                })

        ordered_results = [result for result in results if result is not None]
        complete = all(
            result.get("ok") is True and result.get("complete") is True
            for result in ordered_results
        )
        finished_at = utc_now()
        self._append_work_log(
            work_path,
            _run_finished_markdown(run_id, finished_at, complete),
        )
        completed_count = sum(
            result.get("ok") is True and result.get("complete") is True
            for result in ordered_results
        )
        self._emit({
            "kind": "subagent_run_completed",
            "run_id": run_id,
            "total": len(ordered_results),
            "completed": completed_count,
            "failed": len(ordered_results) - completed_count,
            "work_file": str(work_path),
            "summary": (
                f"{completed_count}/{len(ordered_results)} subagents complete · "
                f"log: {self.work_file}"
            ),
        })
        return {
            "ok": True,
            "agent": "parallel_subagents",
            "execution_mode": "parallel",
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "complete": complete,
            "work_file": str(work_path),
            "tasks": ordered_results,
        }

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_handler is None:
            return
        try:
            self.event_handler(event)
        except Exception:
            # UI or persistence telemetry must never cancel worker execution.
            return

    def _run_one(
        self,
        child_llm: LLMClient,
        task: dict[str, Any],
        run_id: str,
        task_id: str,
    ) -> dict[str, Any]:
        return run_task_agent(
            llm=child_llm,
            rust=RustTools(self.rust_bin),
            workspace_root=self.workspace_root,
            task=task["task"],
            owns=task.get("owns", []),
            max_steps=self.max_steps,
            progress_handler=lambda event: self._emit_task_progress(
                event,
                run_id=run_id,
                task_id=task_id,
                owned_paths=task.get("owns", []),
            ),
        )

    def _emit_task_progress(
        self,
        event: dict[str, Any],
        *,
        run_id: str,
        task_id: str,
        owned_paths: list[str],
    ) -> None:
        summary = event.get("summary")
        payload = {
            key: value
            for key, value in event.items()
            if key not in {"kind", "run_id", "task_id", "owned_paths", "summary"}
        }
        self._emit({
            "kind": "subagent_task_progress",
            "run_id": run_id,
            "task_id": task_id,
            "owned_paths": owned_paths,
            "summary": self._policy.redact_text(summary) if isinstance(summary, str) else "Working",
            **payload,
        })

    def _normalize_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(tasks, list):
            raise ValueError("parallel_subagents arg 'tasks' must be an array.")
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(tasks, start=1):
            if not isinstance(raw, dict):
                raise ValueError("Each parallel task must be an object.")
            task = raw.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("Each parallel task requires non-empty string field 'task'.")
            raw_id = raw.get("id", f"task-{index}")
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError("Parallel task field 'id' must be a non-empty string.")
            task_id = raw_id.strip()
            if task_id in seen_ids:
                raise ValueError(f"Parallel task ids must be unique; duplicate: {task_id}")
            seen_ids.add(task_id)
            item = {"id": task_id, "task": task.strip()}
            raw_owns = raw.get("owns", [])
            if not isinstance(raw_owns, list):
                raise ValueError("Parallel task field 'owns' must be an array when provided.")
            item["owns"] = _validated_owned_paths(self.workspace_root, raw_owns)
            normalized.append(item)
        self._reject_overlapping_ownership(normalized)
        return normalized

    def _reject_overlapping_ownership(self, tasks: list[dict[str, Any]]) -> None:
        claimed: list[tuple[str, Path]] = []
        workspace_root = self.workspace_root.expanduser().resolve()
        for task in tasks:
            for raw_path in task.get("owns", []):
                path = (workspace_root / raw_path).resolve(strict=False)
                for owner, other in claimed:
                    if _paths_overlap(path, other):
                        raise ValueError(
                            "Parallel subagent ownership scopes must not overlap: "
                            f"{owner} owns {other.relative_to(workspace_root)} and "
                            f"{task['id']} owns {path.relative_to(workspace_root)}."
                        )
                claimed.append((task["id"], path))

    def _prepare_owned_directories(self, tasks: list[dict[str, Any]]) -> None:
        workspace_root = self.workspace_root.expanduser().resolve()
        for task in tasks:
            for raw_path in task.get("owns", []):
                path = workspace_root / raw_path
                if path.exists() and not path.is_dir():
                    raise ValueError(
                        f"Independent subagent ownership must be a directory: {raw_path}"
                    )
                path.mkdir(parents=True, exist_ok=True)

    def _resolve_work_path(self) -> Path:
        configured = Path(self.work_file)
        if configured.is_absolute():
            raise ValueError("AGENT_PARALLEL_WORK_FILE must be workspace-relative.")
        root = self.workspace_root.resolve()
        target = (root / configured).resolve()
        if target != root and root not in target.parents:
            raise ValueError("AGENT_PARALLEL_WORK_FILE cannot escape the workspace.")
        return target

    @staticmethod
    def _append_work_log(path: Path, markdown: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _WORK_LOG_LOCK:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    if os.fstat(handle.fileno()).st_size == 0:
                        handle.write("# Parallel Agent Work Log\n")
                    handle.write(markdown)
                    handle.flush()
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _new_llm(self) -> LLMClient:
        child = LLMClient(
            provider=self.parent_llm.provider,
            model=self.parent_llm.model,
        )
        child.reasoning_effort = self.parent_llm.reasoning_effort
        child.reasoning_summary = self.parent_llm.reasoning_summary
        return child


def run_task_agent(
    *,
    llm: LLMClient,
    rust: RustTools,
    workspace_root: Path,
    task: str,
    owns: list[str] | None = None,
    max_steps: int = DEFAULT_SUBAGENT_MAX_STEPS,
    progress_handler: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    isolated_session_id = f"subagent-{uuid.uuid4().hex[:12]}"
    owned_paths = _validated_owned_paths(workspace_root, list(owns or []))
    owned_write_roots = [
        (workspace_root.expanduser().resolve() / owned).resolve(strict=False)
        for owned in owned_paths
    ]
    tool_ctx = ToolContext(
        rust=rust,
        workspace_root=workspace_root,
        search_roots=[],
        owned_write_roots=owned_write_roots,
    )
    allowed_tools = set(TASK_AGENT_READ_TOOLS)
    if owned_paths:
        allowed_tools.update(TASK_AGENT_WRITE_TOOLS)
    registry = build_tool_registry(tool_ctx).restricted(allowed_tools)
    tools = [*registry.schemas(), TASK_AGENT_FINISH_TOOL]
    ownership_text = (
        "\n".join(f"- {owned}" for owned in owned_paths)
        if owned_paths
        else "(read-only task; no mutation scope assigned)"
    )
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": (
            f"Workspace root: {workspace_root}\n"
            f"Owned mutation scopes:\n{ownership_text}\n"
            f"Independent task: {task.strip()}"
        ),
    }]
    policy = PolicyEngine()
    evidence: list[dict[str, Any]] = []
    changed_files: list[dict[str, Any]] = []
    blocked_by_policy = False

    for _step in range(max_steps):
        response = llm.respond(
            instructions=TASK_AGENT_SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
            previous_response_id=None,
            tool_choice=None,
            stream=False,
            event_handler=None,
        )
        calls = _tool_calls(response)
        if not calls:
            return _task_result(
                session_id=isolated_session_id,
                report=policy.redact_text(_response_text(response)),
                evidence=evidence,
                owned_paths=owned_paths,
                changed_files=changed_files,
                complete=not blocked_by_policy,
            )

        outputs: list[dict[str, Any]] = []
        finished_report: str | None = None
        finished_complete = True
        for call in calls:
            name = call["name"]
            arguments = call["arguments"]
            if name == "finish_subagent":
                report = arguments.get("report")
                if isinstance(report, str) and report.strip() and len(calls) == 1:
                    finished_report = report.strip()
                    finished_complete = (
                        arguments.get("complete") is not False
                        and not blocked_by_policy
                    )
                    break
                blocked_by_policy = True
                observation: Any = {
                    "ok": False,
                    "blocked": True,
                    "reason": "finish_subagent_must_be_called_alone",
                }
            elif name not in allowed_tools:
                blocked_by_policy = True
                observation = {
                    "ok": False,
                    "blocked": True,
                    "reason": "tool_not_allowed_for_independent_subagent",
                    "tool": name,
                    "allowed_tools": sorted(allowed_tools),
                }
            else:
                if progress_handler is not None:
                    progress_handler({
                        "status": "running",
                        "summary": f"{name} · running",
                    })
                try:
                    observation = registry.execute(name, arguments, tool_ctx)
                except Exception as exc:
                    observation = {
                        "ok": False,
                        "tool": name,
                        "blocked": True,
                        "error": str(exc),
                    }
                if name in TASK_AGENT_WRITE_TOOLS:
                    observation = verify_mutation_observation(
                        name,
                        arguments,
                        observation,
                        workspace_root=workspace_root,
                        allowed_write_roots=owned_write_roots,
                    )
                    blocked_by_policy |= (
                        isinstance(observation, dict)
                        and observation.get("ok") is False
                    )

            sanitized = policy.sanitize_observation(observation)
            evidence.append({
                "tool": name,
                "summary": _observation_summary(sanitized),
            })
            if (
                name in TASK_AGENT_WRITE_TOOLS
                and isinstance(sanitized, dict)
                and sanitized.get("ok") is not False
                and isinstance(sanitized.get("path"), str)
            ):
                _record_changed_file(changed_files, {
                    "path": _workspace_relative_path(
                        sanitized["path"],
                        workspace_root,
                    ),
                    "tool": name,
                    "after_sha256": sanitized.get("after_sha256"),
                })
            if progress_handler is not None:
                progress_handler({
                    "status": "running",
                    "changed_count": len(changed_files),
                    "summary": f"{name} · {_observation_summary(sanitized)}",
                })
            outputs.append({
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": _bounded_json(sanitized),
            })

        if finished_report is not None:
            return _task_result(
                session_id=isolated_session_id,
                report=policy.redact_text(finished_report),
                evidence=evidence,
                owned_paths=owned_paths,
                changed_files=changed_files,
                complete=finished_complete,
            )
        messages.extend(_response_items(response))
        messages.extend(outputs)

    messages.append({
        "role": "user",
        "content": (
            "The bounded tool budget is exhausted. Do not do more workspace work. "
            "Call finish_subagent now with a concise evidence-backed report and set "
            "complete to whether the assigned task is actually complete from the "
            "tool results already present."
        ),
    })
    final_response = llm.respond(
        instructions=TASK_AGENT_SYSTEM_PROMPT,
        messages=messages,
        tools=[TASK_AGENT_FINISH_TOOL],
        previous_response_id=None,
        tool_choice="required",
        stream=False,
        event_handler=None,
    )
    final_calls = _tool_calls(final_response)
    if len(final_calls) == 1 and final_calls[0]["name"] == "finish_subagent":
        arguments = final_calls[0]["arguments"]
        report = arguments.get("report")
        if isinstance(report, str) and report.strip():
            return _task_result(
                session_id=isolated_session_id,
                report=policy.redact_text(report.strip()),
                evidence=evidence,
                owned_paths=owned_paths,
                changed_files=changed_files,
                complete=(
                    arguments.get("complete") is not False
                    and not blocked_by_policy
                ),
            )

    return _task_result(
        session_id=isolated_session_id,
        report=(
            "Independent subagent stopped at its step limit. The parent should inspect the "
            "reported changes and complete any remaining work."
        ),
        evidence=evidence,
        owned_paths=owned_paths,
        changed_files=changed_files,
        complete=False,
    )


def _task_result(
    *,
    session_id: str,
    report: str,
    evidence: list[dict[str, Any]],
    owned_paths: list[str] | None = None,
    changed_files: list[dict[str, Any]] | None = None,
    complete: bool = True,
) -> dict[str, Any]:
    changed = list(changed_files or [])
    return {
        "ok": True,
        "agent": "parallel_subagent",
        "session_id": session_id,
        "isolated": True,
        "execution_mode": "parallel_member",
        "background": False,
        "tool_policy": "scoped_write" if owned_paths else "read_only",
        "owned_paths": list(owned_paths or []),
        "changed_files": changed,
        "changed_count": len(changed),
        "complete": complete,
        "status": "complete" if complete else "blocked",
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
    return "Independent task completed without a report."


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


def _merge_cost(parent_llm: LLMClient, cost_usd: float) -> None:
    if isinstance(cost_usd, (int, float)):
        parent_llm.turn_cost_usd += max(0.0, float(cost_usd))


def _merge_cost_breakdown(parent_llm: LLMClient, cost: TokenCost) -> None:
    parent_llm.turn_cost = parent_llm.turn_cost + cost
    parent_llm.turn_cost_usd += cost.total


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _paths_overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _validated_owned_paths(workspace_root: Path, raw_paths: list[Any]) -> list[str]:
    root = workspace_root.expanduser().resolve()
    normalized: list[str] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Each owned path must be a non-empty workspace-relative string.")
        candidate = Path(raw_path.strip())
        if candidate.is_absolute():
            raise ValueError("Owned paths must be workspace-relative.")
        if any(part in PROTECTED_OWNERSHIP_PARTS for part in candidate.parts):
            raise ValueError("Owned paths cannot include metadata or generated directories.")
        unresolved = root / candidate
        if _path_has_symlink_component(unresolved, root):
            raise ValueError("Owned paths cannot contain symlink components.")
        resolved = unresolved.resolve(strict=False)
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("Owned paths cannot escape the workspace.") from exc
        if not relative.parts:
            raise ValueError("An independent subagent cannot own the entire workspace.")
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(relative.as_posix())
    return normalized


def _path_has_symlink_component(path: Path, workspace_root: Path) -> bool:
    current = workspace_root
    try:
        relative = path.relative_to(workspace_root)
    except ValueError:
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _workspace_relative_path(value: str, workspace_root: Path) -> str:
    path = Path(value).expanduser().resolve(strict=False)
    try:
        return path.relative_to(workspace_root.expanduser().resolve()).as_posix()
    except ValueError:
        return str(path)


def _record_changed_file(
    changed_files: list[dict[str, Any]],
    change: dict[str, Any],
) -> None:
    path = change.get("path")
    for index, existing in enumerate(changed_files):
        if existing.get("path") == path:
            changed_files[index] = change
            return
    changed_files.append(change)


def _markdown_text(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _bounded_line(value: Any, *, limit: int = 120) -> str:
    text = _markdown_text(value)
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _run_started_markdown(
    run_id: str,
    started_at: str,
    tasks: list[dict[str, Any]],
    policy: PolicyEngine,
) -> str:
    lines = [
        f"\n## Parallel run `{_markdown_text(run_id)}`\n",
        f"- Started: `{_markdown_text(started_at)}`\n",
        "- Execution: `parallel-only`\n",
        "- Status: `running`\n",
        "\n### Work plan\n",
    ]
    for task in tasks:
        ownership = ", ".join(
            f"`{_markdown_text(policy.redact_text(owned))}`"
            for owned in task.get("owns", [])
        ) or "`read-only`"
        lines.append(
            f"- [ ] `{_markdown_text(task['id'])}`: "
            f"{_markdown_text(policy.redact_text(task['task']))}; "
            f"owns: {ownership}\n"
        )
    return "".join(lines)


def _task_result_markdown(
    run_id: str,
    task: dict[str, Any],
    result: dict[str, Any],
    policy: PolicyEngine,
) -> str:
    status = _result_status(result)
    report = result.get("report") or result.get("error") or "No report returned."
    sanitized_report = policy.redact_text(str(report)).strip()
    changed_files = result.get("changed_files")
    changed_paths = [
        _markdown_text(policy.redact_text(item.get("path")))
        for item in changed_files
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ] if isinstance(changed_files, list) else []
    changes = (
        "\n- Changed files:\n" + "".join(f"  - `{path}`\n" for path in changed_paths)
        if changed_paths
        else "\n- Changed files: none\n"
    )
    return (
        f"\n### Task `{_markdown_text(task['id'])}` — {status}\n"
        f"- Run: `{_markdown_text(run_id)}`\n"
        f"- Request: {_markdown_text(policy.redact_text(task['task']))}\n"
        f"{changes}\n"
        f"{sanitized_report}\n"
    )


def _result_status(result: dict[str, Any]) -> str:
    if result.get("ok") is not True:
        return "failed"
    return "complete" if result.get("complete") is True else "blocked"


def _run_finished_markdown(run_id: str, finished_at: str, complete: bool) -> str:
    status = "complete" if complete else "incomplete"
    return (
        f"\n### Run `{_markdown_text(run_id)}` finished\n"
        f"- Finished: `{_markdown_text(finished_at)}`\n"
        f"- Status: `{status}`\n"
    )


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
