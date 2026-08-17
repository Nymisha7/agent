import hashlib
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agent.discovery_agent import (
    ParallelSubagentRunner,
    TASK_AGENT_READ_TOOLS,
    TASK_AGENT_SYSTEM_PROMPT,
    run_task_agent,
)
from agent.tools import ToolContext, build_tool_registry


class ScriptedLLM:
    provider = "openai"
    model = "test-model"
    reasoning_effort = None
    reasoning_summary = None

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.turn_usage = {
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
        self.turn_cost_usd = 0.0

    def respond(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.responses.pop(0)

    def consume_turn_usage(self) -> dict[str, int]:
        usage = dict(self.turn_usage)
        self.turn_usage = {key: 0 for key in self.turn_usage}
        return usage

    def consume_turn_metrics(self) -> tuple[dict[str, int], float]:
        usage = self.consume_turn_usage()
        cost_usd = self.turn_cost_usd
        self.turn_cost_usd = 0.0
        return usage, cost_usd


def tool_call(name: str, arguments: dict[str, Any], call_id: str) -> Any:
    return SimpleNamespace(
        output=[{
            "type": "function_call",
            "name": name,
            "call_id": call_id,
            "arguments": json.dumps(arguments),
        }],
        output_text="",
    )


class FileRust:
    def __init__(self, *, report_wrong_hash: bool = False) -> None:
        self.report_wrong_hash = report_wrong_hash
        self.write_calls: list[dict[str, Any]] = []
        self.edit_calls: list[dict[str, Any]] = []

    def write_file(self, **kwargs: Any) -> dict[str, Any]:
        self.write_calls.append(kwargs)
        path = Path(kwargs["path"])
        if kwargs["create_dirs"]:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(kwargs["content"], encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "ok": True,
            "tool": "write_file",
            "path": str(path),
            "after_sha256": "0" * 64 if self.report_wrong_hash else digest,
        }

    def edit_file(self, **kwargs: Any) -> dict[str, Any]:
        self.edit_calls.append(kwargs)
        path = Path(kwargs["path"])
        content = path.read_text(encoding="utf-8")
        updated = content.replace(
            kwargs["old_text"],
            kwargs["new_text"],
            -1 if kwargs["replace_all"] else 1,
        )
        path.write_text(updated, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "ok": True,
            "tool": "edit_file",
            "path": str(path),
            "after_sha256": "0" * 64 if self.report_wrong_hash else digest,
        }


class DiscoveryAgentTests(unittest.TestCase):
    def test_worker_prompt_is_generic_and_requires_explicit_finish(self) -> None:
        self.assertIn("call finish_subagent as soon as", TASK_AGENT_SYSTEM_PROMPT)
        self.assertNotIn("application/task names", TASK_AGENT_SYSTEM_PROMPT)

    def test_subagent_step_budget_is_configurable_and_bounded(self) -> None:
        parent = ScriptedLLM([])
        with patch.dict("os.environ", {"AGENT_SUBAGENT_MAX_STEPS": "12"}):
            configured = ParallelSubagentRunner.from_environment(
                parent_llm=parent,  # type: ignore[arg-type]
                rust_bin=Path("/tmp/agent-rust"),
                workspace_root=Path("/workspace"),
            )
        with patch.dict("os.environ", {"AGENT_SUBAGENT_MAX_STEPS": "999"}):
            bounded = ParallelSubagentRunner.from_environment(
                parent_llm=parent,  # type: ignore[arg-type]
                rust_bin=Path("/tmp/agent-rust"),
                workspace_root=Path("/workspace"),
            )

        self.assertEqual(configured.max_steps, 12)
        self.assertEqual(bounded.max_steps, 20)

    def test_restricted_registry_physically_excludes_mutation_and_host_tools(self) -> None:
        with TemporaryDirectory() as tmp:
            ctx = ToolContext(
                rust=SimpleNamespace(),
                workspace_root=Path(tmp),
                search_roots=[],
                parallel_runner=lambda **_kwargs: {},
            )
            registry = build_tool_registry(ctx).restricted(TASK_AGENT_READ_TOOLS)

        names = {schema["name"] for schema in registry.schemas()}
        self.assertEqual(names, TASK_AGENT_READ_TOOLS)
        for forbidden in (
            "write_file",
            "edit_file",
            "delete_path",
            "run_system_command",
            "desktop_action",
            "desktop_observe",
            "connected_devices",
            "secret_scan",
            "parallel_subagents",
        ):
            self.assertNotIn(forbidden, names)

    def test_discovery_agent_blocks_hallucinated_mutation_tool(self) -> None:
        llm = ScriptedLLM([
            tool_call("write_file", {"path": "unsafe.txt", "content": "no"}, "write-1"),
            tool_call("finish_subagent", {
                "report": "No mutation was performed.",
                "complete": True,
            }, "finish-1"),
        ])

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_task_agent(
                llm=llm,  # type: ignore[arg-type]
                rust=SimpleNamespace(),  # type: ignore[arg-type]
                workspace_root=root,
                task="Find the target without changing anything",
            )
            self.assertFalse((root / "unsafe.txt").exists())

        exposed = {tool["name"] for tool in llm.requests[0]["tools"]}
        self.assertEqual(exposed, TASK_AGENT_READ_TOOLS | {"finish_subagent"})
        self.assertTrue(result["isolated"])
        self.assertEqual(result["execution_mode"], "parallel_member")
        self.assertFalse(result["background"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["tool_policy"], "read_only")
        self.assertEqual(
            result["evidence"][0]["summary"],
            "tool_not_allowed_for_independent_subagent",
        )

    def test_task_agent_does_not_stream_model_prose_into_lifecycle_events(self) -> None:
        llm = ScriptedLLM([
            SimpleNamespace(output=[], output_text="No issue found."),
        ])
        events: list[dict[str, Any]] = []

        with TemporaryDirectory() as tmp:
            run_task_agent(
                llm=llm,  # type: ignore[arg-type]
                rust=SimpleNamespace(),  # type: ignore[arg-type]
                workspace_root=Path(tmp),
                task="Inspect the tests",
                progress_handler=events.append,
            )

        self.assertFalse(llm.requests[0]["stream"])
        self.assertIsNone(llm.requests[0]["event_handler"])
        self.assertEqual(events, [])

    def test_step_budget_reserves_a_final_completion_decision(self) -> None:
        llm = ScriptedLLM([
            tool_call(
                "write_file",
                {"path": "frontend/app.js", "content": "export const ready = true;"},
                "write-1",
            ),
            tool_call(
                "finish_subagent",
                {"report": "Implemented frontend/app.js.", "complete": True},
                "finish-1",
            ),
        ])

        with TemporaryDirectory() as tmp:
            result = run_task_agent(
                llm=llm,  # type: ignore[arg-type]
                rust=FileRust(),  # type: ignore[arg-type]
                workspace_root=Path(tmp),
                task="Implement the frontend module",
                owns=["frontend"],
                max_steps=1,
            )

        self.assertTrue(result["complete"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(llm.requests[1]["tool_choice"], "required")
        self.assertEqual(
            [tool["name"] for tool in llm.requests[1]["tools"]],
            ["finish_subagent"],
        )

    def test_task_agent_can_write_and_edit_only_inside_owned_scope(self) -> None:
        llm = ScriptedLLM([
            tool_call("write_file", {
                "path": "frontend/app.py",
                "content": "value = 1\n",
            }, "write-1"),
            tool_call("edit_file", {
                "path": "frontend/app.py",
                "old_text": "value = 1",
                "new_text": "value = 2",
            }, "edit-1"),
            tool_call("finish_subagent", {
                "report": "Implemented the frontend.",
                "complete": True,
            }, "finish-1"),
        ])
        rust = FileRust()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frontend").mkdir()
            result = run_task_agent(
                llm=llm,  # type: ignore[arg-type]
                rust=rust,  # type: ignore[arg-type]
                workspace_root=root,
                task="Implement the frontend",
                owns=["frontend"],
            )
            content = (root / "frontend" / "app.py").read_text(encoding="utf-8")

        exposed = {tool["name"] for tool in llm.requests[0]["tools"]}
        self.assertIn("write_file", exposed)
        self.assertIn("edit_file", exposed)
        self.assertNotIn("delete_path", exposed)
        self.assertNotIn("run_system_command", exposed)
        self.assertNotIn("parallel_subagents", exposed)
        self.assertEqual(content, "value = 2\n")
        self.assertEqual(result["tool_policy"], "scoped_write")
        self.assertEqual(result["owned_paths"], ["frontend"])
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(result["changed_files"][0]["path"], "frontend/app.py")
        self.assertEqual(rust.write_calls[0]["workspace_root"].name, "frontend")

    def test_task_agent_blocks_cross_scope_write_before_rust_mutation(self) -> None:
        llm = ScriptedLLM([
            tool_call("write_file", {
                "path": "backend/app.py",
                "content": "unsafe\n",
            }, "write-1"),
            tool_call("finish_subagent", {
                "report": "Reported the cross-scope dependency.",
                "complete": False,
            }, "finish-1"),
        ])
        rust = FileRust()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frontend").mkdir()
            result = run_task_agent(
                llm=llm,  # type: ignore[arg-type]
                rust=rust,  # type: ignore[arg-type]
                workspace_root=root,
                task="Implement the frontend",
                owns=["frontend"],
            )
            self.assertFalse((root / "backend" / "app.py").exists())

        self.assertEqual(rust.write_calls, [])
        self.assertFalse(result["complete"])
        self.assertEqual(result["changed_files"], [])
        self.assertEqual(
            result["evidence"][0]["summary"],
            "subagent_write_scope_violation",
        )

    def test_task_agent_rejects_unverified_mutation_hash(self) -> None:
        llm = ScriptedLLM([
            tool_call("write_file", {
                "path": "frontend/app.py",
                "content": "value = 1\n",
            }, "write-1"),
            SimpleNamespace(output=[], output_text="Stopped after verification failure."),
        ])

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frontend").mkdir()
            result = run_task_agent(
                llm=llm,  # type: ignore[arg-type]
                rust=FileRust(report_wrong_hash=True),  # type: ignore[arg-type]
                workspace_root=root,
                task="Implement the frontend",
                owns=["frontend"],
            )

        self.assertEqual(result["changed_files"], [])
        self.assertEqual(
            result["evidence"][0]["summary"],
            "mutation_verification_failed",
        )

    def test_runner_executes_all_children_in_parallel_and_updates_one_log(self) -> None:
        barrier = threading.Barrier(2, timeout=2)

        class BarrierLLM(ScriptedLLM):
            def respond(self, **kwargs: Any) -> Any:
                barrier.wait()
                return super().respond(**kwargs)

        first_child = BarrierLLM([SimpleNamespace(output=[], output_text="first report")])
        first_child.turn_usage["input"] = 100
        first_child.turn_cost_usd = 0.10
        second_child = BarrierLLM([SimpleNamespace(output=[], output_text="second report")])
        second_child.turn_usage["output"] = 20
        second_child.turn_cost_usd = 0.20
        children = [first_child, second_child]
        factory_lock = threading.Lock()

        def factory() -> ScriptedLLM:
            with factory_lock:
                return children.pop(0)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = ScriptedLLM([])
            events: list[dict[str, Any]] = []
            runner = ParallelSubagentRunner(
                parent_llm=parent,  # type: ignore[arg-type]
                rust_bin=Path("/tmp/agent-rust"),
                workspace_root=root,
                llm_factory=factory,  # type: ignore[arg-type]
                event_handler=events.append,
            )
            result = runner.run(tasks=[
                {"id": "architecture", "task": "inspect architecture"},
                {"id": "tests", "task": "inspect tests"},
            ])
            log = (root / ".agent" / "parallel-work.md").read_text(encoding="utf-8")

        self.assertEqual(result["execution_mode"], "parallel")
        self.assertEqual(
            [task["task_id"] for task in result["tasks"]],
            ["architecture", "tests"],
        )
        self.assertIn("first report", log)
        self.assertIn("second report", log)
        self.assertIn("parallel-only", log)
        self.assertEqual(parent.turn_usage["input"], 100)
        self.assertEqual(parent.turn_usage["output"], 20)
        self.assertAlmostEqual(parent.turn_cost_usd, 0.30)
        event_kinds = [event["kind"] for event in events]
        self.assertEqual(event_kinds[0], "subagent_run_started")
        self.assertEqual(event_kinds[-1], "subagent_run_completed")
        started_positions = [
            index for index, kind in enumerate(event_kinds)
            if kind == "subagent_task_started"
        ]
        completed_positions = [
            index for index, kind in enumerate(event_kinds)
            if kind == "subagent_task_completed"
        ]
        self.assertEqual(len(started_positions), 2)
        self.assertEqual(len(completed_positions), 2)
        self.assertLess(max(started_positions), min(completed_positions))

    def test_runner_rejects_singleton_and_duplicate_tasks(self) -> None:
        parent = ScriptedLLM([])
        runner = ParallelSubagentRunner(
            parent_llm=parent,  # type: ignore[arg-type]
            rust_bin=Path("/tmp/agent-rust"),
            workspace_root=Path("/workspace"),
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            runner.run(tasks=[{"id": "one", "task": "only task"}])
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.run(tasks=[
                {"id": "same", "task": "first"},
                {"id": "same", "task": "second"},
            ])

    def test_runner_rejects_unsafe_or_overlapping_ownership(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = ParallelSubagentRunner(
                parent_llm=ScriptedLLM([]),  # type: ignore[arg-type]
                rust_bin=Path("/tmp/agent-rust"),
                workspace_root=root,
            )
            unsafe_values = [".", "../outside", "/absolute", ".git/work", "target/app"]
            for unsafe in unsafe_values:
                with self.subTest(unsafe=unsafe):
                    with self.assertRaises(ValueError):
                        runner._normalize_tasks([
                            {"id": "one", "task": "first", "owns": [unsafe]},
                            {"id": "two", "task": "second", "owns": ["safe"]},
                        ])
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                runner._normalize_tasks([
                    {"id": "one", "task": "first", "owns": ["app"]},
                    {"id": "two", "task": "second", "owns": ["app/api"]},
                ])

    def test_runner_rejects_symlinked_ownership(self) -> None:
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            root = Path(tmp)
            (root / "linked").symlink_to(Path(outside), target_is_directory=True)
            runner = ParallelSubagentRunner(
                parent_llm=ScriptedLLM([]),  # type: ignore[arg-type]
                rust_bin=Path("/tmp/agent-rust"),
                workspace_root=root,
            )
            with self.assertRaisesRegex(ValueError, "symlink"):
                runner._normalize_tasks([
                    {"id": "one", "task": "first", "owns": ["linked/work"]},
                    {"id": "two", "task": "second", "owns": ["safe"]},
                ])

    def test_one_failed_child_does_not_cancel_its_sibling(self) -> None:
        class FailingLLM(ScriptedLLM):
            def respond(self, **_kwargs: Any) -> Any:
                raise RuntimeError("child failed")

        children = [
            FailingLLM([]),
            ScriptedLLM([SimpleNamespace(output=[], output_text="sibling completed")]),
        ]

        with TemporaryDirectory() as tmp:
            runner = ParallelSubagentRunner(
                parent_llm=ScriptedLLM([]),  # type: ignore[arg-type]
                rust_bin=Path("/tmp/agent-rust"),
                workspace_root=Path(tmp),
                llm_factory=lambda: children.pop(0),  # type: ignore[arg-type]
            )
            result = runner.run(tasks=[
                {"id": "fails", "task": "failing task"},
                {"id": "survives", "task": "successful task"},
            ])

        self.assertFalse(result["complete"])
        self.assertFalse(result["tasks"][0]["ok"])
        self.assertEqual(result["tasks"][1]["report"], "sibling completed")

    def test_each_discovery_run_gets_a_fresh_in_memory_session(self) -> None:
        first_llm = ScriptedLLM([
            SimpleNamespace(output=[], output_text="first report"),
        ])
        second_llm = ScriptedLLM([
            SimpleNamespace(output=[], output_text="second report"),
        ])

        with TemporaryDirectory() as tmp:
            first = run_task_agent(
                llm=first_llm,  # type: ignore[arg-type]
                rust=SimpleNamespace(),  # type: ignore[arg-type]
                workspace_root=Path(tmp),
                task="first task",
            )
            second = run_task_agent(
                llm=second_llm,  # type: ignore[arg-type]
                rust=SimpleNamespace(),  # type: ignore[arg-type]
                workspace_root=Path(tmp),
                task="second task",
            )

        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertIn("first task", first_llm.requests[0]["messages"][0]["content"])
        self.assertNotIn("first task", second_llm.requests[0]["messages"][0]["content"])

    def test_tool_callback_runs_to_completion_before_returning(self) -> None:
        order: list[str] = []

        def run_discovery(**_kwargs: Any) -> dict[str, Any]:
            order.append("child-start")
            order.append("child-finish")
            return {"ok": True, "complete": True}

        with TemporaryDirectory() as tmp:
            ctx = ToolContext(
                rust=SimpleNamespace(),
                workspace_root=Path(tmp),
                search_roots=[],
                parallel_runner=run_discovery,
            )
            registry = build_tool_registry(ctx)
            result = registry.execute(
                "parallel_subagents",
                {
                    "tasks": [
                        {"id": "entry", "task": "find entry points"},
                        {"id": "tests", "task": "find relevant tests"},
                    ],
                },
                ctx,
            )
            order.append("parent-resumed")

        self.assertEqual(order, ["child-start", "child-finish", "parent-resumed"])
        self.assertTrue(result["complete"])

    def test_tool_schema_requires_a_parallel_batch(self) -> None:
        with TemporaryDirectory() as tmp:
            ctx = ToolContext(
                rust=SimpleNamespace(),
                workspace_root=Path(tmp),
                search_roots=[],
                parallel_runner=lambda **_kwargs: {},
            )
            schemas = {
                schema["name"]: schema
                for schema in build_tool_registry(ctx).schemas()
            }

        self.assertNotIn("discovery_subagent", schemas)
        task_schema = schemas["parallel_subagents"]["parameters"]["properties"]["tasks"]
        self.assertEqual(task_schema["minItems"], 2)
        self.assertEqual(task_schema["maxItems"], 8)
        item_properties = task_schema["items"]["properties"]
        self.assertIn("owns", item_properties)


if __name__ == "__main__":
    unittest.main()
