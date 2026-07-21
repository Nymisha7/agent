import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from agent.discovery_agent import (
    DISCOVERY_TOOL_NAMES,
    DiscoverySubagentRunner,
    run_discovery_agent,
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

    def respond(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.responses.pop(0)

    def consume_turn_usage(self) -> dict[str, int]:
        usage = dict(self.turn_usage)
        self.turn_usage = {key: 0 for key in self.turn_usage}
        return usage


class DiscoveryAgentTests(unittest.TestCase):
    def test_restricted_registry_physically_excludes_mutation_and_host_tools(self) -> None:
        with TemporaryDirectory() as tmp:
            ctx = ToolContext(
                rust=SimpleNamespace(),
                workspace_root=Path(tmp),
                search_roots=[],
                discovery_runner=lambda **_kwargs: {},
            )
            registry = build_tool_registry(ctx).restricted(DISCOVERY_TOOL_NAMES)

        names = {schema["name"] for schema in registry.schemas()}
        self.assertEqual(names, DISCOVERY_TOOL_NAMES)
        for forbidden in (
            "write_file",
            "edit_file",
            "delete_path",
            "run_system_command",
            "desktop_action",
            "secret_scan",
            "connected_devices",
            "discovery_subagent",
        ):
            self.assertNotIn(forbidden, names)

    def test_discovery_agent_blocks_hallucinated_mutation_tool(self) -> None:
        llm = ScriptedLLM([
            SimpleNamespace(
                output=[{
                    "type": "function_call",
                    "name": "write_file",
                    "call_id": "write-1",
                    "arguments": json.dumps({"path": "unsafe.txt", "content": "no"}),
                }],
                output_text="",
            ),
            SimpleNamespace(
                output=[{
                    "type": "function_call",
                    "name": "finish_discovery",
                    "call_id": "finish-1",
                    "arguments": json.dumps({"report": "No mutation was performed."}),
                }],
                output_text="",
            ),
        ])

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_discovery_agent(
                llm=llm,  # type: ignore[arg-type]
                rust=SimpleNamespace(),  # type: ignore[arg-type]
                workspace_root=root,
                task="Find the target without changing anything",
            )
            self.assertFalse((root / "unsafe.txt").exists())

        exposed = {tool["name"] for tool in llm.requests[0]["tools"]}
        self.assertEqual(exposed, DISCOVERY_TOOL_NAMES | {"finish_discovery"})
        self.assertTrue(result["isolated"])
        self.assertTrue(result["sequential"])
        self.assertFalse(result["background"])
        self.assertEqual(result["tool_policy"], "read_only_discovery")
        self.assertEqual(
            result["evidence"][0]["summary"],
            "tool_not_allowed_for_discovery_subagent",
        )

    def test_runner_rejects_a_second_active_subagent_instead_of_parallelizing(self) -> None:
        parent = ScriptedLLM([])
        runner = DiscoverySubagentRunner(
            parent_llm=parent,  # type: ignore[arg-type]
            rust_bin=Path("/tmp/agent-rust"),
            workspace_root=Path("/workspace"),
        )
        runner._run_lock.acquire()
        try:
            result = runner.run(task="search while another child is active")
        finally:
            runner._run_lock.release()

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "subagent_already_running")

    def test_each_discovery_run_gets_a_fresh_in_memory_session(self) -> None:
        first_llm = ScriptedLLM([
            SimpleNamespace(output=[], output_text="first report"),
        ])
        second_llm = ScriptedLLM([
            SimpleNamespace(output=[], output_text="second report"),
        ])

        with TemporaryDirectory() as tmp:
            first = run_discovery_agent(
                llm=first_llm,  # type: ignore[arg-type]
                rust=SimpleNamespace(),  # type: ignore[arg-type]
                workspace_root=Path(tmp),
                task="first task",
            )
            second = run_discovery_agent(
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
                discovery_runner=run_discovery,
            )
            registry = build_tool_registry(ctx)
            result = registry.execute(
                "discovery_subagent",
                {"task": "find entry points"},
                ctx,
            )
            order.append("parent-resumed")

        self.assertEqual(order, ["child-start", "child-finish", "parent-resumed"])
        self.assertTrue(result["complete"])


if __name__ == "__main__":
    unittest.main()
