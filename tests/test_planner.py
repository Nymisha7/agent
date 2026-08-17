from pathlib import Path
import json
import threading
import time
from types import SimpleNamespace
from typing import Any

from agent.language_servers import default_language_servers, language_server_context_text
from agent.planner import (
    AgentSession,
    ModelToolCall,
    _approval_was_approved,
    _bounded_recent_history,
    _local_file_mutation_intent_observation,
    run_agent,
)


def test_planner_has_no_commented_out_or_rule_router_functions() -> None:
    source = (Path(__file__).parents[1] / "agent" / "planner.py").read_text()

    assert "# def " not in source
    for removed_name in (
        "_requires_tool_use",
        "_request_frame",
        "_ordinal_selection",
        "_named_project_scope",
        "_should_offer_tools",
        "_preclassified_local_route",
        "_obvious_local_chat_prompt",
        "_local_gate_messages",
        "_local_tools_for_route",
        "LOCAL_GATE_PROMPT",
    ):
        assert removed_name not in source
    assert "what applications are open" not in source


def test_bounded_history_preserves_each_recent_turn() -> None:
    history = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "a" * 4_000},
        {"role": "user", "content": "open my spark"},
        {"role": "assistant", "content": "b" * 4_000},
    ]

    bounded = _bounded_recent_history(history, max_messages=4, max_chars=2_000)

    assert [item["role"] for item in bounded] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert bounded[0]["content"] == "first request"
    assert bounded[2]["content"] == "open my spark"
    assert sum(len(item["content"]) for item in bounded) <= 2_000


class RecordingLLM:
    def __init__(self) -> None:
        self.tool_choices: list[Any] = []
        self.tool_counts: list[int] = []
        self.messages: list[list[dict[str, Any]]] = []
        self.calls = 0

    def respond(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.tool_choices.append(kwargs.get("tool_choice"))
        self.tool_counts.append(len(kwargs.get("tools", [])))
        self.messages.append(list(kwargs.get("messages", [])))
        return SimpleNamespace(output=[], output_text="ok")


class FakeRust:
    def inspect_tree(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "root": str(kwargs["path"]), "tree": [], "files": []}


class PlainFinalAfterInspectionLLM:
    def __init__(self) -> None:
        self.calls = 0

    def respond(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(
                output=[{
                    "type": "function_call",
                    "name": "inspect_tree",
                    "call_id": "inspect-plain-final",
                    "arguments": '{"path":"/workspace","max_files":20}',
                }],
                output_text="",
            )
        return SimpleNamespace(
            output=[],
            output_text="The workspace exists, but I need the exact project path.",
        )


def test_model_receives_tools_without_regex_forcing_a_choice() -> None:
    llm = RecordingLLM()

    assert run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="tell me about the project",
    ) == "ok"

    assert llm.tool_choices == [None]
    assert llm.tool_counts[0] > 0


def test_plain_assistant_answer_after_tool_round_completes_turn() -> None:
    llm = PlainFinalAfterInspectionLLM()

    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root="/workspace",
        user_prompt="inspect the current project state",
    )

    assert answer == "The workspace exists, but I need the exact project path."
    assert llm.calls == 2


def test_conversational_prompt_still_leaves_tool_choice_to_model() -> None:
    llm = RecordingLLM()

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="?",
    )

    assert answer == "ok"
    assert llm.tool_counts[0] > 0
    assert llm.tool_choices == [None]


class LocalLLM:
    mode = "local"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def respond(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        response_index = len(self.requests) - 1
        if response_index < len(self.responses):
            response = self.responses[response_index]
            if isinstance(response, tuple):
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "finish_task",
                        "call_id": f"finish-{response_index}",
                        "arguments": json.dumps({"answer": response[1]}),
                    }],
                    output_text="",
                )
            return SimpleNamespace(output=[], output_text=response)
        return SimpleNamespace(
            output=[{
                "type": "function_call",
                "name": "finish_task",
                "call_id": "finish-local",
                "arguments": '{"answer":"tool path completed"}',
            }],
            output_text="",
        )


def test_local_conversation_uses_one_agent_call() -> None:
    llm = LocalLLM(["Hello!"])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="hi",
    )

    assert answer == "Hello!"
    assert len(llm.requests) == 1
    assert llm.requests[0]["tools"]
    assert llm.requests[0]["tool_choice"] is None


def test_local_invalid_json_tool_envelope_is_recovered_without_leaking() -> None:
    llm = LocalLLM([
        '`json\n{"name":"find_code_changes","arguments":{"user_input":"hi"}}\n`',
        "Hello!",
    ])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="hi",
    )

    assert answer == "Hello!"
    assert len(llm.requests) == 2
    assert "Available tool names:" in llm.requests[1]["messages"][-1]["content"]


def test_local_repeated_invalid_json_tool_envelope_returns_capability_error() -> None:
    invalid = '`\n{"name":"call_one_of_the_available_native_tools","arguments":{}}\n`'
    llm = LocalLLM([invalid, invalid])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="create an app",
    )

    assert "more capable model" in answer
    assert "call_one_of_the_available_native_tools" not in answer
    assert len(llm.requests) == 2


def test_local_tool_response_protocol_is_recovered_without_leaking() -> None:
    llm = LocalLLM([
        '<tool_response>{"path":"/workspace"}</tool_response>',
        "Hello!",
    ])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="hi",
    )

    assert answer == "Hello!"
    assert "tool_response" not in answer
    assert len(llm.requests) == 2


def test_local_fenced_response_envelope_is_unwrapped() -> None:
    llm = LocalLLM(['{}\n\n```json\n{"response":"Hello!"}\n```'])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="hi",
    )

    assert answer == "Hello!"
    assert len(llm.requests) == 1


def test_local_file_mutation_requires_explicit_current_request() -> None:
    call = ModelToolCall(
        name="write_file",
        call_id="write-hi",
        arguments={"path": "hi.txt", "content": "Hello"},
    )

    blocked = _local_file_mutation_intent_observation(call, "hi")
    allowed = _local_file_mutation_intent_observation(
        call,
        "create a task manager app named ctt",
    )

    assert blocked is not None
    assert blocked["reason"] == "file_mutation_intent_missing"
    assert allowed is None


def test_local_router_artifact_does_not_leak_to_user() -> None:
    llm = LocalLLM(["<|im_start|>"])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="hi",
    )

    assert answer == "I could not produce a final answer."
    assert len(llm.requests) == 1
    assert "<|im_start|>" not in answer


def test_model_template_artifacts_are_removed_from_history() -> None:
    llm = LocalLLM(["Clean turn."])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="hi again",
        conversation_history=[
            {"role": "user", "content": "can you open my file named calculator..?"},
            {"role": "assistant", "content": "<|im_start|>"},
        ],
    )

    assert answer == "Clean turn."
    assert "<|im_start|>" not in str(llm.requests[-1]["messages"])


def test_local_agent_retries_fenced_tool_json_as_tool_call() -> None:
    class JsonThenToolLLM:
        mode = "local"

        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def respond(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(
                    output=[],
                    output_text=(
                        "```json\n"
                        '{"name":"inspect_tree","arguments":{"path":"/workspace"}}\n'
                        "```"
                    ),
                )
            if len(self.requests) == 2:
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "inspect_tree",
                        "call_id": "inspect-after-fence",
                        "arguments": '{"path":"/workspace","max_files":20}',
                    }],
                    output_text="",
                )
            return SimpleNamespace(output=[], output_text="inspected")

    llm = JsonThenToolLLM()

    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root="/workspace",
        user_prompt="inspect my project",
    )

    assert answer == "inspected"
    assert len(llm.requests) == 3
    assert all(request["tools"] for request in llm.requests)
    assert any(
        "not an executed action" in str(message.get("content", ""))
        for message in llm.requests[1]["messages"]
    )


def test_local_agent_accepts_unstructured_completion() -> None:
    class ProseThenToolLLM:
        mode = "local"

        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def respond(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(output=[], output_text="workspace")
            if len(self.requests) == 2:
                return SimpleNamespace(output=[], output_text="You can run the requested action yourself.")
            if len(self.requests) == 3:
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "inspect_tree",
                        "call_id": "inspect-after-prose",
                        "arguments": '{"path":"/workspace","max_files":20}',
                    }],
                    output_text="",
                )
            return SimpleNamespace(
                output=[{
                    "type": "function_call",
                    "name": "finish_task",
                    "call_id": "finish-after-prose",
                    "arguments": '{"answer":"tool path completed"}',
                }],
                output_text="",
            )

    llm = ProseThenToolLLM()

    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root="/workspace",
        user_prompt="do the requested work",
    )

    assert answer == "workspace"
    assert len(llm.requests) == 1


def test_local_agent_does_not_route_prose_with_keyword_rules() -> None:
    class ProseThenToolLLM:
        mode = "local"

        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def respond(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(output=[], output_text="workspace")
            if len(self.requests) == 2:
                return SimpleNamespace(
                    output=[],
                    output_text="The workspace probably needs checking before I can answer.",
                )
            if len(self.requests) == 3:
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "inspect_tree",
                        "call_id": "inspect-after-plan",
                        "arguments": '{"path":"/workspace","max_files":20}',
                    }],
                    output_text="",
                )
            return SimpleNamespace(
                output=[{
                    "type": "function_call",
                    "name": "finish_task",
                    "call_id": "finish-after-plan",
                    "arguments": '{"answer":"inspected"}',
                }],
                output_text="",
            )

    llm = ProseThenToolLLM()

    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root="/workspace",
        user_prompt="tell me about my project",
    )

    assert answer == "workspace"
    assert len(llm.requests) == 1


def test_unadvertised_finish_task_is_not_special_cased() -> None:
    class FinishThenToolLLM:
        mode = "local"

        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def respond(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(output=[], output_text="workspace")
            if len(self.requests) == 2:
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "finish_task",
                        "call_id": "finish-too-early",
                        "arguments": '{"answer":"I will inspect it."}',
                    }],
                    output_text="",
                )
            if len(self.requests) == 3:
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "inspect_tree",
                        "call_id": "inspect-after-reject",
                        "arguments": '{"path":"/workspace","max_files":20}',
                    }],
                    output_text="",
                )
            return SimpleNamespace(
                output=[{
                    "type": "function_call",
                    "name": "finish_task",
                    "call_id": "finish-after-reject",
                    "arguments": '{"answer":"inspected"}',
                }],
                output_text="",
            )

    llm = FinishThenToolLLM()

    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root="/workspace",
        user_prompt="tell me about my project",
    )

    assert answer == "workspace"
    assert len(llm.requests) == 1


def test_parallel_read_tools_overlap_before_returning_to_model() -> None:
    class TwoReadsThenFinishLLM:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    output=[
                        {
                            "type": "function_call",
                            "name": "system_info",
                            "call_id": "system-info",
                            "arguments": '{}',
                        },
                        {
                            "type": "function_call",
                            "name": "connected_devices",
                            "call_id": "connected-devices",
                            "arguments": '{"scope":"all"}',
                        },
                    ],
                    output_text="",
                )
            return SimpleNamespace(output=[], output_text="done")

    class SlowRust:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def _observe(self) -> None:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            with self.lock:
                self.active -= 1

        def system_info(self) -> dict[str, Any]:
            self._observe()
            return {"ok": True, "tool": "system_info"}

        def connected_devices(self, **kwargs: Any) -> dict[str, Any]:
            self._observe()
            return {"ok": True, "tool": "connected_devices", "scope": kwargs["scope"]}

    rust = SlowRust()

    answer = run_agent(
        llm=TwoReadsThenFinishLLM(),
        rust=rust,
        workspace_root="/workspace",
        user_prompt="inspect both projects",
    )

    assert answer == "done"
    assert rust.max_active == 2


def test_local_agent_routes_to_compact_tools_with_top_level_descriptions() -> None:
    class InspectThenFinishLLM:
        mode = "local"

        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def respond(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "inspect_tree",
                        "call_id": "inspect-local-tools",
                        "arguments": '{"path":"/workspace","max_files":20}',
                    }],
                    output_text="",
                )
            return SimpleNamespace(output=[], output_text="tool path completed")

    llm = InspectThenFinishLLM()

    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root=".",
        user_prompt="inspect my project",
    )

    assert answer == "tool path completed"
    assert len(llm.requests) == 2
    assert len(llm.requests[0]["tools"]) > 0
    assert llm.requests[0]["tool_choice"] is None
    assert all(tool.get("description") for tool in llm.requests[0]["tools"])
    tool_names = {tool["name"] for tool in llm.requests[0]["tools"]}
    assert "inspect_target" in tool_names
    assert "connected_devices" in tool_names
    followup_tool_names = {tool["name"] for tool in llm.requests[1]["tools"]}
    assert "inspect_target" in followup_tool_names
    assert "connected_devices" in followup_tool_names
    assert all(tool.get("description") for tool in llm.requests[1]["tools"])


def test_desktop_open_intent_runs_without_approval() -> None:
    class SemanticLLM:
        mode = "local"

        def __init__(self) -> None:
            self.calls = 0

        def respond(self, **_kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "desktop_action",
                        "call_id": "launch-semantic-target",
                        "arguments": '{"action":"launch_application","target":"spark"}',
                    }],
                    output_text="",
                )
            return SimpleNamespace(output=[], output_text="Spark opened.")

    class FakeRust:
        def desktop_resolve(self, *, query: str, kind: str, limit: int) -> dict[str, Any]:
            return {
                "ok": True,
                "tool": "desktop_resolve",
                "query": query,
                "kind": kind,
                "ambiguous": False,
                "candidates": [{
                    "kind": "application",
                    "id": "spark.desktop",
                    "target": "spark.desktop",
                    "name": "Spark",
                }],
            }

        def desktop_action(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "verified": True, **kwargs}

    session = AgentSession()
    llm = SemanticLLM()
    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root=".",
        user_prompt="can you open my spark app",
        session=session,
    )

    assert answer == "Spark opened."
    assert llm.calls == 2
    assert not session.pending_approvals


def test_desktop_open_intent_focuses_existing_window_without_approval() -> None:
    class FocusLLM:
        mode = "local"

        def __init__(self) -> None:
            self.calls = 0

        def respond(self, **_kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(output=[{
                    "type": "function_call",
                    "name": "desktop_action",
                    "call_id": "focus-existing-window",
                    "arguments": '{"action":"focus_window","target":"0x3a00007"}',
                }], output_text="")
            return SimpleNamespace(output=[], output_text="Spark is ready.")

    class FakeRust:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def desktop_action(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(dict(kwargs))
            return {"ok": True, "verified": True, **kwargs}

    rust = FakeRust()
    session = AgentSession(desktop_targets=[{
        "kind": "window",
        "id": "0x3a00007",
        "target": "0x3a00007",
        "title": "Spark",
    }])
    approvals: list[dict[str, Any]] = []

    answer = run_agent(
        llm=FocusLLM(),
        rust=rust,
        workspace_root=".",
        user_prompt="open Spark",
        session=session,
        approval_requester=lambda request: approvals.append(dict(request)) or "approved",
    )

    assert answer == "Spark is ready."
    assert rust.calls == [{"action": "focus_window", "target": "0x3a00007"}]
    assert not approvals
    assert not session.pending_approvals


def test_file_open_request_does_not_use_desktop_fast_path() -> None:
    class ChatLLM:
        mode = "local"

        def __init__(self) -> None:
            self.calls = 0

        def respond(self, **_kwargs: Any) -> Any:
            self.calls += 1
            return SimpleNamespace(output=[], output_text="Need a file path.")

    llm = ChatLLM()
    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="open my file named calculator",
    )

    assert answer == "Need a file path."
    assert llm.calls == 1


def test_keep_it_open_clarification_does_not_use_desktop_open_fast_path() -> None:
    class ChatLLM:
        mode = "local"

        def __init__(self) -> None:
            self.calls = 0

        def respond(self, **_kwargs: Any) -> Any:
            self.calls += 1
            return SimpleNamespace(output=[], output_text="Continuing previous close request.")

    llm = ChatLLM()
    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="cli keep it open and vscode",
    )

    assert answer == "Continuing previous close request."
    assert llm.calls == 1


def test_close_window_intent_runs_without_approval() -> None:
    class CloseWindowLLM:
        mode = "local"

        def __init__(self) -> None:
            self.calls = 0

        def respond(self, **_kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(output=[{
                    "type": "function_call",
                    "name": "desktop_action",
                    "call_id": "close-window",
                    "arguments": json.dumps({
                        "action": "close_window",
                        "target": "0xb0b2e",
                    }),
                }], output_text="")
            return SimpleNamespace(output=[], output_text="Notepad closed.")

    class FakeRust:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def desktop_action(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(dict(kwargs))
            return {"ok": True, "verified": True, "tool": "desktop_action", **kwargs}

    rust = FakeRust()
    session = AgentSession(desktop_targets=[{
        "kind": "window",
        "id": "0xb0b2e",
        "target": "0xb0b2e",
        "title": "Notepad",
    }])
    requests: list[dict[str, Any]] = []

    answer = run_agent(
        llm=CloseWindowLLM(),
        rust=rust,
        workspace_root=".",
        user_prompt="close Notepad",
        session=session,
        approval_requester=lambda request: requests.append(dict(request)) or "approved",
    )

    assert answer == "Notepad closed."
    assert requests == []
    assert rust.calls == [{"action": "close_window", "target": "0xb0b2e", "value": None}]
    assert not session.pending_approvals


def test_unverified_close_cannot_be_reported_as_success() -> None:
    class CloseWindowLLM:
        mode = "local"

        def __init__(self) -> None:
            self.calls = 0

        def respond(self, **_kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(output=[{
                    "type": "function_call",
                    "name": "desktop_action",
                    "call_id": "close-browser",
                    "arguments": json.dumps({
                        "action": "close_window",
                        "target": "0xb0b2e",
                    }),
                }], output_text="")
            if self.calls == 2:
                return SimpleNamespace(output=[], output_text="Chrome closed.")
            return SimpleNamespace(
                output=[],
                output_text="I could not confirm that Chrome closed; the window is still visible.",
            )

    class FakeRust:
        def desktop_action(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "verified": False,
                "verification": "not_confirmed",
                "tool": "desktop_action",
                **kwargs,
            }

    llm = CloseWindowLLM()
    session = AgentSession(desktop_targets=[{
        "kind": "window",
        "id": "0xb0b2e",
        "target": "0xb0b2e",
        "title": "Chrome",
    }])

    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root=".",
        user_prompt="close Chrome",
        session=session,
    )

    assert answer == "I could not confirm that Chrome closed; the window is still visible."
    assert llm.calls == 3
    assert session.desktop_targets[-1].get("action") != "close_window"


def test_close_application_approval_does_not_cover_another_process() -> None:
    session = AgentSession(desktop_targets=[
        {"kind": "window", "id": "0x1", "target": "0x1", "name": "editor"},
        {"kind": "window", "id": "0x2", "target": "0x2", "name": "browser"},
    ])
    approved = {
        "tool": "desktop_action",
        "operation": "desktop",
        "prompt": "close the applications",
        "status": "approved",
        "requested_path": "desktop close_window 0x1",
        "args": {"action": "close_window", "target": "0x1"},
    }
    session.pending_approvals = [approved]
    requested = {
        "tool": "desktop_action",
        "operation": "desktop",
        "prompt": "close the applications",
        "requested_path": "desktop close_window 0x2",
        "args": {"action": "close_window", "target": "0x2"},
    }

    assert not _approval_was_approved(session, requested)


def test_local_agent_retries_printed_command_json_without_executing_it() -> None:
    class JsonThenToolLLM:
        mode = "local"

        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def respond(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(
                    output=[],
                    output_text='{"command":"read_directory","directory_path":"/workspace"}',
                )
            if len(self.requests) == 2:
                return SimpleNamespace(
                    output=[{
                        "type": "function_call",
                        "name": "inspect_tree",
                        "call_id": "inspect-after-json",
                        "arguments": '{"path":"/workspace","max_files":20}',
                    }],
                    output_text="",
                )
            return SimpleNamespace(output=[], output_text="tool path completed")

    llm = JsonThenToolLLM()
    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root=".",
        user_prompt="inspect my project",
    )

    assert answer == "tool path completed"
    assert len(llm.requests) == 3
    assert any(
        "not an executed action" in str(message.get("content", ""))
        for message in llm.requests[1]["messages"]
        if isinstance(message, dict)
    )


def test_default_language_server_catalog_covers_requested_languages() -> None:
    servers = default_language_servers()
    names = {spec.server for spec in servers}
    languages = {spec.language for spec in servers}

    assert {"pyright", "clangd", "eclipse.jdt.ls", "tsserver", "gopls", "rust-analyzer"} <= names
    assert {"Python", "C/C++", "Java", "JavaScript/TypeScript", "Go", "Rust"} <= languages


def test_language_server_context_mentions_the_catalog() -> None:
    text = language_server_context_text()

    assert "Preferred language servers" in text
    assert "pyright" in text
    assert "rust-analyzer" in text
