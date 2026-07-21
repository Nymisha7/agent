from pathlib import Path
import json
from types import SimpleNamespace
from typing import Any

from agent.language_servers import default_language_servers, language_server_context_text
from agent.planner import run_agent


def test_planner_has_no_commented_out_or_rule_router_functions() -> None:
    source = (Path(__file__).parents[1] / "agent" / "planner.py").read_text()

    assert "# def " not in source
    for removed_name in (
        "_requires_tool_use",
        "_request_frame",
        "_ordinal_selection",
        "_named_project_scope",
        "_should_offer_tools",
    ):
        assert removed_name not in source


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
        if self.calls == 1:
            if not kwargs.get("tools"):
                return SimpleNamespace(output=[], output_text="How can I help?")
            return SimpleNamespace(
                output=[{
                    "type": "function_call",
                    "name": "finish_task",
                    "call_id": "finish-1",
                    "arguments": '{"answer":"ok"}',
                }],
                output_text="",
            )
        return SimpleNamespace(output=[], output_text="")


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
        user_prompt="has my project been deleted?",
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


class LocalGateLLM:
    mode = "local"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def respond(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        response_index = len(self.requests) - 1
        if response_index < len(self.responses):
            return SimpleNamespace(output=[], output_text=self.responses[response_index])
        return SimpleNamespace(
            output=[{
                "type": "function_call",
                "name": "finish_task",
                "call_id": "finish-local",
                "arguments": '{"answer":"tool path completed"}',
            }],
            output_text="",
        )


def test_local_conversation_gate_keeps_simple_chat_tool_free() -> None:
    llm = LocalGateLLM(["CHAT", "Hello!"])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="hi",
    )

    assert answer == "Hello!"
    assert len(llm.requests) == 2
    assert llm.requests[0]["tools"] == []
    assert llm.requests[1]["tools"] == []


def test_local_chat_retries_fenced_tool_json_as_normal_text() -> None:
    llm = LocalGateLLM([
        "CHAT",
        (
            "```json\n"
            '{"name":"secret_scan","arguments":{"path":"/workspace"}}\n'
            "```"
        ),
        "I cannot retrieve live weather without a weather source.",
    ])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="what is the weather today?",
    )

    assert answer == "I cannot retrieve live weather without a weather source."
    assert len(llm.requests) == 3
    assert all(request["tools"] == [] for request in llm.requests)
    assert "Do not print JSON" in llm.requests[-1]["instructions"]


def test_local_gate_escalates_tool_work_with_compact_schemas() -> None:
    llm = LocalGateLLM(["WORKSPACE"])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="inspect my project",
    )

    assert answer == "tool path completed"
    assert len(llm.requests) == 2
    assert llm.requests[0]["tools"] == []
    assert len(llm.requests[1]["tools"]) > 0
    assert len(json.dumps(llm.requests[1]["tools"])) < 7_000
    assert "description" not in json.dumps(llm.requests[1]["tools"])
    tool_names = {tool["name"] for tool in llm.requests[1]["tools"]}
    assert "inspect_target" in tool_names
    assert "connected_devices" not in tool_names


def test_local_agent_retries_printed_command_json_without_executing_it() -> None:
    llm = LocalGateLLM([
        "WORKSPACE",
        '{"command":"read_directory","directory_path":"/workspace"}',
    ])

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="inspect my project",
    )

    assert answer == "tool path completed"
    assert len(llm.requests) == 3
    correction = llm.requests[2]["messages"][-1]["content"]
    assert "not an executed action" in correction


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
