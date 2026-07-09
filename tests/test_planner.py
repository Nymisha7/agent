from types import SimpleNamespace
from typing import Any

from nym_agent.language_servers import default_language_servers, language_server_context_text
from nym_agent.planner import AgentSession, _requires_workspace_evidence, _should_offer_tools, run_agent


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


class InspectingLLM:
    def __init__(self, inspect_path: str = "/home/pssintern/RA_publish") -> None:
        self.tool_choices: list[Any] = []
        self.tool_counts: list[int] = []
        self.calls = 0
        self.inspect_path = inspect_path

    def respond(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.tool_choices.append(kwargs.get("tool_choice"))
        self.tool_counts.append(len(kwargs.get("tools", [])))
        if self.calls == 1:
            return SimpleNamespace(
                output=[{
                    "type": "function_call",
                    "name": "inspect_tree",
                    "call_id": "inspect-1",
                    "arguments": '{"path":"' + self.inspect_path + '","max_files":20}',
                }],
                output_text="",
            )
        return SimpleNamespace(
            output=[{
                "type": "function_call",
                "name": "finish_task",
                "call_id": "finish-1",
                "arguments": '{"answer":"ok"}',
            }],
            output_text="",
        )


class FakeRust:
    def inspect_tree(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "root": str(kwargs["path"]), "tree": [], "files": []}


def test_project_questions_require_initial_workspace_evidence() -> None:
    llm = RecordingLLM()

    assert run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="tell me about the project",
    ) == "ok"

    assert llm.tool_choices == ["required"]


def test_conceptual_questions_do_not_force_workspace_evidence() -> None:
    assert _requires_workspace_evidence("what is a repository?") is False
    assert _requires_workspace_evidence("tell me about this repo") is True
    assert _requires_workspace_evidence("where is main.py?") is True


def test_non_actionable_prompt_skips_tools_and_returns_chat_response() -> None:
    llm = RecordingLLM()
    session = AgentSession(
        pending_action={
            "status": "unresolved",
            "action": "answer",
            "candidates": [
                {"path": "/workspace/AlphaSuiteRelease", "kind": "directory"},
                {"path": "/workspace/AlphaSuiteClean", "kind": "directory"},
            ],
        }
    )

    answer = run_agent(
        llm=llm,
        rust=object(),
        workspace_root=".",
        user_prompt="?",
        session=session,
    )

    assert answer == "How can I help?"
    assert llm.tool_counts == [0]
    assert llm.tool_choices == [None]
    assert session.pending_action is None


def test_ordinal_pending_selection_requires_tool_use() -> None:
    llm = InspectingLLM()
    session = AgentSession(
        pending_action={
            "status": "unresolved",
            "action": "answer",
            "candidates": [
                {"path": "/home/pssintern/RA_publish", "kind": "directory"},
                {"path": "/home/pssintern/RA_clean", "kind": "directory"},
                {"path": "/home/pssintern/RA2_clean", "kind": "directory"},
            ],
            "display_candidates": [
                {"path": "/home/pssintern/RA_clean", "kind": "directory"},
                {"path": "/home/pssintern/RA2_clean", "kind": "directory"},
                {"path": "/home/pssintern/RA_publish", "kind": "directory"},
            ],
        }
    )

    assert _should_offer_tools("third one", session) is True

    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root="/home/pssintern",
        user_prompt="third one",
        session=session,
    )

    assert answer == "ok"
    assert llm.tool_counts[0] > 0
    assert llm.tool_choices[0] == "required"
    assert session.pending_action is not None
    assert session.pending_action["selected"] == {"path": "/home/pssintern/RA_publish", "kind": "directory"}


def test_question_like_ordinal_typo_still_selects_pending_candidate() -> None:
    llm = InspectingLLM("/home/pssintern/RA_publish/rag_env")
    session = AgentSession(
        pending_action={
            "status": "unresolved",
            "action": "answer",
            "candidates": [
                {"path": "/home/pssintern/RA_publish", "kind": "directory"},
                {"path": "/home/pssintern/RA_clean", "kind": "directory"},
                {"path": "/home/pssintern/RA2_clean", "kind": "directory"},
                {"path": "/home/pssintern/RA_publish/rag_env", "kind": "directory"},
            ],
            "display_candidates": [
                {"path": "/home/pssintern/RA_publish", "kind": "directory"},
                {"path": "/home/pssintern/RA_clean", "kind": "directory"},
                {"path": "/home/pssintern/RA2_clean", "kind": "directory"},
                {"path": "/home/pssintern/RA_publish/rag_env", "kind": "directory"},
            ],
        }
    )

    assert _should_offer_tools("isnt 4rth one?", session) is True

    answer = run_agent(
        llm=llm,
        rust=FakeRust(),
        workspace_root="/home/pssintern",
        user_prompt="isnt 4rth one?",
        session=session,
    )

    assert answer == "ok"
    assert llm.tool_choices[0] == "required"
    assert session.pending_action is not None
    assert session.pending_action["selected"] == {
        "path": "/home/pssintern/RA_publish/rag_env",
        "kind": "directory",
    }


def test_continuation_prompts_require_tool_use() -> None:
    assert _requires_workspace_evidence("continue") is True
    assert _requires_workspace_evidence("go ahead and do it") is True


def test_language_server_requests_require_tool_use() -> None:
    assert _requires_workspace_evidence("start pyright") is True
    assert _requires_workspace_evidence("check rust-analyzer status") is True


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
