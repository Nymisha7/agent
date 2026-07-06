from types import SimpleNamespace
from typing import Any

from nym_agent.language_servers import default_language_servers, language_server_context_text
from nym_agent.planner import _requires_workspace_evidence, run_agent


class RecordingLLM:
    def __init__(self) -> None:
        self.tool_choices: list[Any] = []

    def respond(self, **kwargs: Any) -> Any:
        self.tool_choices.append(kwargs.get("tool_choice"))
        return SimpleNamespace(output=[], output_text="ok")


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
