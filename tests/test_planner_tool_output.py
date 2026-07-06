import json

from nym_agent.planner import _prepare_tool_output


def test_prepare_tool_output_truncates_large_content() -> None:
    observation = {
        "ok": True,
        "path": "README.md",
        "content": "x" * 20_000,
        "entries": ["a" * 500 for _ in range(30)],
    }

    payload = _prepare_tool_output(observation, max_bytes=4_000)

    assert len(payload) <= 4_000
    assert "truncated" in payload.lower()
    assert "README.md" in payload
