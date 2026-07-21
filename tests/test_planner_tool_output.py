import json

from agent.planner import _prepare_event_observation, _prepare_tool_output


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


def test_prepare_event_observation_bounds_large_tool_results() -> None:
    observation = {
        "path": "/workspace",
        "tree": [
            {"path": f"project/file_{index}.txt", "content": "x" * 2_000}
            for index in range(200)
        ],
    }

    compact = _prepare_event_observation(observation, max_bytes=8_000)

    assert compact["event_observation_truncated"] is True
    assert compact["original_bytes"] > 8_000
    assert len(str(compact)) < 12_000
