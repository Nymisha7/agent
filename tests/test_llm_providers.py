import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from agent.attachments import import_attachment
from agent.llm import (
    LLMClient,
    _anthropic_message_to_response,
    _chat_reasoning_effort,
    _chat_completion_to_response,
    _default_model_for_provider,
    _default_reasoning_effort,
    _default_reasoning_summary,
    _friendly_llm_error,
    _image_data_url,
    _model_supports_reasoning,
    _normalize_provider,
    _responses_reasoning_config,
    _responses_messages_to_anthropic,
    _responses_messages_to_chat,
    _responses_input_messages,
    _responses_tools_to_anthropic,
    _responses_tools_to_chat,
)


class LLMProviderTests(unittest.TestCase):
    def test_provider_aliases_normalize(self) -> None:
        self.assertEqual(_normalize_provider("openai"), "openai")
        self.assertEqual(_normalize_provider("compatible"), "openai-compatible")
        self.assertEqual(_normalize_provider("lm-studio"), "lmstudio")
        self.assertEqual(_normalize_provider("llama.cpp"), "llamacpp")
        self.assertEqual(_normalize_provider("v-llm"), "vllm")
        self.assertEqual(_normalize_provider("local-ai"), "localai")
        self.assertEqual(_normalize_provider("claude"), "anthropic")
        self.assertEqual(_normalize_provider("deepseek-ai"), "deepseek")
        self.assertEqual(_normalize_provider("zai"), "glm")
        self.assertEqual(_normalize_provider("zhipuai"), "glm")
        self.assertEqual(_normalize_provider("mini-max"), "minimax")

    def test_provider_defaults_cover_local_free_models(self) -> None:
        self.assertEqual(_default_model_for_provider("deepseek"), "deepseek-v4-flash")
        self.assertEqual(_default_model_for_provider("glm"), "glm-4")
        self.assertEqual(_default_model_for_provider("minimax"), "MiniMax-M3")
        self.assertEqual(_default_model_for_provider("ollama"), "llama3.1")
        self.assertEqual(_default_model_for_provider("llamacpp"), "local-model")
        self.assertEqual(_default_model_for_provider("vllm"), "local-model")
        self.assertEqual(_default_model_for_provider("localai"), "local-model")

    def test_minimax_uses_openai_compatible_hosted_transport(self) -> None:
        with patch.dict("os.environ", {"MINIMAX_API_KEY": "minimax-secret"}, clear=True):
            client = LLMClient(provider="minimax")

        self.assertEqual(client.model, "MiniMax-M3")
        self.assertEqual(client.endpoint, "https://api.minimax.io/v1")
        self.assertEqual(client.mode, "hosted")
        self.assertIsNone(client.configuration_error)
        self.assertIsNotNone(client.client)

    def test_minimax_requires_its_own_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(provider="minimax")

        self.assertEqual(client.configuration_state, "api_key_required")
        self.assertIn("MINIMAX_API_KEY", client.configuration_error or "")

    def test_open_source_local_providers_need_no_login_or_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            clients = {
                provider: LLMClient(provider=provider)
                for provider in ("ollama", "lmstudio", "llamacpp", "vllm", "localai")
            }

        for client in clients.values():
            self.assertEqual(client.mode, "local")
            self.assertIsNone(client.configuration_error)
            self.assertEqual(client.configuration_state, "ready")
            self.assertIsNotNone(client.client)

    def test_local_provider_streams_text_and_tool_activity(self) -> None:
        client = LLMClient(provider="ollama", model="qwen2.5-coder")
        chunks = iter([
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content="Checking ", reasoning_content="private",
                    tool_calls=[],
                ))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content="files", reasoning_content=None,
                    tool_calls=[],
                ))],
                usage=None,
            ),
        ])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=chunks)),
            ),
        )
        events = []

        response = client.respond(
            instructions="Be useful",
            messages=[{"role": "user", "content": "Inspect"}],
            tools=[],
            stream=True,
            event_handler=events.append,
        )

        self.assertEqual(response.output_text, "Checking files")
        self.assertEqual(events[0]["type"], "response.in_progress")
        self.assertTrue(any(event["type"] == "response.output_text.delta" for event in events))
        self.assertFalse(any(event["type"] == "response.reasoning_text.delta" for event in events))
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_local_provider_streams_safe_text_before_tool_capable_response_finishes(self) -> None:
        client = LLMClient(provider="ollama", model="qwen2.5-coder:3b")
        events = []

        class StreamingChunks:
            def __init__(self) -> None:
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index == 0:
                    self.index += 1
                    return SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(
                            content="Hello",
                            reasoning_content=None,
                            tool_calls=[],
                        ))],
                        usage=None,
                    )
                if self.index == 1:
                    self.index += 1
                    self.assert_first_delta_was_emitted()
                    return SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(
                            content=" there.",
                            reasoning_content=None,
                            tool_calls=[],
                        ))],
                        usage=None,
                    )
                raise StopIteration

            @staticmethod
            def assert_first_delta_was_emitted() -> None:
                deltas = [
                    event["delta"]
                    for event in events
                    if event["type"] == "response.output_text.delta"
                ]
                if deltas != ["Hello"]:
                    raise AssertionError(f"expected progressive text, got {deltas!r}")

        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=StreamingChunks())),
            ),
        )

        response = client.respond(
            instructions="Use tools when needed",
            messages=[{"role": "user", "content": "Say hello"}],
            tools=[{
                "type": "function",
                "name": "desktop_action",
                "parameters": {"type": "object", "properties": {}},
            }],
            stream=True,
            event_handler=events.append,
        )

        self.assertEqual(response.output_text, "Hello there.")
        self.assertEqual(
            [
                event["delta"]
                for event in events
                if event["type"] == "response.output_text.delta"
            ],
            ["Hello", " there."],
        )

    def test_local_provider_holds_fragmented_plain_text_tool_call(self) -> None:
        client = LLMClient(provider="ollama", model="qwen2.5-coder:3b")
        chunks = iter([
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content="desktop_",
                    reasoning_content=None,
                    tool_calls=[],
                ))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content='action {"action":"set_mute","value":true}',
                    reasoning_content=None,
                    tool_calls=[],
                ))],
                usage=None,
            ),
        ])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=chunks)),
            ),
        )
        events = []

        response = client.respond(
            instructions="Use tools",
            messages=[{"role": "user", "content": "Mute audio"}],
            tools=[{
                "type": "function",
                "name": "desktop_action",
                "parameters": {"type": "object", "properties": {}},
            }],
            stream=True,
            event_handler=events.append,
        )

        self.assertEqual(response.output_text, "")
        self.assertEqual(response.output[0]["name"], "desktop_action")
        self.assertFalse(any(
            event["type"] == "response.output_text.delta"
            for event in events
        ))

    def test_ollama_warm_preloads_model_with_native_api(self) -> None:
        client = LLMClient(provider="ollama", model="qwen2.5-coder:3b")
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"done":true}'

        with (
            patch.dict("os.environ", {"AGENT_OLLAMA_KEEP_ALIVE": "20m"}, clear=False),
            patch("agent.llm.urllib.request.urlopen", return_value=response) as urlopen,
        ):
            client.warm()

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/generate")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {
                "model": "qwen2.5-coder:3b",
                "stream": False,
                "keep_alive": "20m",
            },
        )

    def test_local_provider_converts_bare_json_content_into_tool_call(self) -> None:
        client = LLMClient(provider="ollama", model="qwen2.5-coder:7b")
        chunks = iter([
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content=(
                        '{"name":"desktop_action","arguments":'
                        '{"action":"set_volume","value":"0"}}'
                    ),
                    reasoning_content=None,
                    tool_calls=[],
                ))],
                usage=None,
            ),
        ])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=chunks)),
            ),
        )
        events = []

        response = client.respond(
            instructions="Use tools",
            messages=[{"role": "user", "content": "Mute audio"}],
            tools=[{
                "type": "function",
                "name": "desktop_action",
                "parameters": {"type": "object", "properties": {}},
            }],
            stream=True,
            event_handler=events.append,
        )

        self.assertEqual(response.output_text, "")
        self.assertEqual(response.output[0]["name"], "desktop_action")
        self.assertEqual(
            response.output[0]["arguments"],
            '{"action": "set_volume", "value": "0"}',
        )
        self.assertTrue(any(event["type"] == "response.output_item.added" for event in events))
        self.assertFalse(any(
            event["type"] == "response.output_text.delta"
            and "desktop_action" in event.get("delta", "")
            for event in events
        ))

    def test_local_provider_converts_fenced_json_into_tool_call(self) -> None:
        client = LLMClient(provider="ollama", model="qwen2.5-coder:7b")
        chunks = iter([
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content=(
                        "```json\n"
                        '{"name":"desktop_action","arguments":'
                        '{"action":"set_volume","value":"0"}}\n'
                        "```\nThis command will mute audio."
                    ),
                    reasoning_content=None,
                    tool_calls=[],
                ))],
                usage=None,
            ),
        ])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=chunks)),
            ),
        )
        events = []

        response = client.respond(
            instructions="Use tools",
            messages=[{"role": "user", "content": "Mute audio"}],
            tools=[{
                "type": "function",
                "name": "desktop_action",
                "parameters": {"type": "object", "properties": {}},
            }],
            stream=True,
            event_handler=events.append,
        )

        self.assertEqual(response.output_text, "")
        self.assertEqual(response.output[0]["name"], "desktop_action")
        self.assertFalse(any(
            event["type"] == "response.output_text.delta"
            for event in events
        ))

    def test_local_chat_does_not_stream_unexecuted_fenced_tool_json(self) -> None:
        client = LLMClient(provider="ollama", model="qwen2.5-coder:7b")
        leaked = (
            "Try this:\n```json\n"
            '{"name":"secret_scan","arguments":{"path":"/workspace"}}\n'
            "```"
        )
        chunks = iter([
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content=leaked,
                    reasoning_content=None,
                    tool_calls=[],
                ))],
                usage=None,
            ),
        ])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=chunks)),
            ),
        )
        events = []

        response = client.respond(
            instructions="Answer normally",
            messages=[{"role": "user", "content": "What is the weather?"}],
            tools=[],
            stream=True,
            event_handler=events.append,
        )

        self.assertEqual(response.output_text, leaked)
        self.assertFalse(any(
            event["type"] == "response.output_text.delta"
            for event in events
        ))

    def test_local_no_tool_chat_streams_safe_text_immediately(self) -> None:
        client = LLMClient(provider="ollama", model="qwen2.5-coder:7b")
        chunks = iter([
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content="Hi",
                    reasoning_content=None,
                    tool_calls=[],
                ))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content=" there.",
                    reasoning_content=None,
                    tool_calls=[],
                ))],
                usage=None,
            ),
        ])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=chunks)),
            ),
        )
        events = []

        response = client.respond(
            instructions="Answer normally",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            stream=True,
            event_handler=events.append,
        )

        self.assertEqual(response.output_text, "Hi there.")
        self.assertEqual(
            [
                event["delta"]
                for event in events
                if event["type"] == "response.output_text.delta"
            ],
            ["Hi", " there."],
        )

    def test_local_provider_does_not_execute_unknown_json_tool_name(self) -> None:
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"name":"not_registered","arguments":{}}',
                tool_calls=[],
            ))],
            usage=None,
        )

        response = _chat_completion_to_response(
            completion,
            text_tool_names={"desktop_action"},
        )

        self.assertEqual(response.output, [])
        self.assertIn("not_registered", response.output_text)

    def test_local_provider_executes_plain_text_tool_calls(self) -> None:
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=(
                    'inspect_target {"path":"/workspace/app","kind":"directory"}'
                    'inspect_target {"path":"/workspace/app/src","kind":"directory"}'
                ),
                tool_calls=[],
            ))],
            usage=None,
        )

        response = _chat_completion_to_response(
            completion,
            text_tool_names={"inspect_target"},
        )

        self.assertEqual(
            [item["name"] for item in response.output],
            ["inspect_target", "inspect_target"],
        )
        self.assertEqual(response.output_text, "")

    def test_local_provider_hides_unexecuted_command_envelope_from_stream(self) -> None:
        client = LLMClient(provider="ollama", model="qwen2.5-coder:7b")
        chunks = iter([
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(
                    content=(
                        '{"command":"read_directory",'
                        '"directory_path":"/workspace"}'
                    ),
                    reasoning_content=None,
                    tool_calls=[],
                ))],
                usage=None,
            ),
        ])
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=chunks)),
            ),
        )
        events = []

        response = client.respond(
            instructions="Use native tools",
            messages=[{"role": "user", "content": "Read the directory"}],
            tools=[{
                "type": "function",
                "name": "inspect_tree",
                "parameters": {"type": "object", "properties": {}},
            }],
            stream=True,
            event_handler=events.append,
        )

        self.assertIn("read_directory", response.output_text)
        self.assertFalse(any(
            event["type"] == "response.output_text.delta"
            for event in events
        ))

    def test_reasoning_capability_detection_is_provider_specific(self) -> None:
        self.assertTrue(_model_supports_reasoning("openai", "gpt-5.4-mini"))
        self.assertTrue(_model_supports_reasoning("deepseek", "deepseek-reasoner"))
        self.assertFalse(_model_supports_reasoning("openai", "gpt-4o"))
        self.assertFalse(_model_supports_reasoning("anthropic", "claude-3-5-haiku-latest"))

    def test_reasoning_defaults_apply_only_to_reasoning_models(self) -> None:
        self.assertEqual(_default_reasoning_effort("openai", "gpt-5.4-mini"), "medium")
        self.assertEqual(_default_reasoning_summary("openai", "gpt-5.4-mini"), "auto")
        self.assertIsNone(_default_reasoning_effort("openai", "gpt-4o"))
        self.assertIsNone(_default_reasoning_summary("deepseek", "deepseek-reasoner"))

    def test_openai_responses_reasoning_config_matches_sdk_shape(self) -> None:
        self.assertEqual(
            _responses_reasoning_config("openai", "gpt-5.4-mini", "high", "auto"),
            {"effort": "high", "summary": "auto"},
        )
        self.assertIsNone(_responses_reasoning_config("openai", "gpt-4o", "high", "auto"))

    def test_chat_reasoning_effort_applies_only_to_reasoning_chat_models(self) -> None:
        self.assertEqual(
            _chat_reasoning_effort("deepseek", "deepseek-reasoner", "low"),
            "low",
        )
        self.assertIsNone(_chat_reasoning_effort("deepseek", "deepseek-chat", "low"))

    def test_openai_compatible_clients_have_default_timeout(self) -> None:
        with patch("agent.llm.OpenAI") as openai:
            LLMClient(provider="ollama", model="qwen2.5-coder:7b")

        self.assertEqual(openai.call_args.kwargs["timeout"], 60.0)

    def test_llm_timeout_env_overrides_default(self) -> None:
        with patch.dict("os.environ", {"AGENT_LLM_TIMEOUT_SECONDS": "7"}, clear=True):
            with patch("agent.llm.OpenAI") as openai:
                LLMClient(provider="ollama", model="qwen2.5-coder:7b")

        self.assertEqual(openai.call_args.kwargs["timeout"], 7.0)

    def test_missing_deepseek_key_is_deferred_until_request(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(provider="deepseek")

        self.assertIsNotNone(client.configuration_error)
        self.assertEqual(client.configuration_state, "api_key_required")
        with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
            client.respond(instructions="", messages=[{"role": "user", "content": "hi"}], tools=[])

    def test_missing_openai_key_is_deferred_until_request(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(provider="openai")

        self.assertIsNotNone(client.configuration_error)
        self.assertEqual(client.configuration_state, "api_key_required")
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            client.respond(instructions="", messages=[{"role": "user", "content": "hi"}], tools=[])

    def test_friendly_error_explains_missing_local_model(self) -> None:
        message = _friendly_llm_error(
            Exception("model 'deepseek-chat' not found"),
            "deepseek",
            "deepseek-chat",
            "http://localhost:11434/v1",
            "local",
        )

        self.assertIn("Local model `deepseek-chat` is not available", message)
        self.assertIn("/model <model>", message)

    def test_responses_messages_convert_to_chat_tool_messages(self) -> None:
        messages = [
            {"role": "user", "content": "Find it"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "glob",
                "arguments": '{"pattern":"*.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": '{"matches":[]}',
            },
        ]

        converted = _responses_messages_to_chat(messages, "system prompt")

        self.assertEqual(converted[0], {"role": "system", "content": "system prompt"})
        self.assertEqual(converted[1], {"role": "user", "content": "Find it"})
        self.assertEqual(converted[2]["tool_calls"][0]["function"]["name"], "glob")
        self.assertEqual(converted[3]["role"], "tool")
        self.assertEqual(converted[3]["tool_call_id"], "call-1")

    def test_responses_input_preserves_plain_text_messages(self) -> None:
        messages = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "second message"},
        ]

        self.assertEqual(
            _responses_input_messages(messages, provider="openai"),
            messages,
        )

    def test_text_attachment_content_is_sent_to_the_model(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "finalreport.txt"
            path.write_text("Release status: ready for review.", encoding="utf-8")
            converted = _responses_input_messages([
                {
                    "role": "user",
                    "content": "Summarize the attachment.",
                    "attachments": [{
                        "filename": path.name,
                        "mime": "text/plain",
                        "size_bytes": path.stat().st_size,
                        "storage_path": str(path),
                    }],
                }
            ], provider="openai")

        parts = converted[0]["content"]
        self.assertEqual(parts[0]["type"], "input_text")
        self.assertIn("Release status: ready for review.", parts[0]["text"])
        self.assertIn("<attached_file", parts[0]["text"])

    def test_attached_snapshot_is_used_after_original_file_is_removed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "finalreport.txt"
            source.write_text("Captured attachment content.", encoding="utf-8")
            with patch.dict("os.environ", {"XDG_DATA_HOME": str(root / "data")}, clear=False):
                attachment = import_attachment(source, source="user_file")
                source.unlink()
                converted = _responses_input_messages([
                    {
                        "role": "user",
                        "content": "Summarize the attachment.",
                        "attachments": [attachment.to_store_input()],
                    }
                ], provider="openai")

        text = converted[0]["content"][0]["text"]
        self.assertIn("Captured attachment content.", text)
        self.assertNotIn(str(source), text)

    def test_image_attachment_model_input_limit_is_configurable(self) -> None:
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "large.png"
            image.write_bytes(b"12345")
            item = {"filename": image.name, "mime": "image/png", "storage_path": str(image)}

            with patch.dict("os.environ", {"AGENT_MAX_IMAGE_ATTACHMENT_BYTES": "4"}):
                with self.assertRaisesRegex(RuntimeError, "model-input limit"):
                    _image_data_url(item)

    def test_responses_tools_convert_to_chat_and_anthropic_shapes(self) -> None:
        tools = [
            {
                "type": "function",
                "name": "glob",
                "description": "Find files",
                "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
            }
        ]

        chat_tools = _responses_tools_to_chat(tools)
        anthropic_tools = _responses_tools_to_anthropic(tools)

        self.assertEqual(chat_tools[0]["function"]["name"], "glob")
        self.assertEqual(anthropic_tools[0]["name"], "glob")
        self.assertEqual(anthropic_tools[0]["input_schema"]["type"], "object")

    def test_chat_completion_tool_call_converts_to_response_shape(self) -> None:
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="glob",
                                    arguments='{"pattern":"*.py"}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )

        response = _chat_completion_to_response(completion)

        self.assertEqual(response.output[0]["type"], "function_call")
        self.assertEqual(response.output[0]["name"], "glob")
        self.assertEqual(response.output_text, "")

    def test_anthropic_tool_use_converts_to_response_shape(self) -> None:
        response = _anthropic_message_to_response(
            {
                "content": [
                    {"type": "thinking", "thinking": "internal reasoning"},
                    {"type": "text", "text": "Checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "grep", "input": {"pattern": "auth"}},
                ],
                "usage": {"input_tokens": 20, "output_tokens": 5},
            }
        )

        self.assertEqual(response.output_text, "Checking")
        self.assertEqual(response.reasoning_text, "internal reasoning")
        self.assertEqual(response.output[0]["call_id"], "toolu_1")
        self.assertEqual(response.output[0]["arguments"], {"pattern": "auth"})

    def test_responses_messages_convert_to_anthropic_tool_blocks(self) -> None:
        converted = _responses_messages_to_anthropic(
            [
                {"role": "user", "content": "Find it"},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "glob",
                    "arguments": '{"pattern":"*.py"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": '{"matches":[]}',
                },
            ]
        )

        self.assertEqual(converted[0], {"role": "user", "content": "Find it"})
        self.assertEqual(converted[1]["content"][0]["type"], "tool_use")
        self.assertEqual(converted[2]["content"][0]["type"], "tool_result")


if __name__ == "__main__":
    unittest.main()
