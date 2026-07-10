import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nym_agent.llm import (
    LLMClient,
    _anthropic_message_to_response,
    _chat_reasoning_effort,
    _chat_completion_to_response,
    _default_model_for_provider,
    _default_reasoning_effort,
    _default_reasoning_summary,
    _friendly_llm_error,
    _model_supports_reasoning,
    _normalize_provider,
    _responses_reasoning_config,
    _responses_messages_to_anthropic,
    _responses_messages_to_chat,
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

    def test_provider_defaults_cover_local_free_models(self) -> None:
        self.assertEqual(_default_model_for_provider("deepseek"), "deepseek-v4-flash")
        self.assertEqual(_default_model_for_provider("glm"), "glm-4")
        self.assertEqual(_default_model_for_provider("ollama"), "llama3.1")
        self.assertEqual(_default_model_for_provider("llamacpp"), "local-model")
        self.assertEqual(_default_model_for_provider("vllm"), "local-model")
        self.assertEqual(_default_model_for_provider("localai"), "local-model")

    def test_open_source_local_providers_need_no_login_or_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            clients = {
                provider: LLMClient(provider=provider)
                for provider in ("ollama", "lmstudio", "llamacpp", "vllm", "localai")
            }

        for client in clients.values():
            self.assertEqual(client.mode, "local")
            self.assertIsNone(client.configuration_error)
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
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_reasoning_capability_detection_is_provider_specific(self) -> None:
        self.assertTrue(_model_supports_reasoning("openai", "gpt-5.4-mini"))
        self.assertTrue(_model_supports_reasoning("deepseek", "deepseek-reasoner"))
        self.assertFalse(_model_supports_reasoning("openai", "gpt-4o"))
        self.assertFalse(_model_supports_reasoning("anthropic", "claude-3-5-haiku-latest"))

    def test_reasoning_defaults_apply_only_to_reasoning_models(self) -> None:
        self.assertEqual(_default_reasoning_effort("openai", "gpt-5.4-mini"), "medium")
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

    def test_missing_deepseek_key_is_deferred_until_request(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(provider="deepseek")

        self.assertIsNotNone(client.configuration_error)
        with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
            client.respond(instructions="", messages=[{"role": "user", "content": "hi"}], tools=[])

    def test_missing_openai_key_is_deferred_until_request(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(provider="openai")

        self.assertIsNotNone(client.configuration_error)
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
