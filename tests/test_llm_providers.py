import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nym_agent.llm import (
    LLMClient,
    _anthropic_message_to_response,
    _chat_completion_to_response,
    _default_model_for_provider,
    _friendly_llm_error,
    _normalize_provider,
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
        self.assertEqual(_normalize_provider("claude"), "anthropic")
        self.assertEqual(_normalize_provider("deepseek-ai"), "deepseek")
        self.assertEqual(_normalize_provider("zai"), "glm")
        self.assertEqual(_normalize_provider("zhipuai"), "glm")

    def test_provider_defaults_cover_local_free_models(self) -> None:
        self.assertEqual(_default_model_for_provider("deepseek"), "deepseek-v4-flash")
        self.assertEqual(_default_model_for_provider("glm"), "glm-4")
        self.assertEqual(_default_model_for_provider("ollama"), "llama3.1")

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
                    {"type": "text", "text": "Checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "grep", "input": {"pattern": "auth"}},
                ],
                "usage": {"input_tokens": 20, "output_tokens": 5},
            }
        )

        self.assertEqual(response.output_text, "Checking")
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
