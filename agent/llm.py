from __future__ import annotations

import json
import base64
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .pricing import calculate_token_cost_usd

DEFAULT_MAX_TEXT_ATTACHMENT_BYTES = 512 * 1024
DEFAULT_MAX_IMAGE_ATTACHMENT_BYTES = 10 * 1024 * 1024


SUPPORTED_PROVIDERS = {
    "openai",
    "openai-compatible",
    "ollama",
    "lmstudio",
    "llamacpp",
    "vllm",
    "localai",
    "copilot",
    "anthropic",
    "gemini",
    "groq",
    "openrouter",
    "bedrock",
    "azure",
    "vertexai",
    "deepseek",
    "glm",
    "minimax",
}
UNAVAILABLE_PROVIDER_TRANSPORTS = frozenset({
    "copilot",
    "gemini",
    "bedrock",
    "azure",
    "vertexai",
})
AVAILABLE_PROVIDERS = frozenset(SUPPORTED_PROVIDERS) - UNAVAILABLE_PROVIDER_TRANSPORTS


def OpenAI(*args: Any, **kwargs: Any) -> Any:
    """Construct the optional SDK client only when a provider needs it."""
    from openai import OpenAI as OpenAIClient

    return OpenAIClient(*args, **kwargs)


@dataclass
class LLMClient:
    model: str | None = None
    provider: str | None = None
    client: Any = field(default=None, init=False, repr=False)
    endpoint: str = field(default="", init=False)
    mode: str = field(default="", init=False)
    configuration_error: str | None = field(default=None, init=False)
    configuration_state: str = field(default="ready", init=False)
    reasoning_effort: str | None = field(default=None, init=False)
    reasoning_summary: str | None = field(default=None, init=False)
    turn_usage: dict[str, int] = field(default_factory=lambda: empty_usage(), init=False)
    turn_cost_usd: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.provider = _normalize_provider(self.provider or os.environ.get("AGENT_LLM_PROVIDER") or "openai")
        self.model = self.model or _default_model_for_provider(self.provider)
        self.reasoning_effort = _default_reasoning_effort(self.provider, self.model)
        self.reasoning_summary = _default_reasoning_summary(self.provider, self.model)
        timeout = _llm_timeout_seconds()
        if self.provider == "openai":
            self.endpoint = "OpenAI"
            self.mode = "hosted"
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key, timeout=timeout)
            else:
                self.configuration_error = _provider_configuration_error("OpenAI", "OPENAI_API_KEY")
                self.configuration_state = "api_key_required"
        elif self.provider == "openai-compatible":
            base_url = os.environ.get("AGENT_OPENAI_COMPAT_BASE_URL") or ""
            self.endpoint = base_url
            self.mode = "compatible"
            if base_url:
                self.client = OpenAI(
                    api_key=os.environ.get("AGENT_OPENAI_COMPAT_API_KEY") or "local",
                    base_url=base_url,
                    timeout=timeout,
                )
            else:
                self.configuration_error = _provider_configuration_error(
                    "OpenAI-compatible provider",
                    "AGENT_OPENAI_COMPAT_BASE_URL",
                )
                self.configuration_state = "endpoint_required"
        elif self.provider == "ollama":
            base_url = _ollama_base_url()
            self.client = OpenAI(
                api_key=os.environ.get("OLLAMA_API_KEY") or "ollama",
                base_url=base_url,
                timeout=timeout,
            )
            self.endpoint = base_url
            self.mode = "local"
        elif self.provider == "lmstudio":
            base_url = os.environ.get("AGENT_LMSTUDIO_BASE_URL") or os.environ.get("LMSTUDIO_BASE_URL") or "http://localhost:1234/v1"
            self.client = OpenAI(
                api_key=os.environ.get("LMSTUDIO_API_KEY") or "lmstudio",
                base_url=base_url,
                timeout=timeout,
            )
            self.endpoint = base_url
            self.mode = "local"
        elif self.provider == "llamacpp":
            base_url = os.environ.get("AGENT_LLAMACPP_BASE_URL") or os.environ.get("LLAMACPP_BASE_URL") or "http://localhost:8080/v1"
            self.client = OpenAI(
                api_key=os.environ.get("LLAMACPP_API_KEY") or "local",
                base_url=base_url,
                timeout=timeout,
            )
            self.endpoint = base_url
            self.mode = "local"
        elif self.provider == "vllm":
            base_url = os.environ.get("AGENT_VLLM_BASE_URL") or os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1"
            self.client = OpenAI(
                api_key=os.environ.get("VLLM_API_KEY") or "local",
                base_url=base_url,
                timeout=timeout,
            )
            self.endpoint = base_url
            self.mode = "local"
        elif self.provider == "localai":
            base_url = os.environ.get("AGENT_LOCALAI_BASE_URL") or os.environ.get("LOCALAI_BASE_URL") or "http://localhost:8080/v1"
            self.client = OpenAI(
                api_key=os.environ.get("LOCALAI_API_KEY") or "local",
                base_url=base_url,
                timeout=timeout,
            )
            self.endpoint = base_url
            self.mode = "local"
        elif self.provider == "anthropic":
            self.endpoint = "Anthropic"
            self.mode = "hosted"
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self.configuration_error = _provider_configuration_error("Anthropic", "ANTHROPIC_API_KEY")
                self.configuration_state = "api_key_required"
        elif self.provider == "gemini":
            self.endpoint = "Google Gemini"
            self.mode = "hosted"
            if not os.environ.get("GOOGLE_API_KEY"):
                self.configuration_error = _provider_configuration_error("Google Gemini", "GOOGLE_API_KEY")
                self.configuration_state = "api_key_required"
            else:
                self.configuration_error = "Google Gemini transport is not implemented yet."
                self.configuration_state = "unavailable"
        elif self.provider == "groq":
            self.endpoint = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            self.mode = "hosted"
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=self.endpoint, timeout=timeout)
            else:
                self.configuration_error = _provider_configuration_error("Groq", "GROQ_API_KEY")
                self.configuration_state = "api_key_required"
        elif self.provider == "openrouter":
            self.endpoint = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.mode = "hosted"
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=self.endpoint, timeout=timeout)
            else:
                self.configuration_error = _provider_configuration_error("OpenRouter", "OPENROUTER_API_KEY")
                self.configuration_state = "api_key_required"
        elif self.provider == "azure":
            self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "Azure OpenAI")
            self.mode = "hosted"
            if not os.environ.get("AZURE_OPENAI_API_KEY"):
                self.configuration_error = _provider_configuration_error("Azure OpenAI", "AZURE_OPENAI_API_KEY")
                self.configuration_state = "api_key_required"
            else:
                self.configuration_error = "Azure OpenAI transport is not implemented yet."
                self.configuration_state = "unavailable"
        elif self.provider == "bedrock":
            self.endpoint = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "AWS Bedrock"
            self.mode = "hosted"
            if not _has_bedrock_credentials():
                self.configuration_error = "AWS Bedrock is not configured. Set AWS_PROFILE, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or AWS_BEARER_TOKEN_BEDROCK."
                self.configuration_state = "credentials_required"
            else:
                self.configuration_error = "AWS Bedrock transport is not implemented yet."
                self.configuration_state = "unavailable"
        elif self.provider == "vertexai":
            self.endpoint = os.environ.get("VERTEX_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "Google Cloud Vertex AI"
            self.mode = "hosted"
            if not (os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")):
                self.configuration_error = "Google Cloud Vertex AI is not configured. Set GOOGLE_CLOUD_PROJECT and authenticate with Application Default Credentials."
                self.configuration_state = "credentials_required"
            else:
                self.configuration_error = "Google Cloud Vertex AI transport is not implemented yet."
                self.configuration_state = "unavailable"
        elif self.provider == "copilot":
            self.endpoint = "GitHub Copilot"
            self.mode = "hosted"
            self.configuration_error = "GitHub Copilot sign-in is not implemented yet."
            self.configuration_state = "credentials_required"
        elif self.provider == "deepseek":
            self.endpoint = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self.mode = "hosted"
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=self.endpoint, timeout=timeout)
            else:
                self.configuration_error = _provider_configuration_error("DeepSeek", "DEEPSEEK_API_KEY")
                self.configuration_state = "api_key_required"
        elif self.provider == "glm":
            self.endpoint = (
                os.environ.get("GLM_BASE_URL")
                or os.environ.get("ZAI_BASE_URL")
                or os.environ.get("ZHIPUAI_BASE_URL")
                or os.environ.get("BIGMODEL_BASE_URL")
                or "https://open.bigmodel.cn/api/paas/v4"
            )
            self.mode = "hosted"
            api_key = _first_env("GLM_API_KEY", "ZAI_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=self.endpoint, timeout=timeout)
            else:
                self.configuration_error = _provider_configuration_error("GLM", "GLM_API_KEY")
                self.configuration_state = "api_key_required"
        elif self.provider == "minimax":
            self.endpoint = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
            self.mode = "hosted"
            api_key = os.environ.get("MINIMAX_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=self.endpoint, timeout=timeout)
            else:
                self.configuration_error = _provider_configuration_error("MiniMax", "MINIMAX_API_KEY")
                self.configuration_state = "api_key_required"
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def complete(self, prompt: str) -> str:
        response = self.respond(
            instructions="",
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        return _get(response, "output_text", "")

    def warm(self) -> None:
        """Load a local Ollama model without consuming a user prompt."""
        if self.provider != "ollama" or not self.model:
            return

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
        }
        keep_alive = os.environ.get("AGENT_OLLAMA_KEEP_ALIVE", "").strip()
        if keep_alive:
            payload["keep_alive"] = keep_alive

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Nym/1.0",
        }
        api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            _ollama_generate_url(self.endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout = _float_env("AGENT_OLLAMA_WARM_TIMEOUT_SECONDS") or _llm_timeout_seconds()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                _friendly_llm_error(
                    exc,
                    self.provider,
                    self.model,
                    self.endpoint,
                    self.mode,
                    body=body,
                )
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                _friendly_llm_error(
                    exc,
                    self.provider,
                    self.model,
                    self.endpoint,
                    self.mode,
                )
            ) from exc

    def respond(
        self,
        *,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        event_handler: Callable[[Any], None] | None = None,
    ) -> Any:
        if self.configuration_error:
            raise RuntimeError(self.configuration_error)
        if self.provider == "openai":
            return self._respond_openai_responses(
                instructions=instructions,
                messages=messages,
                tools=tools,
                previous_response_id=previous_response_id,
                tool_choice=tool_choice,
                stream=stream,
                event_handler=event_handler,
            )
        if self.provider in {"openai-compatible", "ollama", "lmstudio", "llamacpp", "vllm", "localai", "groq", "openrouter", "deepseek", "glm", "minimax"}:
            return self._respond_openai_chat(
                instructions=instructions,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                event_handler=event_handler,
            )
        if self.provider == "anthropic":
            return self._respond_anthropic(
                instructions=instructions,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _respond_openai_responses(
        self,
        *,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        tool_choice: str | dict[str, Any] | None,
        stream: bool,
        event_handler: Callable[[Any], None] | None,
    ) -> Any:
        request_args = {
            "model": self.model,
            "instructions": instructions,
            "input": _responses_input_messages(messages, provider=self.provider),
            "tools": tools,
            "previous_response_id": previous_response_id,
        }
        if tool_choice is not None:
            request_args["tool_choice"] = tool_choice
        reasoning = _responses_reasoning_config(self.provider, self.model, self.reasoning_effort, self.reasoning_summary)
        if reasoning is not None:
            request_args["reasoning"] = reasoning

        counted_input_tokens = self._count_openai_input_tokens(request_args)

        if stream:
            try:
                with self.client.responses.stream(**request_args) as response_stream:
                    if event_handler is None:
                        response_stream.until_done()
                    else:
                        for event in response_stream:
                            event_handler(event)
                    response = response_stream.get_final_response()
            except Exception as exc:
                raise RuntimeError(_friendly_llm_error(exc, self.provider, self.model, self.endpoint, self.mode)) from exc
        else:
            try:
                response = self.client.responses.create(**request_args)
            except Exception as exc:
                raise RuntimeError(_friendly_llm_error(exc, self.provider, self.model, self.endpoint, self.mode)) from exc
        self._record_usage(response, input_tokens=counted_input_tokens)
        return response

    def _respond_openai_chat(
        self,
        *,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
        stream: bool,
        event_handler: Callable[[Any], None] | None,
    ) -> Any:
        request_args: dict[str, Any] = {
            "model": self.model,
            "messages": _responses_messages_to_chat(messages, instructions, provider=self.provider),
        }
        chat_tools = _responses_tools_to_chat(tools)
        allowed_text_tool_names = {
            str(item.get("function", {}).get("name"))
            for item in chat_tools
            if item.get("function", {}).get("name")
        }
        if chat_tools:
            request_args["tools"] = chat_tools
        chat_tool_choice = _chat_tool_choice(tool_choice)
        if chat_tool_choice is not None:
            request_args["tool_choice"] = chat_tool_choice
        if self.mode == "local":
            local_temperature = _float_env("AGENT_LOCAL_TEMPERATURE")
            request_args["temperature"] = (
                local_temperature if local_temperature is not None else 0.0
            )
        reasoning_effort = _chat_reasoning_effort(self.provider, self.model, self.reasoning_effort)
        if reasoning_effort is not None:
            request_args["reasoning_effort"] = reasoning_effort

        try:
            if stream:
                response = self._stream_openai_chat(
                    request_args,
                    event_handler=event_handler,
                    text_tool_names=(
                        allowed_text_tool_names if self.mode == "local" else set()
                    ),
                )
            else:
                completion = self.client.chat.completions.create(**request_args)
                response = _chat_completion_to_response(
                    completion,
                    text_tool_names=(
                        allowed_text_tool_names if self.mode == "local" else set()
                    ),
                )
        except Exception as exc:
            raise RuntimeError(_friendly_llm_error(exc, self.provider, self.model, self.endpoint, self.mode)) from exc
        self._record_usage(response)
        return response

    def _stream_openai_chat(
        self,
        request_args: dict[str, Any],
        *,
        event_handler: Callable[[Any], None] | None,
        text_tool_names: set[str],
    ) -> Any:
        chunks = self.client.chat.completions.create(**request_args, stream=True)
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: Any = None
        sequence = 0
        streamed_local_text = False
        local_text_stream_ready = False

        def emit(event: dict[str, Any]) -> None:
            nonlocal sequence
            event["sequence_number"] = sequence
            sequence += 1
            if event_handler is not None:
                event_handler(event)

        emit({"type": "response.in_progress"})
        for chunk in chunks:
            chunk_usage = _get(chunk, "usage")
            if chunk_usage is not None:
                usage = chunk_usage
            choices = _get(chunk, "choices", []) or []
            if not choices:
                continue
            delta = _get(choices[0], "delta") or {}
            reasoning = _get(delta, "reasoning_content") or _get(delta, "reasoning")
            if self.mode != "local" and isinstance(reasoning, str) and reasoning:
                emit({"type": "response.reasoning_text.delta", "delta": reasoning})
            content = _get(delta, "content")
            if isinstance(content, str) and content:
                text_parts.append(content)
                output_text = "".join(text_parts)
                if self.mode != "local":
                    emit({"type": "response.output_text.delta", "delta": content})
                elif local_text_stream_ready:
                    emit({"type": "response.output_text.delta", "delta": content})
                    streamed_local_text = True
                elif _local_text_stream_is_safe(
                    output_text,
                    allowed_tool_names=text_tool_names,
                ):
                    # Flush the short ambiguous prefix together. Subsequent
                    # chunks can pass through without waiting for generation.
                    emit({"type": "response.output_text.delta", "delta": output_text})
                    streamed_local_text = True
                    local_text_stream_ready = True

            for call_delta in _get(delta, "tool_calls", []) or []:
                index = _get(call_delta, "index", 0)
                index = index if isinstance(index, int) else 0
                call = calls.setdefault(index, {
                    "id": f"call_{index}",
                    "name": "",
                    "arguments": "",
                    "emitted": False,
                })
                call_id = _get(call_delta, "id")
                if isinstance(call_id, str) and call_id:
                    call["id"] = call_id
                function = _get(call_delta, "function") or {}
                name = _get(function, "name")
                if isinstance(name, str) and name:
                    call["name"] = name
                if call["name"] and not call["emitted"]:
                    emit({
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": call["id"],
                            "call_id": call["id"],
                            "name": call["name"],
                        },
                    })
                    call["emitted"] = True
                arguments = _get(function, "arguments")
                if isinstance(arguments, str) and arguments:
                    call["arguments"] += arguments
                    emit({
                        "type": "response.function_call_arguments.delta",
                        "item_id": call["id"],
                        "delta": arguments,
                    })

        output: list[dict[str, Any]] = []
        output_text = "".join(text_parts)
        if text_tool_names:
            fallback_calls, visible_text = _text_content_tool_calls(
                output_text,
                allowed_names=text_tool_names,
            )
            if fallback_calls and not calls and not streamed_local_text:
                for index, fallback in enumerate(fallback_calls):
                    call_id = f"call_text_{index}"
                    calls[index] = {
                        "id": call_id,
                        "name": fallback["name"],
                        "arguments": fallback["arguments"],
                        "emitted": True,
                    }
                    emit({
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": call_id,
                            "call_id": call_id,
                            "name": fallback["name"],
                        },
                    })
                    emit({
                        "type": "response.function_call_arguments.delta",
                        "item_id": call_id,
                        "delta": fallback["arguments"],
                    })
                output_text = visible_text
            if (
                output_text
                and not streamed_local_text
                and not _looks_like_command_envelope(output_text)
            ):
                emit({"type": "response.output_text.delta", "delta": output_text})
        elif (
            self.mode == "local"
            and output_text
            and not streamed_local_text
            and not _looks_like_command_envelope(output_text)
        ):
            emit({"type": "response.output_text.delta", "delta": output_text})
        for index in sorted(calls):
            call = calls[index]
            emit({
                "type": "response.function_call_arguments.done",
                "item_id": call["id"],
                "arguments": call["arguments"],
            })
            output.append({
                "type": "function_call",
                "call_id": call["id"],
                "name": call["name"],
                "arguments": call["arguments"] or "{}",
            })
        emit({"type": "response.completed"})
        return SimpleNamespace(
            output=output,
            output_text=output_text,
            usage=usage,
        )

    def _respond_anthropic(
        self,
        *,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _int_env("ANTHROPIC_MAX_TOKENS", default=4096),
            "messages": _responses_messages_to_anthropic(messages),
        }
        if instructions.strip():
            payload["system"] = instructions
        anthropic_tools = _responses_tools_to_anthropic(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools
        anthropic_tool_choice = _anthropic_tool_choice(tool_choice)
        if anthropic_tool_choice is not None:
            payload["tool_choice"] = anthropic_tool_choice

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": _required_env("ANTHROPIC_API_KEY", "Anthropic"),
                "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_float_env("AGENT_LLM_TIMEOUT_SECONDS") or 120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(_friendly_llm_error(exc, self.provider, self.model, self.endpoint, self.mode, body=body)) from exc
        except Exception as exc:
            raise RuntimeError(_friendly_llm_error(exc, self.provider, self.model, self.endpoint, self.mode)) from exc

        result = _anthropic_message_to_response(data)
        self._record_usage(result)
        return result

    def reset_turn_usage(self) -> None:
        self.turn_usage = empty_usage()
        self.turn_cost_usd = 0.0

    def consume_turn_usage(self) -> dict[str, int]:
        usage = dict(self.turn_usage)
        self.reset_turn_usage()
        return usage

    def consume_turn_metrics(self) -> tuple[dict[str, int], float]:
        usage = dict(self.turn_usage)
        cost_usd = self.turn_cost_usd
        self.reset_turn_usage()
        return usage, cost_usd

    def estimate_cost_usd(self, usage: dict[str, int]) -> float:
        return calculate_token_cost_usd(
            provider=self.provider,
            model=self.model,
            usage=usage,
        )

    def _record_usage(self, response: Any, *, input_tokens: int | None = None) -> None:
        usage = _get(response, "usage")
        if usage is None and input_tokens is None:
            return

        delta = empty_usage()
        delta["input"] = input_tokens if input_tokens is not None else (
            _int_field(usage, "input_tokens") or _int_field(usage, "prompt_tokens")
        )
        delta["output"] = (
            _int_field(usage, "output_tokens")
            or _int_field(usage, "completion_tokens")
        )

        input_details = _get(usage, "input_tokens_details") or {}
        output_details = _get(usage, "output_tokens_details") or {}
        delta["cache_read"] = _int_field(input_details, "cached_tokens")
        delta["cache_write"] = _int_field(input_details, "cache_creation_tokens")
        delta["reasoning"] = _int_field(output_details, "reasoning_tokens")
        for name, value in delta.items():
            self.turn_usage[name] += value
        self.turn_cost_usd += self.estimate_cost_usd(delta)

    def _count_openai_input_tokens(self, request_args: dict[str, Any]) -> int | None:
        try:
            response = self.client.responses.input_tokens.count(**request_args)
        except Exception:
            # Counting improves accuracy but must not make an otherwise valid
            # model request unavailable on an older endpoint or SDK.
            return None
        value = _get(response, "input_tokens")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return max(0, int(value))


def empty_usage() -> dict[str, int]:
    return {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache_read": 0,
        "cache_write": 0,
    }


def _normalize_provider(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-")
    aliases = {
        "openai-chat": "openai-compatible",
        "compatible": "openai-compatible",
        "openai-compatible-chat": "openai-compatible",
        "lm-studio": "lmstudio",
        "llama.cpp": "llamacpp",
        "llama-cpp": "llamacpp",
        "llama_cpp": "llamacpp",
        "v-llm": "vllm",
        "local-ai": "localai",
        "anthropic-claude": "anthropic",
        "claude": "anthropic",
        "deepseek-ai": "deepseek",
        "deepseek-chat": "deepseek",
        "github-copilot": "copilot",
        "github": "copilot",
        "google": "gemini",
        "google-gemini": "gemini",
        "google-ai": "gemini",
        "google-ai-studio": "gemini",
        "google-vertex": "vertexai",
        "google-vertex-ai": "vertexai",
        "google-cloud-vertex": "vertexai",
        "google-cloud-vertexai": "vertexai",
        "gcp-vertex": "vertexai",
        "vertex": "vertexai",
        "vertex-ai": "vertexai",
        "aws": "bedrock",
        "aws-bedrock": "bedrock",
        "amazon-bedrock": "bedrock",
        "azure-openai": "azure",
        "open-router": "openrouter",
        "z-ai": "glm",
        "z.ai": "glm",
        "zai": "glm",
        "zhipu": "glm",
        "zhipuai": "glm",
        "bigmodel": "glm",
        "mini-max": "minimax",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {value}. Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
    return normalized


def _default_model_for_provider(provider: str) -> str:
    defaults = {
        "openai": os.environ.get("OPENAI_MODEL") or "gpt-5.4-mini",
        "openai-compatible": os.environ.get("AGENT_OPENAI_COMPAT_MODEL") or "local-model",
        "ollama": os.environ.get("OLLAMA_MODEL") or "llama3.1",
        "lmstudio": os.environ.get("LMSTUDIO_MODEL") or "local-model",
        "llamacpp": os.environ.get("LLAMACPP_MODEL") or "local-model",
        "vllm": os.environ.get("VLLM_MODEL") or "local-model",
        "localai": os.environ.get("LOCALAI_MODEL") or "local-model",
        "copilot": os.environ.get("COPILOT_MODEL") or "gpt-4.1",
        "anthropic": os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
        "gemini": os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL") or "gemini-2.5-pro",
        "groq": os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile",
        "openrouter": os.environ.get("OPENROUTER_MODEL") or "anthropic/claude-sonnet-4.5",
        "bedrock": os.environ.get("BEDROCK_MODEL") or "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "azure": os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("AZURE_OPENAI_MODEL") or "gpt-4.1",
        "vertexai": os.environ.get("VERTEX_MODEL") or os.environ.get("GOOGLE_VERTEX_MODEL") or "gemini-2.5-pro",
        "deepseek": os.environ.get("DEEPSEEK_MODEL")
        or "deepseek-v4-flash",
        "glm": os.environ.get("GLM_MODEL")
        or os.environ.get("ZAI_MODEL")
        or "glm-4",
        "minimax": os.environ.get("MINIMAX_MODEL") or "MiniMax-M3",
    }
    return defaults[provider]


def _default_reasoning_effort(provider: str, model: str | None) -> str | None:
    if not _model_supports_reasoning(provider, model):
        return None
    raw = (os.environ.get("AGENT_REASONING_EFFORT") or "medium").strip().casefold()
    if raw in {"minimal", "low", "medium", "high"}:
        return raw
    return "medium"


def _default_reasoning_summary(provider: str, model: str | None) -> str | None:
    if provider != "openai" or not _model_supports_reasoning(provider, model):
        return None
    raw = (os.environ.get("AGENT_REASONING_SUMMARY") or "").strip().casefold()
    if raw in {"auto", "concise", "detailed"}:
        return raw
    return "auto"


def _model_supports_reasoning(provider: str | None, model: str | None) -> bool:
    provider_name = (provider or "").strip().casefold()
    model_name = (model or "").strip().casefold()
    if not provider_name or not model_name:
        return False

    reasoning_prefixes = ("gpt-5", "o1", "o3", "o4")
    if provider_name == "openai":
        return model_name.startswith(reasoning_prefixes)
    if provider_name in {"deepseek", "ollama", "lmstudio", "llamacpp", "vllm", "localai", "openai-compatible", "groq", "openrouter", "glm", "minimax"}:
        return "reason" in model_name or model_name.startswith(("o1", "o3", "o4"))
    if provider_name == "anthropic":
        return "thinking" in model_name or "claude-3.7" in model_name or "claude-sonnet-4" in model_name
    return False


def _responses_reasoning_config(
    provider: str | None,
    model: str | None,
    effort: str | None,
    summary: str | None,
) -> dict[str, str] | None:
    if provider != "openai" or not _model_supports_reasoning(provider, model):
        return None
    result: dict[str, str] = {}
    if effort in {"minimal", "low", "medium", "high"}:
        result["effort"] = effort
    if summary in {"auto", "concise", "detailed"}:
        result["summary"] = summary
    return result or None


def _chat_reasoning_effort(
    provider: str | None,
    model: str | None,
    effort: str | None,
) -> str | None:
    if provider not in {"openai-compatible", "ollama", "lmstudio", "llamacpp", "vllm", "localai", "groq", "openrouter", "deepseek", "glm", "minimax"}:
        return None
    if not _model_supports_reasoning(provider, model):
        return None
    if effort in {"minimal", "low", "medium", "high"}:
        return effort
    return None


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _required_env(name: str, provider_name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(_provider_configuration_error(provider_name, name))
    return value


def _has_bedrock_credentials() -> bool:
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or os.environ.get("AWS_PROFILE")
        or (
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
    )


def _provider_configuration_error(provider_name: str, env_name: str) -> str:
    return (
        f"{provider_name} is not configured. Set {env_name}, or switch to a local model with "
        "`/model ollama <installed-model>` or `agent --provider ollama --model <installed-model>`."
    )


def _friendly_llm_error(
    exc: Exception,
    provider: str | None,
    model: str | None,
    endpoint: str,
    mode: str,
    *,
    body: str | None = None,
) -> str:
    provider_name = provider or "unknown"
    model_name = model or "unknown"
    text = body or str(exc)
    text_lower = text.casefold()

    if "api key" in text_lower or "authentication" in text_lower or "unauthorized" in text_lower:
        return (
            f"{provider_name} is not authenticated. Set the provider API key or switch providers with "
            f"`/model ollama <installed-model>`. Active model: {model_name}."
        )

    if "connection" in text_lower or "refused" in text_lower or "connection error" in text_lower:
        return (
            f"Cannot reach {provider_name} at {endpoint or 'configured endpoint'}. "
            f"Start the local server or switch providers with `/model <source> <model>`."
        )

    if "model" in text_lower and ("not found" in text_lower or "does not exist" in text_lower):
        if mode == "local":
            return (
                f"Local model `{model_name}` is not available at {endpoint}. "
                f"Install/pull that model, or switch to an installed one with `/model <model>`."
            )
        return (
            f"Model `{model_name}` is not available for {provider_name}. "
            f"Choose a valid model with `/model <model>` or configure the provider model env var."
        )

    return f"{provider_name} request failed for model `{model_name}`: {text}"


def _ollama_base_url() -> str:
    raw = (
        os.environ.get("AGENT_OLLAMA_BASE_URL")
        or os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("OLLAMA_HOST")
        or "http://localhost:11434"
    )
    raw = raw.rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def _ollama_generate_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    return f"{root}/api/generate"


def _responses_input_messages(messages: list[dict[str, Any]], *, provider: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in messages:
        role = item.get("role")
        attachments = _attachment_items(item)
        if attachments and role == "user":
            converted.append({
                "role": role,
                "content": _openai_content_parts(item, provider=provider),
            })
        else:
            # Keep ordinary Responses input byte-for-byte compatible with the
            # previous working transport. `input_text` is specifically for a
            # user multimodal content part, not a blanket replacement for
            # assistant/system history accepted by every Responses model.
            converted.append(item)
    return converted


def _responses_messages_to_chat(
    messages: list[dict[str, Any]], instructions: str, *, provider: str = "openai-compatible",
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if instructions.strip():
        converted.append({"role": "system", "content": instructions})
    for item in messages:
        item_type = item.get("type")
        if item_type == "function_call":
            converted.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                ],
            })
            continue
        if item_type == "function_call_output":
            converted.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": str(item.get("output", "")),
            })
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant", "system"}:
            text = _message_content_text(content)
            attachments = _attachment_items(item)
            if attachments and role == "user":
                converted.append({"role": role, "content": _chat_content_parts(text, attachments, provider)})
            else:
                converted.append({"role": role, "content": text})
    return converted


def _responses_messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in messages:
        item_type = item.get("type")
        if item_type == "function_call":
            converted.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": item.get("call_id"),
                        "name": item.get("name"),
                        "input": _json_object(item.get("arguments", "{}")),
                    }
                ],
            })
            continue
        if item_type == "function_call_output":
            converted.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": item.get("call_id"),
                        "content": str(item.get("output", "")),
                    }
                ],
            })
            continue
        role = item.get("role")
        if role == "system":
            role = "user"
        if role in {"user", "assistant"}:
            text = _message_content_text(item.get("content"))
            attachments = _attachment_items(item)
            if attachments and role == "user":
                converted.append({"role": role, "content": _anthropic_content_parts(text, attachments)})
            else:
                converted.append({"role": role, "content": text})
    return converted


def _attachment_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    value = message.get("attachments")
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _attachment_manifest(attachments: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"Attached file: {item.get('filename', 'file')} ({item.get('mime', 'application/octet-stream')}, {item.get('size_bytes', 0)} bytes)"
        for item in attachments
    )


def _non_image_attachment_content(attachments: list[dict[str, Any]]) -> str:
    return "\n".join(_attachment_content(item) for item in attachments)


def _attachment_content(item: dict[str, Any]) -> str:
    filename = str(item.get("filename") or "file")
    mime = item.get("mime")
    path = item.get("storage_path")
    if not _is_text_attachment(mime, filename) or not isinstance(path, str):
        return _attachment_manifest([item])
    try:
        limit = _max_text_attachment_bytes()
        with Path(path).open("rb") as source:
            raw = source.read(limit + 1)
    except OSError:
        return f"Attached file is unavailable: {filename}."

    truncated = len(raw) > limit
    text = raw[:limit].decode("utf-8", errors="replace")
    suffix = "\n[Attachment content truncated at configured limit.]" if truncated else ""
    return (
        f"<attached_file name={filename!r} mime={mime!r}>\n"
        f"{text}{suffix}\n"
        "</attached_file>"
    )


def _is_text_attachment(mime: object, filename: str) -> bool:
    if isinstance(mime, str) and (
        mime.startswith("text/")
        or mime in {"application/json", "application/xml", "application/javascript"}
    ):
        return True
    return Path(filename).suffix.casefold() in {
        ".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".xml",
    }


def _max_text_attachment_bytes() -> int:
    raw = os.environ.get("AGENT_MAX_TEXT_ATTACHMENT_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_TEXT_ATTACHMENT_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_TEXT_ATTACHMENT_BYTES
    return value if value > 0 else DEFAULT_MAX_TEXT_ATTACHMENT_BYTES


def _image_data_url(item: dict[str, Any]) -> str:
    mime = item.get("mime")
    path = item.get("storage_path")
    if not isinstance(mime, str) or not mime.startswith("image/") or not isinstance(path, str):
        raise ValueError("Only image attachments can be converted to visual model input.")
    try:
        image_path = Path(path)
        size = image_path.stat().st_size
        limit = _max_image_attachment_bytes()
        if size > limit:
            raise RuntimeError(
                f"Image attachment exceeds the configured model-input limit of {limit} bytes: "
                f"{item.get('filename', image_path.name)}"
            )
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise RuntimeError(f"Attachment is no longer available: {item.get('filename', path)}") from exc
    return f"data:{mime};base64,{encoded}"


def _max_image_attachment_bytes() -> int:
    raw = os.environ.get("AGENT_MAX_IMAGE_ATTACHMENT_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_IMAGE_ATTACHMENT_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_IMAGE_ATTACHMENT_BYTES
    return value if value > 0 else DEFAULT_MAX_IMAGE_ATTACHMENT_BYTES


def _openai_content_parts(message: dict[str, Any], *, provider: str) -> list[dict[str, Any]]:
    attachments = _attachment_items(message)
    parts: list[dict[str, Any]] = [{"type": "input_text", "text": _message_content_text(message.get("content"))}]
    images = [item for item in attachments if isinstance(item.get("mime"), str) and item["mime"].startswith("image/")]
    if images and provider != "openai":
        raise RuntimeError(f"{provider} does not support image attachments through this transport yet.")
    for image in images:
        parts.append({"type": "input_image", "image_url": _image_data_url(image), "detail": "auto"})
    non_images = [item for item in attachments if item not in images]
    if non_images:
        parts[0]["text"] += "\n" + _non_image_attachment_content(non_images)
    return parts


def _chat_content_parts(text: str, attachments: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    images = [item for item in attachments if isinstance(item.get("mime"), str) and item["mime"].startswith("image/")]
    if images and provider not in {"openrouter", "openai-compatible"}:
        raise RuntimeError(f"{provider} does not support image attachments through this transport yet.")
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": _image_data_url(image)}})
    non_images = [item for item in attachments if item not in images]
    if non_images:
        parts[0]["text"] += "\n" + _non_image_attachment_content(non_images)
    return parts


def _anthropic_content_parts(text: str, attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for item in attachments:
        mime = item.get("mime")
        if isinstance(mime, str) and mime.startswith("image/"):
            data_url = _image_data_url(item)
            encoded = data_url.split(",", 1)[1]
            parts.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}})
    non_images = [item for item in attachments if not (isinstance(item.get("mime"), str) and item["mime"].startswith("image/"))]
    if non_images:
        parts[0]["text"] += "\n" + _non_image_attachment_content(non_images)
    return parts


def _responses_tools_to_chat(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return converted


def _responses_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        converted.append({
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


def _chat_tool_choice(tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if tool_choice == "required":
        return "required"
    if tool_choice in {"auto", "none"}:
        return tool_choice
    return None


def _anthropic_tool_choice(tool_choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "auto":
        return {"type": "auto"}
    return None


def _chat_completion_to_response(
    response: Any,
    *,
    text_tool_names: set[str] | None = None,
) -> Any:
    choice = (_get(response, "choices") or [None])[0]
    message = _get(choice, "message") if choice is not None else None
    output: list[dict[str, Any]] = []
    for call in _get(message, "tool_calls", []) or []:
        function = _get(call, "function") or {}
        output.append({
            "type": "function_call",
            "call_id": _get(call, "id"),
            "name": _get(function, "name"),
            "arguments": _get(function, "arguments", "{}"),
        })
    content = _get(message, "content", "") if message is not None else ""
    if not output and isinstance(content, str) and text_tool_names:
        fallback_calls, content = _text_content_tool_calls(
            content,
            allowed_names=text_tool_names,
        )
        for index, call in enumerate(fallback_calls):
            output.append({
                "type": "function_call",
                "call_id": f"call_text_{index}",
                "name": call["name"],
                "arguments": call["arguments"],
            })
    return SimpleNamespace(
        output=output,
        output_text=content or "",
        usage=_get(response, "usage"),
    )


_JSON_TOOL_FENCE_RE = re.compile(
    r"(?P<fence>`{1,3})(?:json)?\s*(?P<payload>\{.*?\})\s*(?P=fence)",
    flags=re.DOTALL | re.IGNORECASE,
)


def _text_content_tool_calls(
    content: str,
    *,
    allowed_names: set[str],
) -> tuple[list[dict[str, str]], str]:
    if not content.strip() or not allowed_names:
        return [], content

    tagged = list(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL))
    calls: list[dict[str, str]] = []
    if tagged:
        for match in tagged:
            parsed = _text_tool_call_object(match.group(1), allowed_names=allowed_names)
            if parsed is None:
                return [], content
            calls.append(parsed)
        visible = re.sub(
            r"<tool_call>\s*\{.*?\}\s*</tool_call>",
            "",
            content,
            flags=re.DOTALL,
        ).strip()
        return calls, visible

    fenced = list(_JSON_TOOL_FENCE_RE.finditer(content))
    if fenced:
        for match in fenced:
            parsed = _text_tool_call_object(
                match.group("payload"),
                allowed_names=allowed_names,
            )
            if parsed is None:
                return [], content
            calls.append(parsed)
        return calls, ""

    plain_calls = _plain_text_tool_calls(content, allowed_names=allowed_names)
    if plain_calls:
        return plain_calls, ""

    parsed = _text_tool_call_object(content.strip(), allowed_names=allowed_names)
    if parsed is None:
        return [], content
    return [parsed], ""


def _plain_text_tool_calls(
    content: str,
    *,
    allowed_names: set[str],
) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    decoder = json.JSONDecoder()
    position = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        name_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", content[position:])
        if name_match is None:
            return []
        name = name_match.group(0)
        if name not in allowed_names:
            return []
        position += name_match.end()
        while position < len(content) and content[position].isspace():
            position += 1
        try:
            arguments, position = decoder.raw_decode(content, position)
        except json.JSONDecodeError:
            return []
        if not isinstance(arguments, dict):
            return []
        calls.append({
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        })
    return calls


def _text_tool_call_object(
    payload: str,
    *,
    allowed_names: set[str],
) -> dict[str, str] | None:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    arguments = value.get("arguments")
    if not isinstance(name, str) or name not in allowed_names:
        return None
    if "arguments" not in value:
        serialized_arguments = "{}"
    elif isinstance(arguments, dict):
        serialized_arguments = json.dumps(arguments, ensure_ascii=False)
    elif isinstance(arguments, str):
        try:
            decoded_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded_arguments, dict):
            return None
        serialized_arguments = json.dumps(decoded_arguments, ensure_ascii=False)
    else:
        return None
    return {"name": name, "arguments": serialized_arguments}


def _looks_like_command_envelope(content: str) -> bool:
    if re.search(r"</?tool_(?:call|response)\b", content, flags=re.IGNORECASE):
        return True
    candidates = [content.strip()]
    candidates.extend(
        match.group("payload")
        for match in _JSON_TOOL_FENCE_RE.finditer(content)
    )
    return any(_is_command_envelope_object(candidate) for candidate in candidates)


def _local_text_stream_is_safe(
    content: str,
    *,
    allowed_tool_names: set[str],
) -> bool:
    text = content.strip()
    if not text:
        return False
    lowered = text.casefold()

    structural_prefixes = ("`", "<tool_call>", "<tool_response>")
    if any(
        lowered.startswith(prefix) or prefix.startswith(lowered)
        for prefix in structural_prefixes
    ):
        return False
    if text.startswith("{") or text.startswith("["):
        return False

    word_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text)
    if word_match is not None:
        word = word_match.group(0)
        matching_names = {
            name
            for name in allowed_tool_names
            if name.casefold().startswith(word.casefold())
        }
        if matching_names:
            rest = text[word_match.end():]
            if not rest:
                return False
            if any(name.casefold() == word.casefold() for name in matching_names):
                if not rest.strip() or rest.lstrip().startswith("{"):
                    return False

    return not _looks_like_command_envelope(text)


def _safe_local_chat_stream_prefix(content: str) -> bool:
    return _local_text_stream_is_safe(content, allowed_tool_names=set())


def _is_command_envelope_object(payload: str) -> bool:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return False
    if not isinstance(value, dict):
        return False
    command = value.get("command") or value.get("tool")
    if isinstance(command, str) and command.strip():
        return True
    name = value.get("name")
    return (
        isinstance(name, str)
        and bool(name.strip())
        and "arguments" in value
    )


def _anthropic_message_to_response(data: dict[str, Any]) -> Any:
    output: list[dict[str, Any]] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in data.get("content", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif item.get("type") == "thinking":
            thinking = item.get("thinking")
            if isinstance(thinking, str):
                reasoning_parts.append(thinking)
        elif item.get("type") == "tool_use":
            output.append({
                "type": "function_call",
                "call_id": item.get("id"),
                "name": item.get("name"),
                "arguments": item.get("input") or {},
            })
    return SimpleNamespace(
        output=output,
        output_text="\n".join(text_parts).strip(),
        reasoning_text="\n".join(reasoning_parts).strip(),
        usage=data.get("usage"),
    )


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _int_field(value: Any, key: str) -> int:
    raw = _get(value, key, 0)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return 0


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _llm_timeout_seconds() -> float:
    return _float_env("AGENT_LLM_TIMEOUT_SECONDS") or 60.0


def _int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
