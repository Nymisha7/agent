from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from openai import OpenAI


SUPPORTED_PROVIDERS = {
    "openai",
    "openai-compatible",
    "ollama",
    "lmstudio",
    "anthropic",
    "deepseek",
    "glm",
}


@dataclass
class LLMClient:
    model: str | None = None
    provider: str | None = None
    client: Any = field(default=None, init=False, repr=False)
    endpoint: str = field(default="", init=False)
    mode: str = field(default="", init=False)
    configuration_error: str | None = field(default=None, init=False)
    turn_usage: dict[str, int] = field(default_factory=lambda: empty_usage(), init=False)

    def __post_init__(self) -> None:
        self.provider = _normalize_provider(self.provider or os.environ.get("NYM_LLM_PROVIDER") or "openai")
        self.model = self.model or _default_model_for_provider(self.provider)
        if self.provider == "openai":
            self.endpoint = "OpenAI"
            self.mode = "hosted"
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.configuration_error = _provider_configuration_error("OpenAI", "OPENAI_API_KEY")
        elif self.provider == "openai-compatible":
            base_url = os.environ.get("NYM_OPENAI_COMPAT_BASE_URL") or ""
            self.endpoint = base_url
            self.mode = "compatible"
            if base_url:
                self.client = OpenAI(
                    api_key=os.environ.get("NYM_OPENAI_COMPAT_API_KEY") or "local",
                    base_url=base_url,
                )
            else:
                self.configuration_error = _provider_configuration_error(
                    "OpenAI-compatible provider",
                    "NYM_OPENAI_COMPAT_BASE_URL",
                )
        elif self.provider == "ollama":
            base_url = _ollama_base_url()
            self.client = OpenAI(
                api_key=os.environ.get("OLLAMA_API_KEY") or "ollama",
                base_url=base_url,
            )
            self.endpoint = base_url
            self.mode = "local"
        elif self.provider == "lmstudio":
            base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            self.client = OpenAI(
                api_key=os.environ.get("LMSTUDIO_API_KEY") or "lmstudio",
                base_url=base_url,
            )
            self.endpoint = base_url
            self.mode = "local"
        elif self.provider == "anthropic":
            self.endpoint = "Anthropic"
            self.mode = "hosted"
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self.configuration_error = _provider_configuration_error("Anthropic", "ANTHROPIC_API_KEY")
        elif self.provider == "deepseek":
            self.client, self.endpoint, self.mode = _deepseek_client()
        elif self.provider == "glm":
            self.client, self.endpoint, self.mode = _glm_client()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def complete(self, prompt: str) -> str:
        response = self.respond(
            instructions="",
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        return _get(response, "output_text", "")

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
        if self.provider in {"openai-compatible", "ollama", "lmstudio", "deepseek", "glm"}:
            return self._respond_openai_chat(
                instructions=instructions,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
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
            "input": messages,
            "tools": tools,
            "previous_response_id": previous_response_id,
        }
        if tool_choice is not None:
            request_args["tool_choice"] = tool_choice

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
        self._record_usage(response)
        return response

    def _respond_openai_chat(
        self,
        *,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
    ) -> Any:
        request_args: dict[str, Any] = {
            "model": self.model,
            "messages": _responses_messages_to_chat(messages, instructions),
        }
        chat_tools = _responses_tools_to_chat(tools)
        if chat_tools:
            request_args["tools"] = chat_tools
        chat_tool_choice = _chat_tool_choice(tool_choice)
        if chat_tool_choice is not None:
            request_args["tool_choice"] = chat_tool_choice

        try:
            response = self.client.chat.completions.create(**request_args)
        except Exception as exc:
            raise RuntimeError(_friendly_llm_error(exc, self.provider, self.model, self.endpoint, self.mode)) from exc
        self._record_usage(response)
        return _chat_completion_to_response(response)

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
            with urllib.request.urlopen(request, timeout=_float_env("NYM_LLM_TIMEOUT_SECONDS") or 120) as response:
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

    def consume_turn_usage(self) -> dict[str, int]:
        usage = dict(self.turn_usage)
        self.reset_turn_usage()
        return usage

    def estimate_cost_usd(self, usage: dict[str, int]) -> float:
        input_price = _float_env("NYM_INPUT_COST_USD_PER_MILLION_TOKENS")
        output_price = _float_env("NYM_OUTPUT_COST_USD_PER_MILLION_TOKENS")
        if input_price is None and output_price is None:
            return 0.0

        input_cost = (usage.get("input", 0) / 1_000_000) * (input_price or 0.0)
        output_cost = (usage.get("output", 0) / 1_000_000) * (output_price or 0.0)
        return input_cost + output_cost

    def _record_usage(self, response: Any) -> None:
        usage = _get(response, "usage")
        if usage is None:
            return

        self.turn_usage["input"] += (
            _int_field(usage, "input_tokens")
            or _int_field(usage, "prompt_tokens")
        )
        self.turn_usage["output"] += (
            _int_field(usage, "output_tokens")
            or _int_field(usage, "completion_tokens")
        )

        input_details = _get(usage, "input_tokens_details") or {}
        output_details = _get(usage, "output_tokens_details") or {}
        self.turn_usage["cache_read"] += _int_field(input_details, "cached_tokens")
        self.turn_usage["cache_write"] += _int_field(input_details, "cache_creation_tokens")
        self.turn_usage["reasoning"] += _int_field(output_details, "reasoning_tokens")


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
        "anthropic-claude": "anthropic",
        "claude": "anthropic",
        "deepseek-ai": "deepseek",
        "deepseek-chat": "deepseek",
        "z-ai": "glm",
        "z.ai": "glm",
        "zai": "glm",
        "zhipu": "glm",
        "zhipuai": "glm",
        "bigmodel": "glm",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {value}. Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
    return normalized


def _default_model_for_provider(provider: str) -> str:
    defaults = {
        "openai": os.environ.get("OPENAI_MODEL") or "gpt-4o",
        "openai-compatible": os.environ.get("NYM_OPENAI_COMPAT_MODEL") or "local-model",
        "ollama": os.environ.get("OLLAMA_MODEL") or "llama3.1",
        "lmstudio": os.environ.get("LMSTUDIO_MODEL") or "local-model",
        "anthropic": os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
        "deepseek": os.environ.get("DEEPSEEK_MODEL")
        or os.environ.get("OLLAMA_DEEPSEEK_MODEL")
        or os.environ.get("OLLAMA_MODEL")
        or "deepseek-chat",
        "glm": os.environ.get("GLM_MODEL")
        or os.environ.get("ZAI_MODEL")
        or os.environ.get("OLLAMA_GLM_MODEL")
        or os.environ.get("OLLAMA_MODEL")
        or "glm-4",
    }
    return defaults[provider]


def _deepseek_client() -> tuple[OpenAI, str, str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        return (
            OpenAI(api_key=api_key, base_url=base_url),
            base_url,
            "hosted",
        )
    base_url = _ollama_base_url()
    return (
        OpenAI(api_key=os.environ.get("OLLAMA_API_KEY") or "ollama", base_url=base_url),
        base_url,
        "local",
    )


def _glm_client() -> tuple[OpenAI, str, str]:
    api_key = _first_env("GLM_API_KEY", "ZAI_API_KEY", "ZHIPUAI_API_KEY", "BIGMODEL_API_KEY")
    if api_key:
        base_url = (
            os.environ.get("GLM_BASE_URL")
            or os.environ.get("ZAI_BASE_URL")
            or os.environ.get("ZHIPUAI_BASE_URL")
            or os.environ.get("BIGMODEL_BASE_URL")
            or "https://open.bigmodel.cn/api/paas/v4"
        )
        return (
            OpenAI(api_key=api_key, base_url=base_url),
            base_url,
            "hosted",
        )
    base_url = _ollama_base_url()
    return (
        OpenAI(api_key=os.environ.get("OLLAMA_API_KEY") or "ollama", base_url=base_url),
        base_url,
        "local",
    )


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


def _provider_configuration_error(provider_name: str, env_name: str) -> str:
    return (
        f"{provider_name} is not configured. Set {env_name}, or switch to a local provider with "
        "`/provider ollama <installed-model>` or `nym --provider ollama --model <installed-model>`."
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
            f"`/provider ollama <installed-model>`. Active model: {model_name}."
        )

    if "connection" in text_lower or "refused" in text_lower or "connection error" in text_lower:
        return (
            f"Cannot reach {provider_name} at {endpoint or 'configured endpoint'}. "
            f"Start the local server or switch providers with `/provider <provider> [model]`."
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
    raw = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    raw = raw.rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def _responses_messages_to_chat(messages: list[dict[str, Any]], instructions: str) -> list[dict[str, Any]]:
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
            converted.append({"role": role, "content": _message_content_text(content)})
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
            converted.append({"role": role, "content": _message_content_text(item.get("content"))})
    return converted


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


def _chat_completion_to_response(response: Any) -> Any:
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
    return SimpleNamespace(
        output=output,
        output_text=content or "",
        usage=_get(response, "usage"),
    )


def _anthropic_message_to_response(data: dict[str, Any]) -> Any:
    output: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for item in data.get("content", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                text_parts.append(text)
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


def _int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
