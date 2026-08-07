from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import curses
import hashlib
import os
import re
import shutil
import shlex
import subprocess
import sys
import uuid
import textwrap
import threading
import tempfile
import time
import webbrowser
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from enum import Enum
import json
import urllib.error
import urllib.parse
import urllib.request

from cryptography.fernet import Fernet, InvalidToken

from .context_builder import build_stored_context
from .bundle import bundled_rust_binary
from .attachments import (
    Attachment,
    attachment_from_store,
    import_attachment,
    maintain_attachment_store,
)
from .config import AgentConfig, load_agent_config
from .gateway import create_inbound_address, start_gateway
from .language_servers import LanguageServerManager
from .llm import (
    AVAILABLE_PROVIDERS,
    LLMClient,
    UNAVAILABLE_PROVIDER_TRANSPORTS,
    _normalize_provider,
)
from .planner import (
    AgentSession,
    agent_session_from_dict,
    agent_session_to_dict,
    run_agent,
)
from .rust_tools import RustTools
from .session_store import SessionInfo, SessionStore, TokenUsage
from .skills import SkillCatalog, discover_skill_catalog
from .system_events import drain_system_events, resolve_main_system_event_session_key

if TYPE_CHECKING:
    from .gateway_impl import AgentGateway

try:
    import fcntl
except ImportError:  # pragma: no cover - Unix/WSL is the primary TUI runtime.
    fcntl = None  # type: ignore[assignment]


SESSION_LIST_LIMIT = 10
TUI_TRANSCRIPT_LIMIT = 200
USAGE_PANEL_WIDTH = 30
USAGE_PANEL_MIN_TERMINAL_WIDTH = 105
DEFAULT_AGENT_NAME = "Agent"
MAX_AGENT_NAME_CHARS = 40
DEFAULT_TUI_PASTE_KEYS = ("ctrl+v", "ctrl+shift+v", "shift+insert", "alt+v")
DEFAULT_TUI_COPY_KEYS = ("alt+c", "ctrl+y")
DEFAULT_CONTEXT_WINDOWS = {
    "gpt-5.5": 1_000_000,
    "gpt-5.5-mini": 400_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-3-5-sonnet-latest": 200_000,
    "claude-3-5-haiku-latest": 200_000,
    "llama3.1": 128_000,
    "llama3.3": 128_000,
    "qwen2.5-coder": 128_000,
    "qwen2.5-coder-7b-instruct": 128_000,
    "qwen3": 128_000,
    "qwen3.5": 128_000,
    "qwen3.6": 256_000,
    "qwen3-coder": 256_000,
    "qwen3-coder-next": 256_000,
    "deepseek-r1": 128_000,
    "deepseek-v3": 128_000,
    "deepseek-v3.2": 128_000,
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    "deepseek-v4-flash": 128_000,
    "deepseek-v4-pro": 128_000,
    "glm-4": 128_000,
    "glm-4.5": 128_000,
    "glm-4.7": 128_000,
    "glm-4.7-flash": 128_000,
    "glm-5": 128_000,
    "glm-5.1": 128_000,
    "gemma3": 128_000,
    "gemma4": 128_000,
    "codestral": 128_000,
    "codellama": 128_000,
    "gpt-oss": 128_000,
    "starcoder2": 128_000,
}


LOCAL_COMMANDS = (
    ("/model", "Choose a model or local runtime"),
    ("/name", "Show or change this agent's name"),
    ("/install", "Install an open-source/open-weight model locally"),
    ("/reasoning", "Set reasoning effort for supported models"),
    ("/skills", "Show layered workspace and personal skills"),
    ("/gateway", "Show control-plane routing and session status"),
    ("/status", "Show model, context, and session usage"),
    ("/setup", "Connect a model"),
    ("/help", "Show commands and keyboard shortcuts"),
    ("/exit", "Close Agent"),
)

PROVIDER_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "glm": "GLM_API_KEY",
    "openai-compatible": "AGENT_OPENAI_COMPAT_API_KEY",
    "voice": "AGENT_VOICE_API_KEY",
}
_CREDENTIAL_ENV_NAMES = frozenset(PROVIDER_API_KEY_ENVS.values())


def _credentials_path() -> Path:
    config_home = Path(
        os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    ).expanduser()
    return config_home / "agent" / "credentials.enc"


def _legacy_credentials_path() -> Path:
    return _credentials_path().with_name("credentials.json")


def _credential_key_path() -> Path:
    return _credentials_path().with_name("credentials.key")


def _preferences_path() -> Path:
    config_home = Path(
        os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    ).expanduser()
    return config_home / "agent" / "preferences.json"


def _load_agent_name() -> str:
    configured = os.environ.get("AGENT_NAME") or os.environ.get("NYM_AGENT_NAME")
    if configured:
        try:
            return _normalize_agent_name(configured)
        except ValueError:
            return DEFAULT_AGENT_NAME
    try:
        payload = json.loads(_preferences_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return DEFAULT_AGENT_NAME
    if not isinstance(payload, dict):
        return DEFAULT_AGENT_NAME
    try:
        return _normalize_agent_name(payload.get("agent_name"))
    except ValueError:
        return DEFAULT_AGENT_NAME


def _persist_agent_name(name: str) -> str:
    normalized = _normalize_agent_name(name)
    path = _preferences_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["agent_name"] = normalized
    serialized = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_private_write(path, serialized)
    return normalized


def _load_tui_paste_keys() -> tuple[str, ...]:
    return _load_tui_keys(
        default_keys=DEFAULT_TUI_PASTE_KEYS,
        env_names=("AGENT_TUI_PASTE_KEYS", "NYM_TUI_PASTE_KEYS"),
        preference_key="paste_keys",
    )


def _load_tui_copy_keys() -> tuple[str, ...]:
    return _load_tui_keys(
        default_keys=DEFAULT_TUI_COPY_KEYS,
        env_names=("AGENT_TUI_COPY_KEYS", "NYM_TUI_COPY_KEYS"),
        preference_key="copy_keys",
    )


def _load_tui_keys(
    *,
    default_keys: tuple[str, ...],
    env_names: tuple[str, str],
    preference_key: str,
) -> tuple[str, ...]:
    keys = list(default_keys)
    configured = next((os.environ.get(name) for name in env_names if os.environ.get(name)), None)
    raw_value: Any = None
    if configured:
        raw_value = configured.split(",")
    else:
        try:
            payload = json.loads(_preferences_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            raw_value = payload.get(preference_key)
            if raw_value is None and isinstance(payload.get("tui"), dict):
                raw_value = payload["tui"].get(preference_key)
    for key in _normalized_key_list(raw_value):
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def _load_tui_mouse_capture() -> bool:
    configured = os.environ.get("AGENT_TUI_MOUSE_CAPTURE") or os.environ.get("NYM_TUI_MOUSE_CAPTURE")
    if configured is not None:
        return _truthy_config_value(configured)
    try:
        payload = json.loads(_preferences_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return True
    if not isinstance(payload, dict):
        return True
    raw_value = payload.get("mouse_capture")
    if raw_value is None and isinstance(payload.get("tui"), dict):
        raw_value = payload["tui"].get("mouse_capture")
    return True if raw_value is None else _truthy_config_value(raw_value)


def _truthy_config_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().casefold() in {"1", "true", "yes", "on"}


def _normalized_key_list(value: Any) -> tuple[str, ...]:
    candidates = value.split(",") if isinstance(value, str) else value if isinstance(value, list) else ()
    keys: list[str] = []
    for item in candidates:
        if not isinstance(item, str):
            continue
        key = "+".join(part.strip().casefold() for part in item.split("+") if part.strip())
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _normalize_agent_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Agent name must be text.")
    name = " ".join(value.strip().split())
    if not name:
        raise ValueError("Agent name cannot be empty.")
    if len(name) > MAX_AGENT_NAME_CHARS:
        raise ValueError(f"Agent name must be {MAX_AGENT_NAME_CHARS} characters or fewer.")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError("Agent name cannot contain control characters.")
    return name


def _load_persisted_api_keys() -> None:
    payload, migrated = _read_persisted_credentials()
    for env_name, api_key in payload.items():
        if (
            env_name in _CREDENTIAL_ENV_NAMES
            and isinstance(api_key, str)
            and api_key
            and not os.environ.get(env_name)
        ):
            os.environ[env_name] = api_key
    if migrated and payload:
        try:
            _write_persisted_credentials(payload)
            _legacy_credentials_path().unlink()
        except OSError:
            pass


def _persist_api_key(env_name: str, api_key: str) -> None:
    if env_name not in _CREDENTIAL_ENV_NAMES:
        raise ValueError(f"Unsupported credential environment variable: {env_name}")
    credentials, _migrated = _read_persisted_credentials()
    credentials[env_name] = api_key
    _write_persisted_credentials(credentials)


def _read_persisted_credentials() -> tuple[dict[str, str], bool]:
    path = _credentials_path()
    try:
        encrypted = path.read_bytes()
        decrypted = _credential_cipher().decrypt(encrypted)
        payload = json.loads(decrypted.decode("utf-8"))
        return _validated_credentials(payload), False
    except (FileNotFoundError, OSError, InvalidToken, UnicodeDecodeError, json.JSONDecodeError):
        pass
    try:
        payload = json.loads(_legacy_credentials_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}, False
    return _validated_credentials(payload), True


def _validated_credentials(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if key in _CREDENTIAL_ENV_NAMES and isinstance(value, str) and value
    }


def _credential_cipher() -> Fernet:
    configured_key = os.environ.get("AGENT_CREDENTIAL_ENCRYPTION_KEY", "").strip()
    if configured_key:
        return Fernet(configured_key.encode("ascii"))
    key_path = _credential_key_path()
    try:
        key = key_path.read_bytes()
        os.chmod(key_path.parent, 0o700)
        os.chmod(key_path, 0o600)
    except FileNotFoundError:
        key = Fernet.generate_key()
        key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(key_path.parent, 0o700)
        try:
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            key = key_path.read_bytes()
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(key_path, 0o600)
    return Fernet(key)


def _write_persisted_credentials(credentials: dict[str, str]) -> None:
    serialized = json.dumps(credentials, separators=(",", ":")).encode("utf-8")
    _atomic_private_write(_credentials_path(), _credential_cipher().encrypt(serialized))


def _atomic_private_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temp_path = path.with_suffix(".tmp")
    descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise
    os.chmod(temp_path, 0o600)
    temp_path.replace(path)
    os.chmod(path, 0o600)

PROVIDER_LOGIN_URLS = {
    "copilot": "https://github.com/login/device",
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "gemini": "https://aistudio.google.com/app/apikey",
    "groq": "https://console.groq.com/keys",
    "openrouter": "https://openrouter.ai/settings/keys",
    "bedrock": "https://console.aws.amazon.com/bedrock/home",
    "azure": "https://ai.azure.com",
    "vertexai": "https://console.cloud.google.com/vertex-ai",
    "deepseek": "https://platform.deepseek.com/api_keys",
    "glm": "https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
    "openai-compatible": "https://platform.openai.com/api-keys",
    "voice": "https://platform.openai.com/api-keys",
}
PROVIDER_DISPLAY_NAMES = {
    "copilot": "GitHub Copilot",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "bedrock": "AWS Bedrock",
    "azure": "Azure OpenAI",
    "vertexai": "Google Cloud Vertex AI",
    "deepseek": "DeepSeek",
    "glm": "GLM",
    "openai-compatible": "Custom OpenAI-compatible",
    "voice": "Voice",
    "ollama": "Ollama",
    "lmstudio": "LM Studio",
    "llamacpp": "llama.cpp",
    "vllm": "vLLM",
    "localai": "LocalAI",
}
PROVIDER_ARGUMENT_COMMANDS = {"/provider", "/login", "/auth", "/apikey", "/key"}
LOCAL_PROVIDERS = {"ollama", "lmstudio", "llamacpp", "vllm", "localai"}
UNIMPLEMENTED_PROVIDER_TRANSPORTS = UNAVAILABLE_PROVIDER_TRANSPORTS
PROVIDER_SORT_ORDER = {
    "copilot": 0,
    "anthropic": 1,
    "openai": 2,
    "gemini": 3,
    "groq": 4,
    "openrouter": 5,
    "bedrock": 6,
    "azure": 7,
    "vertexai": 8,
    "deepseek": 9,
    "glm": 10,
    "ollama": 11,
    "lmstudio": 12,
    "llamacpp": 13,
    "vllm": 14,
    "localai": 15,
    "openai-compatible": 16,
}
MODEL_PICKER_SORT_ORDER = {
    "openai": 0,
    "anthropic": 1,
    "ollama": 2,
    "lmstudio": 3,
    "llamacpp": 4,
    "vllm": 5,
    "localai": 6,
    "groq": 7,
    "openrouter": 8,
    "deepseek": 9,
    "glm": 10,
    "gemini": 11,
    "copilot": 12,
    "bedrock": 13,
    "azure": 14,
    "vertexai": 15,
    "openai-compatible": 16,
}
PROVIDER_MODEL_HINTS = {
    "copilot": (
        "gpt-4.1",
        "gpt-5.4-mini",
        "claude-sonnet-4.5",
        "gemini-2.5-pro",
    ),
    "openai": (
        "gpt-5.5",
        "gpt-5.5-mini",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    ),
    "anthropic": (
        "claude-sonnet-4.5",
        "claude-opus-4.1",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ),
    "gemini": (
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
    ),
    "groq": (
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct-0905",
    ),
    "openrouter": (
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-5.4-mini",
        "google/gemini-2.5-pro",
        "meta-llama/llama-3.3-70b-instruct",
    ),
    "bedrock": (
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-opus-4-1-20250805-v1:0",
        "amazon.nova-pro-v1:0",
        "meta.llama3-3-70b-instruct-v1:0",
    ),
    "azure": (
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ),
    "vertexai": (
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "claude-sonnet-4.5",
        "claude-opus-4.1",
    ),
    "ollama": (
        "qwen2.5-coder",
        "qwen3-coder",
        "qwen3",
        "deepseek-r1",
        "llama3.3",
        "llama3.1",
        "gemma3",
        "stable-code",
    ),
    "lmstudio": ("gpt-oss-20b", "llama-3.1-8b", "local-model"),
    "llamacpp": (
        "gemma-3-1b-it",
        "local-model",
        "qwen2.5-coder-7b-instruct",
        "deepseek-coder-v2-lite-instruct",
        "codellama-13b-instruct",
    ),
    "vllm": (
        "qwen2.5-1.5b-instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Codestral-22B-v0.1",
    ),
    "localai": (
        "llama-3.2-1b-instruct",
        "hermes-2-theta-llama-3-8b",
        "deepseek-coder",
        "llama-3.1-instruct",
        "codestral",
    ),
    "openai-compatible": ("local-model",),
    "deepseek": (
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-chat",
        "deepseek-reasoner",
    ),
    "glm": ("glm-5.1", "glm-5", "glm-4.7", "glm-4.7-flash", "glm-4", "glm-4.5"),
}
LOCAL_INSTALL_CATALOG = (
    {"provider": "ollama", "model": "qwen3", "install_id": "qwen3:8b", "parameters": "8B", "size": "~5.2 GB", "memory": "8 GB+ RAM", "context": "40K", "quantization": "Ollama default"},
    {"provider": "lmstudio", "model": "gpt-oss-20b", "install_id": "openai/gpt-oss-20b", "parameters": "20B (3.6B active)", "size": "varies by quantization", "memory": "16 GB+ RAM", "context": "128K", "quantization": "choose in LM Studio"},
    {"provider": "llamacpp", "model": "gemma-3-1b-it", "install_id": "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M", "parameters": "1B", "size": "~1 GB", "memory": "4 GB+ RAM", "context": "32K", "quantization": "Q4_K_M"},
    {"provider": "vllm", "model": "qwen2.5-1.5b-instruct", "install_id": "Qwen/Qwen2.5-1.5B-Instruct", "parameters": "1.5B", "size": "~3 GB", "memory": "6 GB+ VRAM", "context": "32K", "quantization": "full precision weights"},
    {"provider": "localai", "model": "llama-3.2-1b-instruct", "install_id": "llama-3.2-1b-instruct:q4_k_m", "parameters": "1B", "size": "~1 GB", "memory": "4 GB+ RAM", "context": "128K", "quantization": "Q4_K_M"},
    {"provider": "ollama", "model": "qwen2.5-coder", "install_id": "qwen2.5-coder:7b", "parameters": "7B", "size": "~4.7 GB", "memory": "8 GB+ RAM", "context": "128K", "quantization": "Ollama default"},
    {"provider": "ollama", "model": "qwen3-coder", "install_id": "qwen3-coder:30b", "parameters": "30B (3.3B active)", "size": "~19 GB", "memory": "24 GB+ RAM", "context": "256K", "quantization": "Ollama default"},
    {"provider": "lmstudio", "model": "llama-3.1-8b", "install_id": "llama-3.1-8b@q4_k_m", "parameters": "8B", "size": "~5 GB", "memory": "8 GB+ RAM", "context": "128K", "quantization": "Q4_K_M"},
    {"provider": "llamacpp", "model": "qwen2.5-coder-7b-instruct", "install_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M", "parameters": "7B", "size": "~4.7 GB", "memory": "8 GB+ RAM", "context": "128K", "quantization": "Q4_K_M"},
    {"provider": "vllm", "model": "Qwen/Qwen2.5-Coder-32B-Instruct", "install_id": "Qwen/Qwen2.5-Coder-32B-Instruct", "parameters": "32.5B", "size": "~65 GB", "memory": "70 GB+ VRAM", "context": "128K", "quantization": "full precision weights"},
    {"provider": "vllm", "model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "install_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "parameters": "16B (2.4B active)", "size": "~32 GB", "memory": "36 GB+ VRAM", "context": "128K", "quantization": "full precision weights"},
    {"provider": "ollama", "model": "deepseek-r1", "install_id": "deepseek-r1:8b", "parameters": "8B", "size": "~5.2 GB", "memory": "8 GB+ RAM", "context": "128K", "quantization": "Ollama default"},
    {"provider": "ollama", "model": "llama3.3", "install_id": "llama3.3:70b", "parameters": "70B", "size": "~43 GB", "memory": "48 GB+ RAM", "context": "128K", "quantization": "Ollama default"},
    {"provider": "ollama", "model": "llama3.1", "install_id": "llama3.1:8b", "parameters": "8B", "size": "~4.9 GB", "memory": "8 GB+ RAM", "context": "128K", "quantization": "Ollama default"},
    {"provider": "ollama", "model": "gemma3", "install_id": "gemma3:4b", "parameters": "4B", "size": "~3.3 GB", "memory": "6 GB+ RAM", "context": "128K", "quantization": "Ollama default"},
    {"provider": "ollama", "model": "stable-code", "install_id": "stable-code:3b", "parameters": "3B", "size": "~1.6 GB", "memory": "4 GB+ RAM", "context": "16K", "quantization": "Ollama default"},
)
COLOR_HEADER = 1
COLOR_USER = 2
COLOR_ASSISTANT = 3
COLOR_THINKING = 4
COLOR_GUARDRAIL = 5
COLOR_MUTED = 6
COLOR_ERROR = 7


@dataclass
class AppContext:
    workspace_root: Path
    search_roots: list[Path]
    rust: RustTools
    llm: LLMClient
    language_servers: LanguageServerManager
    session: AgentSession
    session_id: str
    store: SessionStore
    stored_context: str | None = None
    debug: bool = False
    config: AgentConfig = field(default_factory=AgentConfig)
    skills: SkillCatalog = field(default_factory=SkillCatalog)
    gateway: AgentGateway | None = None
    agent_id: str = "main"
    agent_name: str = DEFAULT_AGENT_NAME
    tool_allowlist: tuple[str, ...] | None = None
    route_key: str | None = None
    config_path: Path | None = None

    pending_provider: str | None = None
    pending_model: str | None = None
    last_local_command_result: dict[str, Any] | None = None
    pending_attachments: list[Attachment] = field(default_factory=list)


class LocalCommandText(str):
    """Human-readable command output with a typed UI result contract."""

    def __new__(
        cls,
        text: str,
        *,
        code: str = "ok",
        setup_required: bool = False,
        error: bool = False,
        secret_provider: str | None = None,
        next_command: str | None = None,
    ) -> "LocalCommandText":
        value = super().__new__(cls, text)
        value.command_result = {
            "code": code,
            "setup_required": setup_required,
            "error": error,
            **({"secret_provider": secret_provider} if secret_provider else {}),
            **({"next_command": next_command} if next_command else {}),
        }
        return value


def _command_text(
    text: str,
    *,
    code: str,
    setup_required: bool = False,
    error: bool = False,
    secret_provider: str | None = None,
    next_command: str | None = None,
) -> LocalCommandText:
    return LocalCommandText(
        text,
        code=code,
        setup_required=setup_required,
        error=error,
        secret_provider=secret_provider,
        next_command=next_command,
    )


def _prepend_command_text(prefix: str, result: str) -> LocalCommandText:
    metadata = getattr(
        result,
        "command_result",
        {"code": "ok", "setup_required": False, "error": False},
    )
    return _command_text(f"{prefix}\n{result}", **metadata)

@dataclass(frozen=True)
class PaletteEntry:
    value: str
    label: str
    description: str
    complete_to: str
    execute: bool = False

class AccessMode(Enum):
    NO_AUTH = "no_auth"
    API_KEY = "api_key"
    BROWSER_LOGIN = "browser_login"
    OPTIONAL_API_KEY = "optional_api_key"


@dataclass(frozen=True, slots=True)
class ModelRecord:
    id: str
    display_name: str
    provider_id: str
    access: "AccessMode"
    state: "ModelState"
    capabilities: frozenset[str]
    context_window: int | None
    local: bool
    installed: bool | None
    selectable: bool
    action_label: str | None = None

class ModelState(Enum):
    READY = "ready"
    AUTH_REQUIRED = "auth_required"
    LOGIN_REQUIRED = "login_required"
    SERVER_OFFLINE = "server_offline"
    MODEL_NOT_INSTALLED = "model_not_installed"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"

@dataclass
class ApprovalQueueState:
    ctx: AppContext
    lock: threading.Condition = field(default_factory=threading.Condition, init=False, repr=False)
    selected_index: int = 0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            pending = self.pending_items()
            selected = min(self.selected_index, max(0, len(pending) - 1))
            return {
                "pending": pending,
                "selected_index": selected,
            }

    def pending_items(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.ctx.session.pending_approvals
            if isinstance(item, dict) and item.get("status") == "pending"
        ]

    def request(self, request: dict[str, Any]) -> str:
        normalized = self._normalize_request(request)
        with self.lock:
            self._upsert_request(normalized)
            self.selected_index = len(self.pending_items()) - 1
            persist_agent_state(self.ctx)
            while self._request_status(normalized["id"]) == "pending":
                self.lock.wait()
            return self._request_decision(normalized["id"]) or "denied"

    def approve_selected(self) -> bool:
        return self._decide_selected("approved")

    def deny_selected(self) -> bool:
        return self._decide_selected("denied")

    def next_item(self) -> None:
        with self.lock:
            pending = self.pending_items()
            if pending:
                self.selected_index = min(len(pending) - 1, self.selected_index + 1)

    def previous_item(self) -> None:
        with self.lock:
            pending = self.pending_items()
            if pending:
                self.selected_index = max(0, self.selected_index - 1)

    def _decide_selected(self, decision: str) -> bool:
        with self.lock:
            pending = self.pending_items()
            if not pending:
                return False
            self.selected_index = min(self.selected_index, len(pending) - 1)
            request = pending[self.selected_index]
            request["status"] = "approved" if decision == "approved" else "denied"
            request["decision"] = decision
            request["decision_at"] = datetime.now(timezone.utc).isoformat()
            persist_agent_state(self.ctx)
            self.lock.notify_all()
            return True

    def _upsert_request(self, request: dict[str, Any]) -> None:
        request_id = str(request.get("id") or uuid.uuid4().hex)
        request["id"] = request_id
        request["status"] = "pending"
        request.setdefault("decision", None)
        request.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        approvals = [item for item in self.ctx.session.pending_approvals if _approval_item_id(item) != request_id]
        approvals.append(request)
        self.ctx.session.pending_approvals = approvals

    def _request_status(self, request_id: str) -> str:
        request = self._request_by_id(request_id)
        if request is None:
            return "denied"
        return str(request.get("status") or "denied")

    def _request_decision(self, request_id: str) -> str | None:
        request = self._request_by_id(request_id)
        if request is None:
            return None
        decision = request.get("decision")
        return str(decision) if isinstance(decision, str) and decision else None

    def _request_by_id(self, request_id: str) -> dict[str, Any] | None:
        for item in self.ctx.session.pending_approvals:
            if _approval_item_id(item) == request_id:
                return item
        return None

    def _normalize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(request)
        normalized.setdefault("id", uuid.uuid4().hex)
        normalized.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        normalized.setdefault("status", "pending")
        return normalized


def _approval_item_id(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("id")
        if isinstance(value, str):
            return value
    return ""


@dataclass
class LiveTurnState:
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    active: bool = False
    phase: str = "idle"
    prompt: str = ""
    feed: list[tuple[str, str]] = field(default_factory=list)
    _reasoning_active: bool = field(default=False, init=False, repr=False)
    _text_buf: str = field(default="", init=False, repr=False)
    _current_tool: str | None = field(default=None, init=False, repr=False)
    error: str | None = None

    def start(self, prompt: str) -> None:
        with self.lock:
            self.active = True
            self.phase = "thinking"
            self.prompt = prompt
            self.feed = []
            self._reasoning_active = False
            self._text_buf = ""
            self._current_tool = None
            self.error = None

    def _flush_reasoning(self) -> None:
        self._reasoning_active = False

    def _flush_text(self) -> None:
        text = self._text_buf.strip()
        if text:
            self.feed.append(("text", text))
        self._text_buf = ""

    def _drop_reasoning(self) -> None:
        self.feed = [(kind, content) for kind, content in self.feed if kind not in {"thinking", "reasoning"}]

    def update(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        delta = event.get("delta")
        with self.lock:
            if kind == "reasoning_delta" and isinstance(delta, str):
                # Never retain or render raw chain-of-thought. The UI shows a
                # concise activity state and user-facing results instead.
                self._reasoning_active = True
                self.phase = "reasoning"
            elif kind == "reasoning_started":
                self._reasoning_active = True
                self.phase = "reasoning"
            elif kind == "text_delta" and isinstance(delta, str):
                self._drop_reasoning()
                self._flush_reasoning()
                self._text_buf += delta
                self.phase = "responding"
            elif kind == "tool_call_started":
                self._flush_text()
                name = event.get("name", "")
                self._current_tool = name if isinstance(name, str) else ""
                self.phase = "tool_call"
            elif kind == "tool_call_arguments_done":
                label = _tool_activity_label(self._current_tool)
                self.feed.append(("tool", label))
                self._current_tool = None
                self.phase = "tool_call"
            elif kind == "tool_result":
                self._flush_reasoning()
                self._flush_text()
                self.feed.append(_live_tool_result_feed_item(event))
                self.phase = "observing"
            elif isinstance(kind, str) and kind.startswith("subagent_"):
                self._flush_reasoning()
                self._flush_text()
                summary = _subagent_activity_label(event)
                if summary:
                    self.feed.append(("subagent", summary))
                self.phase = "subagents"
            elif kind == "approval_request":
                self._flush_reasoning()
                self._flush_text()
                summary = event.get("summary")
                if isinstance(summary, str) and summary:
                    self.feed.append(("guardrail", summary))
                self.phase = "observing"
            elif kind == "approval_decision":
                self._flush_reasoning()
                self._flush_text()
                summary = event.get("summary")
                if isinstance(summary, str) and summary:
                    self.feed.append(("guardrail", summary))
                self.phase = "observing"
            elif kind == "response_completed":
                self._flush_reasoning()
                self._flush_text()
                # A model response may be followed by tool execution and another
                # model step. The worker marks the whole agent turn complete.
                self.phase = "working"

    def finish(self, error: str | None = None) -> None:
        with self.lock:
            self._flush_reasoning()
            self._flush_text()
            # Ordinary tool traces are ephemeral, but retain the most recent
            # parallel orchestration summary until the next turn so fast worker
            # batches remain inspectable after the final answer arrives.
            self.feed = [
                (kind, content)
                for kind, content in self.feed
                if kind == "subagent"
            ][-12:]
            self.prompt = ""
            self._current_tool = None
            self.active = False
            self.phase = "error" if error else "completed"
            self.error = error

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            feed_snapshot = list(self.feed)
            if self._reasoning_active and not self._text_buf:
                feed_snapshot.append(("thinking", "Reasoning"))
            if self._text_buf:
                feed_snapshot.append(("text", self._text_buf))
            if self._current_tool:
                feed_snapshot.append(("tool", _tool_activity_label(self._current_tool)))
            return {
                "active": self.active,
                "phase": self.phase,
                "prompt": self.prompt,
                "feed": feed_snapshot,
                "error": self.error,
            }


def _tool_activity_label(name: str | None) -> str:
    tool = (name or "tool").strip()
    labels = {
        "read_path": "Reading files",
        "list_path": "Listing files",
        "path_status": "Checking path status",
        "inspect_tree": "Exploring the workspace",
        "inspect_target": "Inspecting a target",
        "glob": "Finding files",
        "grep": "Searching code",
        "language_server": "Checking code intelligence",
        "write_file": "Writing a file",
        "edit_file": "Editing a file",
        "delete_path": "Deleting a path",
        "run_system_command": "Running a command",
        "system_info": "Inspecting the system",
        "connected_devices": "Checking connected devices",
        "desktop_capabilities": "Checking desktop capabilities",
        "desktop_observe": "Observing the desktop",
        "desktop_resolve": "Resolving desktop target",
        "process_list": "Checking running processes",
        "desktop_action": "Performing a desktop action",
        "parallel_subagents": "Spawning parallel subagents",
        "load_skill": "Loading a skill",
        "finish_task": "Preparing the response",
    }
    return labels.get(tool, f"Running {tool}")


def _live_tool_result_feed_item(event: dict[str, Any]) -> tuple[str, str]:
    name = event.get("name")
    summary = event.get("summary")
    observation = event.get("observation")
    label = str(name) if isinstance(name, str) and name else "tool"

    if isinstance(observation, dict):
        if observation.get("blocked"):
            reason = observation.get("reason")
            guidance = observation.get("guidance") or observation.get("error")
            detail = guidance if isinstance(guidance, str) and guidance else summary
            if isinstance(reason, str) and reason:
                return ("guardrail", f"{reason}: {detail or label}")
            return ("guardrail", str(detail or f"{label} blocked"))
        if observation.get("ok") is False:
            error = observation.get("error")
            return ("tool_error", str(error or summary or f"{label} failed"))

    return ("tool_result", str(summary or f"{label} completed"))


def _subagent_activity_label(event: dict[str, Any]) -> str:
    """Render lifecycle fields without requiring display text in the protocol."""
    summary = event.get("summary")
    detail = str(summary) if isinstance(summary, str) and summary else ""
    kind = event.get("kind")
    task_id = event.get("task_id")
    if isinstance(task_id, str) and task_id:
        status = event.get("status")
        if kind in {"subagent_task_started", "subagent_task_progress"}:
            status = "running"
        state = str(status) if isinstance(status, str) and status else "working"
        return f"{task_id} · {state}" + (f" — {detail}" if detail else "")
    return detail


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Agent CLI coding agent",
    )

    parser.add_argument(
        "--root",
        default=None,
        help="Workspace root. Defaults to current directory.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Agent JSON config path. Defaults to user and workspace .agent/config.json files.",
    )

    parser.add_argument(
        "--channel",
        default=None,
        help="Route this invocation through a durable named channel session.",
    )

    parser.add_argument(
        "--account-id",
        default="default",
        help="Channel account identity used for deterministic session routing.",
    )

    parser.add_argument(
        "--sender-id",
        default=None,
        help="Sender identity for per-sender channel session scope.",
    )

    parser.add_argument(
        "--peer-kind",
        choices=("direct", "group", "channel"),
        default=None,
        help="Optional channel peer kind used by routing bindings.",
    )

    parser.add_argument(
        "--peer-id",
        default=None,
        help="Optional channel peer id used by routing bindings.",
    )

    parser.add_argument("--guild-id", default=None, help="Optional guild id used by routing bindings.")
    parser.add_argument("--team-id", default=None, help="Optional team id used by routing bindings.")

    parser.add_argument(
        "--rust-bin",
        default=None,
        help="Path to agent-rust binary. Defaults to the built binary in the repository.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model to use for the selected provider. Defaults to the provider's configured model.",
    )

    parser.add_argument(
        "--provider",
        default=None,
        choices=sorted(AVAILABLE_PROVIDERS),
        help=(
            "LLM provider. Defaults to AGENT_LLM_PROVIDER or openai. "
            "Use ollama, lmstudio, llamacpp, vllm, or localai for no-login local models."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information.",
    )

    parser.add_argument(
        "--tui",
        action="store_true",
        help="Start the terminal UI instead of the line prompt.",
    )

    parser.add_argument(
        "--tui-bridge",
        choices=(
            "snapshot", "submit", "stream-submit", "complete", "gateway",
            "approve", "deny", "voice-record", "voice-stream", "voice-speak",
        ),
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--bridge-session-id",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--bridge-prompt",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--bridge-request-id",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "session_command",
        nargs="?",
        help="Use 'resume' or pass a session id/prefix to resume.",
    )

    parser.add_argument(
        "session_id",
        nargs="?",
        help="Session id/prefix when using 'resume'.",
    )

    return parser


def resolve_rust_bin(
    args: argparse.Namespace,
    workspace_root: Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    if getattr(args, "rust_bin", None):
        rust_bin = Path(args.rust_bin).expanduser()
        if not rust_bin.is_absolute():
            rust_bin = workspace_root / rust_bin
        return rust_bin.resolve()

    bundled = bundled_rust_binary()
    if bundled is not None:
        return bundled.resolve()

    repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    candidate_roots = [
        repo_root,
        workspace_root.resolve(),
        workspace_root.resolve().parent,
    ]

    for candidate_root in candidate_roots:
        for candidate in (
            candidate_root / "agent-rust" / "target" / "release" / "agent-rust",
            candidate_root / "agent-rust" / "target" / "debug" / "agent-rust",
            candidate_root / "target" / "release" / "agent-rust",
            candidate_root / "target" / "debug" / "agent-rust",
        ):
            if candidate.exists():
                return candidate.resolve()

    return (repo_root / "agent-rust" / "target" / "debug" / "agent-rust").resolve()


def build_context(
    args: argparse.Namespace,
    *,
    store: SessionStore,
    session_info: SessionInfo,
) -> AppContext:
    maintain_attachment_store()
    workspace_root = Path(session_info.workspace_root).expanduser().resolve()
    explicit_config = getattr(args, "config", None)
    config_path = (
        Path(explicit_config).expanduser().resolve()
        if explicit_config
        else None
    )
    config = load_agent_config(workspace_root, explicit_path=config_path)
    stored_agent_id = session_info.agent if session_info.agent in config.agents else None
    agent_id = stored_agent_id or config.default_agent_id
    tool_allowlist = config.tool_allowlist(agent_id)
    skills = discover_skill_catalog(
        workspace_root,
        config,
        agent_id=agent_id,
        tool_allowlist=tool_allowlist,
    )
    rust_bin = resolve_rust_bin(
        args,
        workspace_root,
        repo_root=Path(__file__).resolve().parents[1],
    )
    session = load_agent_session(session_info)
    stored_context = build_stored_context(
        store=store,
        session=session_info,
    ).text
    provider = getattr(args, "provider", None) or session_info.provider
    model = _selected_model(args, session_info)
    llm = LLMClient(model=model, provider=provider)
    if session.reasoning_effort and llm.reasoning_effort is not None:
        llm.reasoning_effort = session.reasoning_effort
    if session_info.provider != llm.provider or session_info.model != llm.model:
        store.update_llm_config(session_info.id, provider=llm.provider, model=llm.model)

    return AppContext(
        workspace_root=workspace_root,
        search_roots=parse_search_roots(workspace_root),
        rust=RustTools(rust_bin=rust_bin),
        llm=llm,
        language_servers=LanguageServerManager(),
        session=session,
        session_id=session_info.id,
        store=store,
        pending_attachments=[
            attachment
            for item in session.pending_attachments
            if (attachment := attachment_from_store(item)) is not None
        ],
        stored_context=stored_context,
        debug=args.debug,
        config=config,
        skills=skills,
        gateway=start_gateway(config=config, store=store),
        agent_id=agent_id,
        agent_name=_load_agent_name(),
        tool_allowlist=tool_allowlist,
        route_key=_session_route_key(store, session_info.id),
        config_path=config_path,
    )


def _session_route_key(store: SessionStore, session_id: str) -> str | None:
    try:
        routes = store.list_routes_for_session(session_id)
    except (AttributeError, OSError):
        return None
    return routes[0].route_key if routes else None


def _selected_model(args: argparse.Namespace, session_info: SessionInfo) -> str | None:
    if getattr(args, "model", None):
        return args.model
    provider_override = getattr(args, "provider", None)
    if provider_override and provider_override != session_info.provider:
        return None
    return session_info.model


def default_workspace_root(args: argparse.Namespace) -> Path:
    if args.root:
        return Path(args.root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if cwd == Path("/") and (repo_root / "agent-rust").exists() and (repo_root / "README.md").exists():
        return repo_root.resolve()

    if cwd.name == "agent":
        repo_candidates = [cwd.parent, cwd]
    elif cwd.name == "agent":
        repo_candidates = [cwd.parent.parent, cwd.parent, cwd]
    else:
        repo_candidates = [cwd]

    for candidate in repo_candidates:
        if (candidate / "agent-rust").exists() and (candidate / "README.md").exists():
            return candidate.resolve()

    return cwd


def parse_search_roots(_workspace_root: Path) -> list[Path]:
    return []


def load_session_messages(store: SessionStore, session_id: str) -> list[dict[str, Any]]:
    messages = store.list_messages(session_id, limit=20)
    return [
        {
            "role": message.role,
            "content": message.content,
            "attachments": [
                {
                    "id": item.id, "filename": item.filename, "mime": item.mime,
                    "size_bytes": item.size_bytes, "storage_path": item.storage_path,
                    "source": item.source,
                }
                for item in message.attachments
            ],
        }
        for message in messages
        if message.role in {"user", "assistant"}
    ]


def handle_prompt(
    ctx: AppContext,
    prompt: str,
    *,
    stream_event: Callable[[dict[str, Any]], None] | None = None,
    approval_requester: Callable[[dict[str, Any]], str] | None = None,
) -> str:
    gateway = getattr(ctx, "gateway", None)
    lease = (
        gateway.session_lease(ctx.session_id)
        if gateway is not None
        else nullcontext()
    )
    with lease:
        return _run_prompt_turn(
            ctx,
            prompt,
            stream_event=stream_event,
            approval_requester=approval_requester,
        )


def _run_prompt_turn(
    ctx: AppContext,
    prompt: str,
    *,
    stream_event: Callable[[dict[str, Any]], None] | None = None,
    approval_requester: Callable[[dict[str, Any]], str] | None = None,
) -> str:
    conversation_history = load_session_messages(ctx.store, ctx.session_id)
    route_key = getattr(ctx, "route_key", None)
    write_guard = {"expected_route_key": route_key} if route_key else {}
    pending_attachments = getattr(ctx, "pending_attachments", None)
    current_attachments = list(pending_attachments or ())
    if pending_attachments is not None:
        pending_attachments.clear()
    ctx.session.pending_attachments.clear()
    if current_attachments:
        user_messages = [ctx.store.add_message_with_attachments(
            ctx.session_id, "user", prompt,
            [item.to_store_input() for item in current_attachments],
            last_prompt=prompt,
            **write_guard,
        )]
    else:
        user_messages = ctx.store.add_messages(
            ctx.session_id, [("user", prompt)], last_prompt=prompt, **write_guard,
        )
    persist_agent_state(ctx)
    _emit_transcript_update(ctx, user_messages)
    ctx.store.add_event(
        ctx.session_id,
        event_type="turn_started",
        summary=f"User prompt: {truncate(prompt, 260)}",
        data={"prompt": prompt},
    )
    ctx.store.add_event(
        ctx.session_id,
        event_type="assistant_stream_started",
        summary="Assistant stream started",
        data={"prompt": prompt},
    )
    _emit_gateway_hook(ctx, "turn_started", {
        "session_id": ctx.session_id,
        "route_key": getattr(ctx, "route_key", None),
        "agent_id": getattr(ctx, "agent_id", "main"),
    })
    ctx.llm.reset_turn_usage()
    agent_prompt = _prompt_with_system_events(ctx, prompt)
    try:
        try:
            answer = run_agent(
                llm=ctx.llm,
                rust=ctx.rust,
                workspace_root=str(ctx.workspace_root),
                search_roots=[str(root) for root in ctx.search_roots],
                user_prompt=agent_prompt,
                user_visible_prompt=prompt,
                session=ctx.session,
                stored_context=ctx.stored_context,
                conversation_history=conversation_history,
                current_attachments=[item.to_store_input() for item in current_attachments],
                record_event=lambda **kwargs: ctx.store.add_event(ctx.session_id, **kwargs),
                stream_event=stream_event,
                approval_requester=approval_requester,
                language_servers=ctx.language_servers,
                skill_catalog=getattr(ctx, "skills", None),
                tool_allowlist=getattr(ctx, "tool_allowlist", None),
                debug=ctx.debug,
            )
        except Exception as exc:
            _emit_gateway_hook(ctx, "turn_failed", {
                "session_id": ctx.session_id,
                "route_key": getattr(ctx, "route_key", None),
                "error": str(exc),
            })
            raise
    finally:
        usage = ctx.llm.consume_turn_usage()
        ctx.store.add_usage(
            ctx.session_id,
            tokens=TokenUsage(
                input=usage.get("input", 0),
                output=usage.get("output", 0),
                reasoning=usage.get("reasoning", 0),
                cache_read=usage.get("cache_read", 0),
                cache_write=usage.get("cache_write", 0),
            ),
            cost_usd=ctx.llm.estimate_cost_usd(usage),
        )

    ctx.store.add_event(
        ctx.session_id,
        event_type="assistant_answer",
        summary=f"Assistant answer: {truncate(answer, 260)}",
        data={"answer": answer},
    )
    ctx.store.add_event(
        ctx.session_id,
        event_type="assistant_stream_completed",
        summary="Assistant stream completed",
        data={"answer": answer},
    )
    assistant_message = ctx.store.add_message(ctx.session_id, "assistant", answer, **write_guard)
    _emit_transcript_update(ctx, [assistant_message])
    persist_agent_state(ctx)
    ctx.stored_context = None
    _emit_gateway_hook(ctx, "turn_completed", {
        "session_id": ctx.session_id,
        "route_key": getattr(ctx, "route_key", None),
    })
    return answer


def _prompt_with_system_events(ctx: AppContext, prompt: str) -> str:
    events: list[str] = []
    seen_keys: set[str] = set()
    for key in (
        getattr(ctx, "route_key", None),
        resolve_main_system_event_session_key(getattr(ctx, "config", None)),
        "global",
    ):
        if not isinstance(key, str) or not key.strip() or key in seen_keys:
            continue
        seen_keys.add(key)
        events.extend(drain_system_events(key))
    if not events:
        return prompt
    event_text = "\n".join(f"- {event}" for event in events)
    return f"System events since the last prompt:\n{event_text}\n\nUser request: {prompt}"


def _emit_gateway_hook(ctx: AppContext, event: str, payload: dict[str, Any]) -> None:
    gateway = getattr(ctx, "gateway", None)
    hooks = getattr(gateway, "hooks", None)
    emit = getattr(hooks, "emit", None)
    if callable(emit):
        emit(event, payload)


def _emit_transcript_update(ctx: AppContext, messages: list[Any]) -> None:
    """Publish a post-commit transcript update for hook subscribers."""
    if not messages:
        return
    message = messages[-1]
    route_key = getattr(ctx, "route_key", None)
    agent_id = getattr(ctx, "agent_id", "main")
    attachment_items = tuple(getattr(message, "attachments", ()) or ())
    message_payload = {"role": message.role, "content": message.content}
    if attachment_items:
        message_payload["attachments"] = [
            {"filename": item.filename, "mime": item.mime, "size_bytes": item.size_bytes}
            for item in attachment_items
        ]
    payload = {
        "session_id": ctx.session_id,
        "route_key": route_key,
        "agent_id": agent_id,
        "message": message_payload,
        "message_id": str(message.id),
        "message_seq": message.seq,
    }
    if route_key:
        payload["target"] = {
            "agent_id": agent_id,
            "session_id": ctx.session_id,
            "session_key": route_key,
        }
    _emit_gateway_hook(ctx, "session_transcript_updated", payload)


def _record_local_command_exchange(ctx: AppContext, prompt: str, answer: str) -> None:
    """Persist slash-command output so every UI sees the same transcript."""
    # Composer controls use these commands as a private bridge protocol.  They
    # should update the pending-attachment state without making a command
    # exchange look like part of the user's conversation.
    if _is_attachment_bridge_command(prompt):
        return
    logged_prompt = _redact_local_command(prompt)
    route_key = getattr(ctx, "route_key", None)
    write_guard = {"expected_route_key": route_key} if route_key else {}
    messages = ctx.store.add_messages(
        ctx.session_id,
        [
            ("user", logged_prompt),
            ("assistant", answer),
        ],
        last_prompt=logged_prompt,
        **write_guard,
    )
    _emit_transcript_update(ctx, messages)


def _is_attachment_bridge_command(prompt: str) -> bool:
    try:
        parts = shlex.split(prompt)
    except ValueError:
        return False
    return bool(parts) and parts[0].casefold() == "/__nym_attach"


def _cli_approval_requester(request: dict[str, Any]) -> str:
    operation = _approval_text(request.get("operation")).strip().replace("_", " ")
    target = _approval_display_text(request) or "requested target"
    action = operation or _approval_text(request.get("tool")).strip() or "action"
    try:
        decision = input(f"Allow {action} on {target!r} once? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return "denied"
    return "approved" if decision.strip().casefold() in {"y", "yes"} else "denied"


def repl(ctx: AppContext) -> int:
    print("Agent started.")
    print(f"Session: {ctx.session_id}")
    print("Type '/exit' to quit.")
    print()

    try:
        while True:
            try:
                user_input = input("agent> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not user_input:
                continue

            if _is_exit_command(user_input):
                return 0

            try:
                answer = _handle_local_command(ctx, user_input)
                if answer is None:
                    answer = handle_prompt(
                        ctx,
                        user_input,
                        approval_requester=_cli_approval_requester,
                    )
                else:
                    _record_local_command_exchange(ctx, user_input, answer)
                print(answer)
            except Exception as exc:
                if ctx.debug:
                    raise
                print(f"Error: {exc}")
    finally:
        _stop_language_servers(ctx)

    return 0


def create_new_session(args: argparse.Namespace, store: SessionStore) -> SessionInfo:
    workspace_root = default_workspace_root(args)
    config = _load_startup_config(args, workspace_root)
    channel = getattr(args, "channel", None)
    if channel:
        routed = start_gateway(config=config, store=store).open_session(
            create_inbound_address(
                channel=channel,
                account_id=getattr(args, "account_id", None) or "default",
                sender_id=getattr(args, "sender_id", None),
                peer_kind=getattr(args, "peer_kind", None),
                peer_id=getattr(args, "peer_id", None),
                guild_id=getattr(args, "guild_id", None),
                team_id=getattr(args, "team_id", None),
            ),
            workspace_root=workspace_root,
            provider=args.provider,
            model=args.model,
        )
        return routed.session
    provider = args.provider
    model = args.model
    if provider is None and model is None:
        remembered = _latest_workspace_llm_config(
            store,
            workspace_root,
            config=config,
            agent_id=config.default_agent_id,
        )
        if remembered is not None:
            provider, model = remembered
    return store.create_session(
        workspace_root=workspace_root,
        provider=provider,
        model=model,
        agent_id=config.default_agent_id,
    )


def _load_startup_config(args: argparse.Namespace, workspace_root: Path) -> AgentConfig:
    explicit_config = getattr(args, "config", None)
    config_path = (
        Path(explicit_config).expanduser().resolve()
        if explicit_config
        else None
    )
    return load_agent_config(workspace_root, explicit_path=config_path)


def _latest_workspace_llm_config(
    store: SessionStore,
    workspace_root: Path,
    *,
    config: AgentConfig,
    agent_id: str,
) -> tuple[str | None, str | None] | None:
    workspace_key = _workspace_root_key(workspace_root)
    for session in store.list_sessions(limit=None):
        if _workspace_root_key(session.workspace_root) != workspace_key:
            continue
        if _effective_session_agent_id(session, config) != agent_id:
            continue
        if session.provider or session.model:
            return session.provider, session.model
    return None


def _effective_session_agent_id(session: SessionInfo, config: AgentConfig) -> str:
    return session.agent if session.agent in config.agents else config.default_agent_id


def _workspace_root_key(path: Path | str) -> str:
    raw = str(path).strip()
    wsl_key = _wsl_mount_workspace_key(raw)
    if wsl_key is not None:
        return wsl_key
    drive_key = _windows_drive_workspace_key(raw)
    if drive_key is not None:
        return drive_key
    resolved = str(Path(path).expanduser().resolve())
    wsl_key = _wsl_mount_workspace_key(resolved)
    if wsl_key is not None:
        return wsl_key
    drive_key = _windows_drive_workspace_key(resolved)
    if drive_key is not None:
        return drive_key
    return os.path.normcase(resolved).replace("\\", "/")


def _wsl_mount_workspace_key(value: str) -> str | None:
    normalized = value.replace("\\", "/")
    match = re.match(r"^/mnt/([A-Za-z])/(.+)$", normalized)
    if not match:
        return None
    drive, rest = match.groups()
    return f"{drive}:/{rest}".rstrip("/").casefold()


def _windows_drive_workspace_key(value: str) -> str | None:
    normalized = value.replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.+)$", normalized)
    if not match:
        return None
    drive, rest = match.groups()
    return f"{drive}:/{rest}".rstrip("/").casefold()


def choose_session(store: SessionStore) -> SessionInfo | None:
    sessions = store.list_sessions(limit=SESSION_LIST_LIMIT)
    if not sessions:
        print("No sessions found.")
        return None

    print("Resume a previous session")
    print()

    for index, item in enumerate(sessions, start=1):
        title = item.last_prompt or item.title
        print(f"{index:>2}. {item.id}  {format_age(item.updated_at):>8}  {truncate(title, 96)}")

    print()
    choice = input("Select session number/id, or press Enter to cancel: ").strip()
    if not choice:
        return None

    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(sessions):
            return sessions[index - 1]
        raise ValueError(f"Invalid session number: {choice}")

    session_id = store.resolve_session_id(choice)
    return store.get_session(session_id)


def load_existing_session(command: str, store: SessionStore) -> SessionInfo:
    session_id = store.resolve_session_id(command)
    return store.get_session(session_id)


def load_agent_session(session_info: SessionInfo) -> AgentSession:
    return agent_session_from_dict(session_info.state)


def persist_agent_state(ctx: AppContext) -> None:
    ctx.store.save_agent_state(
        ctx.session_id,
        agent_session_to_dict(ctx.session),
    )


def _handle_local_command(
    ctx: AppContext,
    user_input: str,
    *,
    install_progress: Callable[[str], None] | None = None,
) -> str | None:
    result = _dispatch_local_command(
        ctx,
        user_input,
        install_progress=install_progress,
    )
    if result is None:
        ctx.last_local_command_result = None
        return None
    metadata = getattr(
        result,
        "command_result",
        {"code": "ok", "setup_required": False, "error": False},
    )
    ctx.last_local_command_result = dict(metadata)
    return str(result)


def _dispatch_local_command(
    ctx: AppContext,
    user_input: str,
    *,
    install_progress: Callable[[str], None] | None = None,
) -> str | None:
    text = _normalized_command_prompt(user_input.strip())
    if not text.startswith("/"):
        return None
    if text == "/":
        return _slash_help_text()
    if text.casefold().startswith("/name "):
        return _set_agent_name_command(ctx, _unquote_agent_name_argument(text[6:]))
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        return f"Invalid command quoting: {exc}"
    command = parts[0].casefold()

    if command == "/__nym_attach":
        if len(parts) != 2:
            return "The selected file could not be added."
        try:
            attachment = import_attachment(parts[1], source="user_file")
        except ValueError as exc:
            return str(exc)
        ctx.pending_attachments.append(attachment)
        ctx.session.pending_attachments.append(attachment.to_store_input())
        persist_agent_state(ctx)
        return f"Attached for next message: {attachment.filename} ({attachment.mime}, {attachment.size_bytes} bytes)."

    if command in {"/login", "/auth"}:
        provider = parts[1] if len(parts) >= 2 else _active_provider(ctx)
        return _login_provider(ctx, provider)

    if command in {"/apikey", "/key"}:
        provider = parts[1] if len(parts) >= 2 else ""
        api_key = parts[2] if len(parts) >= 3 else ""
        if not provider:
            return "Usage: /apikey <provider> [api-key]"
        if not api_key:
            try:
                normalized = _normalize_provider(provider)
            except ValueError as exc:
                return str(exc)
            env_name = PROVIDER_API_KEY_ENVS.get(normalized)
            api_key = os.environ.get(env_name, "") if env_name else ""
            if not api_key:
                return _command_text(
                    f"Paste the key using the hidden TUI prompt: /apikey {provider}\n"
                    f"Non-TUI fallback: /apikey {provider} <api-key>",
                    code="api_key_required",
                    setup_required=True,
                    secret_provider=normalized,
                )
        return _set_provider_api_key(ctx, provider, api_key)

    if command == "/name":
        if len(parts) == 1:
            name = getattr(ctx, "agent_name", DEFAULT_AGENT_NAME)
            return f"Agent name: {name}\nChange it with: /name <new name>"
        return _set_agent_name_command(ctx, " ".join(parts[1:]))

    if command in {"/models", "/model"} and len(parts) == 1:
        return _models_text(ctx)

    if command == "/model":
        if len(parts) < 2:
            return _models_text(ctx)
        provider: str | None = None
        if len(parts) >= 3:
            try:
                provider = _normalize_provider(parts[1])
                model = parts[2]
            except ValueError:
                provider = None
                model = parts[1]
        else:
            model = parts[1]
        return _switch_model(ctx, model=model, provider=provider)

    if command == "/install":
        if len(parts) not in {3, 4} or (len(parts) == 4 and parts[3].casefold() not in {"--yes", "--confirm"}):
            return "Usage: /install <provider> <model> [--yes]"
        return _install_local_model(
            ctx,
            provider=parts[1],
            model=parts[2],
            progress=install_progress,
            confirmed=len(parts) == 4,
        )

    if command == "/reasoning":
        if len(parts) == 1:
            effort = getattr(getattr(ctx, "llm", None), "reasoning_effort", None)
            if effort is None:
                return f"Reasoning effort is provider-controlled for `{ctx.llm.model}`."
            return f"Reasoning effort: {effort}\nChange with: /reasoning minimal|low|medium|high"
        if len(parts) != 2:
            return "Usage: /reasoning minimal|low|medium|high"
        effort = parts[1].casefold()
        if effort not in {"minimal", "low", "medium", "high"}:
            return "Reasoning effort must be one of: minimal, low, medium, high."
        if getattr(getattr(ctx, "llm", None), "reasoning_effort", None) is None:
            return f"`{ctx.llm.model}` does not expose configurable reasoning effort through this provider."
        ctx.llm.reasoning_effort = effort
        ctx.session.reasoning_effort = effort
        persist_agent_state(ctx)
        return f"Reasoning effort set to {effort} for {_model_source_label(_active_provider(ctx))} · {ctx.llm.model}."

    if command == "/status":
        return _status_text(ctx)

    if command == "/tools":
        return "Tools are managed automatically. Describe what you need; Agent selects the right capability and asks before sensitive actions."

    if command == "/skills":
        return ctx.skills.status_text()

    if command == "/gateway":
        return _gateway_text(ctx)

    if command in {"/setup", "/connect"}:
        return _setup_text(ctx, parts[1] if len(parts) == 2 else None)

    if command == "/help":
        return _slash_help_text()

    if command in {"/exit", "/quit", "/q"}:
        return "Exiting Agent."

    return f"Unknown local command: {parts[0]}"


def _slash_help_text() -> str:
    lines = ["Commands"]
    lines.extend(f"{name} - {description}" for name, description in LOCAL_COMMANDS)
    lines.append("")
    lines.append("Type / to open the command menu. Use Up/Down to select and Tab to complete.")
    lines.append("Enter sends · Esc exits · PgUp/PgDn scrolls")
    return "\n".join(lines)


def _provider_switch_text(ctx: AppContext) -> str:
    provider = _active_provider(ctx)
    configuration = _llm_configuration(ctx)
    configuration_state = _llm_configuration_state(ctx)
    lines = [f"Model source switched to {_model_source_label(provider)} with model {ctx.llm.model}."]
    if configuration_state == "ready":
        lines.append("Configuration: ready")
        return "\n".join(lines)

    env_name = PROVIDER_API_KEY_ENVS.get(provider)
    display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider)
    if configuration_state == "endpoint_required":
        lines.extend([
            "",
            "OpenAI-compatible models need an endpoint before requests can run.",
            "Set environment variable: AGENT_OPENAI_COMPAT_BASE_URL",
            "Optional API key: /apikey openai-compatible",
        ])
        return "\n".join(lines)

    if configuration_state == "api_key_required" and env_name:
        lines.extend([
            "",
            f"{display_name} needs an API key before requests can run.",
            f"Set key: /apikey {provider}",
        ])
        if provider in PROVIDER_LOGIN_URLS:
            lines.append(f"Open account/API keys: /login {provider}")
        lines.extend([
            f"Environment variable: {env_name}",
            "Keys loaded with /apikey are used for this Agent process and are not written to session history.",
        ])
        return "\n".join(lines)

    lines.append(f"Configuration: {configuration}")
    return "\n".join(lines)


def _providers_text(ctx: AppContext) -> str:
    provider = _active_provider(ctx)
    providers = ", ".join(sorted(AVAILABLE_PROVIDERS))
    return (
        f"Active model: {ctx.llm.model}\n"
        f"Model source: {_model_source_label(provider)}\n"
        f"Mode: {_llm_mode(ctx)}\n"
        f"Endpoint: {_llm_endpoint(ctx)}\n"
        f"Configuration: {_llm_configuration(ctx)}\n"
        f"Available model sources: {providers}\n"
        "Switch with: /model <source> <model>"
    )


def _models_text(ctx: AppContext) -> str:
    active_provider = _active_provider(ctx)
    options = _model_options_for_display(ctx)

    lines = [
        f"Active model: {ctx.llm.model}",
        f"Model source: {_model_source_label(active_provider)}",
        "",
        "Models:",
        "Hosted models use their named provider and may require provider credentials.",
        "Open-source models use a local runtime and are installed on this computer; they never require login.",
    ]

    _append_provider_model_text(lines, options, ctx)

    lines.extend([
        "",
        "Switch model: /model <model>",
        "Choose exact source/model: /model <source> <model>",
        "Hosted providers ask for a key only when one is required.",
        "Install locally: /install <provider> <model>.",
        "Supported installers: Ollama, LM Studio, llama.cpp, vLLM, and LocalAI.",
    ])

    return "\n".join(lines)


def _append_provider_model_text(
    lines: list[str],
    options: list[dict[str, Any]],
    ctx: Any,
) -> None:
    if not options:
        return
    active_provider = _active_provider(ctx)
    active_model = getattr(getattr(ctx, "llm", None), "model", None)
    lines.append("")
    current_provider: str | None = None
    for option in options:
        provider = option["provider"]
        if provider != current_provider:
            current_provider = provider
            lines.append("")
            lines.append(
                f"{_model_source_label(provider)} - {_provider_access_label(provider)}"
            )
        marker = (
            "*"
            if option["provider"] == active_provider and option["model"] == active_model
            else " "
        )
        lines.append(
            f"{marker} {option['model']}  "
            f"{_model_state_label(option)}{_model_metadata_suffix(option['provider'], option['model'])}"
        )

def _resolve_local_model_name(
    ctx: Any,
    provider: str,
    requested_model: str,
) -> tuple[str | None, str | None]:
    discovered, discovery_error = _discover_provider_models(
        ctx,
        provider,
    )

    if discovery_error:
        return None, _local_model_setup_error(provider, requested_model, discovery_error)

    # Exact installed model name.
    if requested_model in discovered:
        return requested_model, None

    requested_lower = requested_model.casefold()

    # Allow an untagged alias only when it resolves to one model.
    matches = [
        model
        for model in discovered
        if model.casefold() == requested_lower
        or model.casefold().startswith(
            f"{requested_lower}:"
        )
    ]

    if len(matches) == 1:
        return matches[0], None

    if len(matches) > 1:
        choices = ", ".join(matches)

        return None, (
            f"`{requested_model}` matches multiple installed models: "
            f"{choices}. Choose the exact model name."
        )

    installed = ", ".join(discovered) or "none"

    message = (
        f"{_model_source_label(provider)} · {requested_model}\n"
        "Status: model not installed\n"
        f"Installed models: {installed}."
    )
    install_entry = _local_install_entry(provider, requested_model)
    if install_entry is not None:
        message = (
            f"{message}\n"
            f"Install locally now: /install {provider} {requested_model}"
        )
        if provider == "ollama":
            message = f"{message}\nTerminal alternative: ollama pull {requested_model}"
    elif provider == "lmstudio":
        message = f"{message}\nDownload or load it in LM Studio first; no login is required."
    else:
        message = f"{message}\n{_manual_local_install_text(provider, requested_model)}"
    return None, _command_text(
        message,
        code="model_not_installed",
        setup_required=True,
        next_command=(
            f"/install {provider} {requested_model}"
            if install_entry is not None
            else None
        ),
    )


def _local_install_entry(provider: str, model: str) -> dict[str, str] | None:
    return next(
        (
            entry
            for entry in LOCAL_INSTALL_CATALOG
            if entry["provider"] == provider and entry["model"].casefold() == model.casefold()
        ),
        None,
    )


def _local_model_setup_error(provider: str, model: str, detail: str) -> str:
    source = _model_source_label(provider)
    if provider == "ollama":
        text = (
            f"{source} · {model}\n"
            "Status: runtime unavailable\n"
            "Start Ollama if it is installed. Otherwise install it from:\n"
            "https://ollama.com/download\n"
            f"Once it is running, install the model here: /install ollama {model}\n"
            f"Details: {detail}"
        )
        return _command_text(text, code="runtime_unavailable", setup_required=True)
    if provider == "lmstudio":
        text = (
            f"{source} · {model}\n"
            "Status: runtime unavailable\n"
            "Install LM Studio, load the model, and start its local server first.\n"
            "Download: https://lmstudio.ai/download\n"
            f"Details: {detail}"
        )
        return _command_text(text, code="runtime_unavailable", setup_required=True)
    if provider in {"llamacpp", "vllm", "localai"}:
        install_entry = _local_install_entry(provider, model)
        setup_hint = (
            f"Preview/install locally: /install {provider} {model}"
            if install_entry is not None
            else _manual_local_install_text(provider, model)
        )
        text = (
            f"{source} · {model}\n"
            "Status: runtime unavailable\n"
            f"Start the {source} OpenAI-compatible server with that model loaded.\n"
            f"{setup_hint}\n"
            f"Details: {detail}"
        )
        return _command_text(text, code="runtime_unavailable", setup_required=True)
    return detail


def _switch_model(
    ctx: Any,
    *,
    model: str,
    provider: str | None = None,
) -> str:
    resolved_provider = provider or _provider_for_model(
        model,
        _active_provider(ctx),
    )

    if resolved_provider in UNIMPLEMENTED_PROVIDER_TRANSPORTS:
        source = _model_source_label(resolved_provider)
        return _command_text(
            "\n".join([
                f"{source} · {model}",
                "Status: unavailable",
                f"{source} transport is not implemented in this Agent build.",
                "No credentials were requested or changed.",
            ]),
            code="unavailable",
            setup_required=True,
            error=True,
        )

    if resolved_provider in LOCAL_PROVIDERS:
        resolved_model, setup_error = _resolve_local_model_name(
            ctx,
            resolved_provider,
            model,
        )
        if setup_error:
            return setup_error
        if resolved_model:
            model = resolved_model

    try:
        candidate = LLMClient(
            model=model,
            provider=resolved_provider,
        )
        _apply_saved_reasoning_effort(ctx, candidate)
    except Exception as exc:
        return _command_text(
            f"Could not use model `{model}`: {exc}",
            code="unavailable",
            setup_required=True,
            error=True,
        )

    configuration_error = getattr(
        candidate,
        "configuration_error",
        None,
    )

    if configuration_error:
        configuration_state = _candidate_configuration_state(candidate)
        # Local models should show server/install instructions,
        # not an API-key prompt.
        if resolved_provider in LOCAL_PROVIDERS:
            return _command_text(
                _handle_model_setup(candidate),
                code="runtime_unavailable",
                setup_required=True,
            )

        # Make the selected hosted model active even while it waits for
        # credentials. TUI bridge commands run in separate processes, so an
        # in-memory-only pending selection would snap back to the prior model
        # as soon as this command completes.
        previous_llm = ctx.llm
        try:
            ctx.llm = candidate
            _persist_llm_config(ctx)
        except Exception as exc:
            ctx.llm = previous_llm
            return _command_text(
                f"Could not save model `{model}`: {exc}",
                code="unavailable",
                setup_required=True,
                error=True,
            )

        ctx.pending_provider = resolved_provider
        ctx.pending_model = str(getattr(candidate, "model", None) or model)

        missing_api_key = configuration_state == "api_key_required"
        browser_setup = configuration_state == "credentials_required"
        opened_url = (
            _open_provider_setup_url(resolved_provider)
            if missing_api_key or browser_setup
            else None
        )
        open_line = None
        if opened_url:
            opened, url = opened_url
            verb = "Opened" if opened else "Open"
            page_kind = "API-key" if missing_api_key else "setup"
            open_line = f"{verb} {_model_source_label(resolved_provider)} {page_kind} page: {url}"

        lines = [f"{_model_source_label(resolved_provider)} · {model}"]
        if missing_api_key:
            lines.extend([
                "Status: API key required",
                f"Set key: /apikey {resolved_provider}",
            ])
        elif browser_setup:
            lines.extend([
                "Status: provider sign-in or cloud credentials required",
                f"Details: {configuration_error}",
            ])
        else:
            lines.extend([
                "Status: unavailable",
                f"Details: {configuration_error}",
            ])
        if open_line:
            lines.append(open_line)
        if missing_api_key:
            lines.append(f"Complete the secure {_model_source_label(resolved_provider)} key prompt to continue.")
        code = (
            "api_key_required"
            if missing_api_key
            else "credentials_required"
            if browser_setup
            else "unavailable"
        )
        return _command_text(
            "\n".join(lines),
            code=code,
            setup_required=True,
            error=code == "unavailable",
            secret_provider=resolved_provider if missing_api_key else None,
        )

    previous_llm = ctx.llm

    try:
        ctx.llm = candidate
        ctx.pending_provider = None
        ctx.pending_model = None
        _persist_llm_config(ctx)
    except Exception as exc:
        ctx.llm = previous_llm
        return _command_text(
            f"Could not save model `{model}`: {exc}",
            code="unavailable",
            setup_required=True,
            error=True,
        )

    return _command_text(_model_switch_text(ctx), code="ok")


def _install_local_model(
    ctx: Any,
    *,
    provider: str,
    model: str,
    progress: Callable[[str], None] | None = None,
    confirmed: bool = False,
) -> str:
    try:
        normalized = _normalize_provider(provider)
    except ValueError as exc:
        return str(exc)

    if normalized not in LOCAL_PROVIDERS:
        return (
            f"{_model_source_label(normalized)} is hosted; models are not installed locally. "
            f"Choose it with: /model {normalized} {model}"
        )

    install_entry = _local_install_entry(normalized, model)
    if install_entry is None and normalized in {"llamacpp", "localai", "vllm"} and "/" not in model:
        source = _model_source_label(normalized)
        available = ", ".join(
            entry["model"]
            for entry in LOCAL_INSTALL_CATALOG
            if entry["provider"] == normalized
        )
        identifier_kind = {
            "llamacpp": "Hugging Face GGUF repository",
            "localai": "LocalAI gallery entry with known size metadata",
            "vllm": "Hugging Face model repository",
        }[normalized]
        return _command_text(
            "Status: model is not in the install catalog\n"
            f"Choose a listed {source} model: {available or 'none'}\n"
            f"Or provide a full {identifier_kind} identifier after `/install {normalized}`.\n"
            "Agent will not preview or start an unknown-size local download.",
            code="model_not_installed",
            setup_required=True,
        )
    install_id = str(install_entry["install_id"] if install_entry else model)

    if not confirmed:
        return _local_install_preview(normalized, model, install_entry)

    if normalized != "ollama":
        return _install_non_ollama_model(
            ctx,
            provider=normalized,
            model=model,
            install_id=install_id,
            progress=progress,
        )

    if shutil.which("ollama") is None:
        return _command_text(
            "Status: runtime not installed\n"
            "Ollama runtime is not installed.\n"
            "Install Ollama: https://ollama.com/download\n"
            "Start Ollama, then run this command again:\n"
            f"/install ollama {model}",
            code="runtime_not_installed",
            setup_required=True,
            next_command=f"/install ollama {model}",
        )

    if progress is not None:
        progress("Ollama · checking local runtime")

    runtime_error = _ensure_ollama_running(ctx, progress=progress)
    if runtime_error:
        return runtime_error

    if progress is not None:
        progress(f"Ollama · downloading {model} locally")

    try:
        process = subprocess.Popen(
            ["ollama", "pull", install_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except OSError as exc:
        return _command_text(
            f"Status: install failed\nCould not start Ollama: {exc}",
            code="install_failed",
            setup_required=True,
            error=True,
        )

    output_tail = ""
    stdout = process.stdout
    if stdout is not None:
        while True:
            chunk = stdout.read(256)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            output_tail = (output_tail + text)[-8000:]
            summary = _clean_install_progress(text)
            if summary and progress is not None:
                progress(summary)

    returncode = process.wait()
    if returncode != 0:
        detail = _clean_install_progress(output_tail) or "Ollama pull failed."
        return _command_text(
            "Status: install failed\n"
            f"Could not install `{model}` with Ollama.\n"
            f"{truncate(detail, 1000)}\n"
            f"Retry: /install ollama {model}",
            code="install_failed",
            setup_required=True,
            error=True,
            next_command=f"/install ollama {model}",
        )

    verification_error = _verify_local_model_ready(ctx, "ollama", model)
    if verification_error:
        return _command_text(
            "Status: install could not be verified\n"
            f"Ollama reported a successful pull for `{model}`, but Agent could not verify it locally.\n"
            f"{verification_error}\n"
            f"Retry: /install ollama {model}",
            code="install_unverified",
            setup_required=True,
            error=True,
            next_command=f"/install ollama {model}",
        )

    if progress is not None:
        progress(f"Ollama · installed {model}; selecting model")
    switch_result = _switch_model(ctx, model=model, provider="ollama")
    return _prepend_command_text(f"Installed `{model}` with Ollama.", switch_result)


def _local_install_preview(
    provider: str,
    model: str,
    entry: dict[str, str] | None,
) -> str:
    source = _model_source_label(provider)
    metadata = entry or {
        "install_id": model,
        "parameters": "unknown",
        "size": "unknown",
        "memory": "unknown",
        "context": "unknown",
        "quantization": "provider default",
    }
    next_command = f"/install {provider} {model} --yes"
    return _command_text("\n".join([
        "Local model install preview",
        f"Model: {model}",
        f"Provider: {source}",
        f"Exact artifact: {metadata['install_id']}",
        f"Parameters: {metadata['parameters']}",
        f"Download: {metadata['size']}",
        f"Recommended memory: {metadata['memory']}",
        f"Context: {metadata['context']}",
        f"Quantization: {metadata['quantization']}",
        "Location: local runtime storage on this computer",
        "Authentication: none",
        "",
        f"Confirm download: {next_command}",
        "Nothing has been downloaded yet.",
        "During installation, press Esc or Ctrl+C to stop without closing Agent.",
    ]), code="install_confirmation_required", next_command=next_command)


def _install_non_ollama_model(
    ctx: Any,
    *,
    provider: str,
    model: str,
    install_id: str,
    progress: Callable[[str], None] | None,
) -> str:
    source = _model_source_label(provider)
    command: list[str]
    runtime_url: str

    if provider == "lmstudio":
        if shutil.which("lms") is None:
            return _runtime_not_installed_text(
                source,
                "https://lmstudio.ai/download",
                provider,
                model,
            )
        command = ["lms", "get", install_id]
        runtime_url = "https://lmstudio.ai/docs/cli/local-models/get"
    elif provider == "localai":
        if shutil.which("local-ai") is None:
            return _runtime_not_installed_text(
                source,
                "https://localai.io/installation/",
                provider,
                model,
            )
        command = ["local-ai", "models", "install", install_id]
        runtime_url = "https://localai.io/models/"
    elif provider == "llamacpp":
        llama_cli = shutil.which("llama-cli")
        if llama_cli is None:
            return _runtime_not_installed_text(
                source,
                "https://github.com/ggml-org/llama.cpp/releases",
                provider,
                model,
            )
        command = [llama_cli, "-hf", install_id, "-p", "", "-n", "1"]
        runtime_url = "https://github.com/ggml-org/llama.cpp"
    elif provider == "vllm":
        if shutil.which("vllm") is None:
            return _runtime_not_installed_text(
                source,
                "https://docs.vllm.ai/en/latest/getting_started/installation/",
                provider,
                model,
            )
        hf_cli = shutil.which("hf")
        if hf_cli is None:
            return _command_text(
                "Status: runtime helper unavailable\n"
                "vLLM is installed, but the Hugging Face `hf` download command is unavailable.\n"
                f"Install huggingface_hub, then retry: /install {provider} {model}",
                code="runtime_unavailable",
                setup_required=True,
                next_command=f"/install {provider} {model}",
            )
        command = [hf_cli, "download", install_id]
        runtime_url = "https://docs.vllm.ai/en/latest/models/supported_models/"
    else:
        return _manual_local_install_text(provider, model)

    ok, detail = _run_local_install_command(
        command,
        label=f"{source} · downloading {model} locally",
        progress=progress,
    )
    if not ok:
        return _command_text(
            "Status: install failed\n"
            f"Could not install `{model}` with {source}.\n"
            f"{detail}\n"
            f"Provider instructions: {runtime_url}",
            code="install_failed",
            setup_required=True,
            error=True,
        )

    activation_error = _activate_local_runtime(
        ctx,
        provider=provider,
        model=model,
        install_id=install_id,
        progress=progress,
    )
    if activation_error:
        return _command_text(
            f"Download command completed for `{model}` with {source}.\n"
            "Status: install not ready\n"
            f"{activation_error}\n"
            f"Then select it with: /model {provider} {model}",
            code="install_not_ready",
            setup_required=True,
            next_command=f"/model {provider} {model}",
        )

    verification_error = _verify_local_model_ready(ctx, provider, model)
    if verification_error:
        return _command_text(
            "Status: install could not be verified\n"
            f"{source} completed its download command, but `{model}` is not exposed by its local API.\n"
            f"{verification_error}\n"
            f"Retry: /install {provider} {model}",
            code="install_unverified",
            setup_required=True,
            error=True,
            next_command=f"/install {provider} {model}",
        )

    if progress is not None:
        progress(f"{source} · installed locally; selecting {model}")
    switch_result = _switch_model(ctx, model=model, provider=provider)
    return _prepend_command_text(f"Installed `{model}` locally with {source}.", switch_result)


def _verify_local_model_ready(ctx: Any, provider: str, model: str) -> str | None:
    discovered, discovery_error = _discover_provider_models(ctx, provider)
    if discovery_error:
        return f"Could not query {_model_source_label(provider)}: {discovery_error}"
    if not _local_model_is_available(model, discovered):
        available = ", ".join(discovered) or "none"
        return f"Local API models: {available}."
    return None


def _run_local_install_command(
    command: list[str],
    *,
    label: str,
    progress: Callable[[str], None] | None,
) -> tuple[bool, str]:
    if progress is not None:
        progress(label)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except OSError as exc:
        return False, str(exc)

    output_tail = ""
    if process.stdout is not None:
        while True:
            chunk = process.stdout.read(256)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            output_tail = (output_tail + text)[-8000:]
            summary = _clean_install_progress(text)
            if summary and progress is not None:
                progress(summary)
    returncode = process.wait()
    detail = _clean_install_progress(output_tail)
    return returncode == 0, detail or ("Installation completed." if returncode == 0 else "Installation failed.")


def _runtime_not_installed_text(
    source: str,
    download_url: str,
    provider: str,
    model: str,
) -> str:
    return _command_text(
        "Status: runtime not installed\n"
        f"{source} is not installed on this computer.\n"
        f"Install the local runtime: {download_url}\n"
        f"Then retry: /install {provider} {model}",
        code="runtime_not_installed",
        setup_required=True,
        next_command=f"/install {provider} {model}",
    )


def _clean_install_progress(value: str) -> str:
    without_ansi = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    segments = [
        segment.strip()
        for segment in re.split(r"[\r\n]+", without_ansi)
        if segment.strip()
    ]
    if not segments:
        return ""
    return truncate(" ".join(segments[-2:]), 160)


def _ensure_ollama_running(
    ctx: Any,
    *,
    progress: Callable[[str], None] | None = None,
) -> str | None:
    _models, discovery_error = _discover_provider_models(ctx, "ollama")
    if not discovery_error:
        return None

    base_url = _provider_base_url(ctx, "ollama")
    hostname = (urllib.parse.urlparse(_normalize_base_url(base_url)).hostname or "").casefold()
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        return _command_text(
            "Status: runtime unavailable\n"
            f"Could not reach the configured Ollama server at {base_url}.\n"
            f"Details: {discovery_error}",
            code="runtime_unavailable",
            setup_required=True,
        )

    if progress is not None:
        progress("Ollama · starting local runtime")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return _command_text(
            f"Status: runtime unavailable\nCould not start Ollama locally: {exc}",
            code="runtime_unavailable",
            setup_required=True,
            error=True,
        )

    last_error = discovery_error
    for _attempt in range(40):
        time.sleep(0.25)
        _models, last_error = _discover_provider_models(ctx, "ollama")
        if not last_error:
            if progress is not None:
                progress("Ollama · local runtime started")
            return None

    return _command_text(
        "Status: runtime unavailable\n"
        "Ollama is installed but its local server did not start.\n"
        "Run `ollama serve` in another terminal, then retry the install.\n"
        f"Details: {last_error}",
        code="runtime_unavailable",
        setup_required=True,
    )


def _activate_local_runtime(
    ctx: Any,
    *,
    provider: str,
    model: str,
    install_id: str,
    progress: Callable[[str], None] | None,
) -> str | None:
    source = _model_source_label(provider)
    if progress is not None:
        progress(f"{source} · starting local model server")

    if provider == "lmstudio":
        load_ok, load_detail = _run_local_install_command(
            ["lms", "load", install_id, "--identifier", model],
            label=f"{source} · loading {model}",
            progress=progress,
        )
        if not load_ok:
            return f"LM Studio downloaded the model but could not load it: {load_detail}"
        server_ok, server_detail = _run_local_install_command(
            ["lms", "server", "start"],
            label=f"{source} · starting local API server",
            progress=progress,
        )
        if not server_ok:
            return f"LM Studio could not start its local server: {server_detail}"
    else:
        base_url = _provider_base_url(ctx, provider)
        parsed = urllib.parse.urlparse(_normalize_base_url(base_url))
        hostname = (parsed.hostname or "").casefold()
        if hostname not in {"localhost", "127.0.0.1", "::1"}:
            return f"The configured {source} endpoint is remote ({base_url}); start that runtime on its host."
        port = parsed.port or {
            "llamacpp": 8080,
            "vllm": 8000,
            "localai": 8080,
        }[provider]
        if provider == "llamacpp":
            server = shutil.which("llama-server")
            if server is None:
                return "`llama-server` is not installed or not on PATH."
            command = [server, "-hf", install_id, "--alias", model, "--port", str(port)]
        elif provider == "vllm":
            command = [
                "vllm",
                "serve",
                install_id,
                "--served-model-name",
                model,
                "--port",
                str(port),
            ]
        elif provider == "localai":
            command = ["local-ai", "run", install_id, "--address", f":{port}"]
        else:
            return f"No automatic local server activation is configured for {source}."

        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return f"Could not start {source}: {exc}"

    last_error: str | None = None
    for _attempt in range(120):
        models, discovery_error = _discover_provider_models(ctx, provider)
        if not discovery_error and _local_model_is_available(model, models):
            if progress is not None:
                progress(f"{source} · local server ready")
            return None
        if discovery_error:
            last_error = discovery_error
        else:
            available = ", ".join(models) or "none"
            last_error = f"server is online, but `{model}` is not ready (API models: {available})"
        time.sleep(0.5)
    return f"{source} did not become ready: {last_error or 'unknown startup error'}"


def _manual_local_install_text(provider: str, model: str) -> str:
    source = _model_source_label(provider)
    if provider == "lmstudio":
        text = (
            "Status: manual setup required\n"
            f"Automatic installation is not available for {source}.\n"
            "Open LM Studio, download and load the model, then start the local server.\n"
            f"After it is loaded: /model lmstudio {model}"
        )
        return _command_text(text, code="manual_setup_required", setup_required=True)
    if provider == "llamacpp":
        text = (
            "Status: manual setup required\n"
            f"Automatic installation is not available for {source}.\n"
            "Download a compatible GGUF model and start llama.cpp with it.\n"
            f"Then run: /model llamacpp {model}"
        )
        return _command_text(text, code="manual_setup_required", setup_required=True)
    if provider == "vllm":
        text = (
            "Status: manual setup required\n"
            f"Start vLLM with `{model}` so it can download/load the model from its configured registry.\n"
            f"Then run: /model vllm {model}"
        )
        return _command_text(text, code="manual_setup_required", setup_required=True)
    if provider == "localai":
        text = (
            "Status: manual setup required\n"
            f"Install `{model}` through the LocalAI model gallery and start the LocalAI server.\n"
            f"Then run: /model localai {model}"
        )
        return _command_text(text, code="manual_setup_required", setup_required=True)
    return _command_text(
        f"{source} cannot install `{model}` automatically.",
        code="manual_setup_required",
        setup_required=True,
    )


def _normalize_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")

    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    for suffix in (
        "/chat/completions",
        "/responses",
        "/models",
    ):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")

    return url


def _strip_api_suffix(base_url: str) -> str:
    url = _normalize_base_url(base_url)

    if url.endswith("/v1"):
        return url[:-3].rstrip("/")

    return url


def _openai_models_url(base_url: str) -> str:
    url = _normalize_base_url(base_url)

    if not url:
        return ""

    if not url.endswith("/v1"):
        url = f"{url}/v1"

    return f"{url}/models"


def _ollama_tags_url(base_url: str) -> str:
    root = _strip_api_suffix(base_url)

    if not root:
        return ""

    return f"{root}/api/tags"


def _ollama_ps_url(base_url: str) -> str:
    root = _strip_api_suffix(base_url)

    if not root:
        return ""

    return f"{root}/api/ps"


def _get_json(
    url: str,
    *,
    api_key: str | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    if not url:
        raise RuntimeError("Endpoint is not configured.")

    headers = {
        "Accept": "application/json",
        "User-Agent": "Agent/1.0",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url,
        headers=headers,
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw = response.read().decode(
                "utf-8",
                errors="replace",
            )

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        if exc.code in {401, 403}:
            raise RuntimeError(
                "Authentication failed."
            ) from exc

        if exc.code == 404:
            raise RuntimeError(
                f"Model-list endpoint was not found: {url}"
            ) from exc

        raise RuntimeError(
            f"Endpoint returned HTTP {exc.code}: {body}"
        ) from exc

    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)

        raise RuntimeError(
            f"Could not connect to {url}: {reason}"
        ) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Endpoint returned invalid JSON: {url}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Endpoint returned an unexpected response: {url}"
        )

    return payload


def _discover_ollama_models(
    base_url: str,
) -> list[str]:
    payload = _get_json(
        _ollama_tags_url(base_url),
    )

    items = payload.get("models", [])

    if not isinstance(items, list):
        return []

    models: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        model = item.get("model") or item.get("name")

        if isinstance(model, str) and model:
            models.append(model)

    return sorted(set(models))


def _discover_ollama_loaded_models(
    base_url: str,
) -> list[dict[str, Any]]:
    payload = _get_json(
        _ollama_ps_url(base_url),
    )

    items = payload.get("models", [])

    if not isinstance(items, list):
        return []

    models: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        name = item.get("model") or item.get("name")
        if not isinstance(name, str) or not name:
            continue

        detail = item.get("details")
        details = detail if isinstance(detail, dict) else {}
        models.append({
            "model": name,
            "size": item.get("size"),
            "size_vram": item.get("size_vram"),
            "expires_at": item.get("expires_at"),
            "parameters": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
        })

    return sorted(models, key=lambda item: str(item["model"]).casefold())


def _discover_openai_compatible_models(
    base_url: str,
    *,
    api_key: str | None = None,
) -> list[str]:
    payload = _get_json(
        _openai_models_url(base_url),
        api_key=api_key,
    )

    items = payload.get("data", [])

    if not isinstance(items, list):
        return []

    models: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        model = item.get("id")

        if isinstance(model, str) and model:
            models.append(model)

    return sorted(set(models))


def _provider_base_url(
    ctx: Any,
    provider: str,
) -> str:
    # First use the endpoint already resolved by LLMClient.
    active_llm = getattr(ctx, "llm", None)

    if (
        active_llm is not None
        and getattr(active_llm, "provider", None) == provider
    ):
        endpoint = getattr(active_llm, "endpoint", None)

        if isinstance(endpoint, str) and endpoint.strip():
            return endpoint.strip()

    # Then use provider-specific environment configuration.
    if provider == "ollama":
        return os.environ.get(
            "AGENT_OLLAMA_BASE_URL",
            os.environ.get(
                "OLLAMA_HOST",
                "http://localhost:11434",
            ),
        )

    if provider == "lmstudio":
        return os.environ.get(
            "AGENT_LMSTUDIO_BASE_URL",
            os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        )

    if provider == "llamacpp":
        return os.environ.get(
            "AGENT_LLAMACPP_BASE_URL",
            os.environ.get("LLAMACPP_BASE_URL", "http://localhost:8080/v1"),
        )

    if provider == "vllm":
        return os.environ.get(
            "AGENT_VLLM_BASE_URL",
            os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        )

    if provider == "localai":
        return os.environ.get(
            "AGENT_LOCALAI_BASE_URL",
            os.environ.get("LOCALAI_BASE_URL", "http://localhost:8080/v1"),
        )

    if provider == "openai-compatible":
        return os.environ.get(
            "AGENT_OPENAI_COMPAT_BASE_URL",
            "",
        )

    if provider == "openai":
        return "https://api.openai.com/v1"

    if provider == "groq":
        return os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    if provider == "openrouter":
        return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    return ""



def _discover_provider_models(
    ctx: Any,
    provider: str,
) -> tuple[list[str], str | None]:
    base_url = _provider_base_url(ctx, provider)

    try:
        if provider == "ollama":
            return _discover_ollama_models(base_url), None

        if provider in {"lmstudio", "llamacpp", "vllm", "localai"}:
            key_env = {
                "lmstudio": "LMSTUDIO_API_KEY",
                "llamacpp": "LLAMACPP_API_KEY",
                "vllm": "VLLM_API_KEY",
                "localai": "LOCALAI_API_KEY",
            }[provider]
            return _discover_openai_compatible_models(
                base_url,
                api_key=os.environ.get(key_env),
            ), None

        if provider == "openai-compatible":
            if not base_url:
                return [], "Endpoint is not configured."

            api_key = os.environ.get(
                "AGENT_OPENAI_COMPAT_API_KEY",
            )

            return (
                _discover_openai_compatible_models(
                    base_url,
                    api_key=api_key,
                ),
                None,
            )

        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")

            if not api_key:
                return [], "OpenAI API key is required."

            return (
                _discover_openai_compatible_models(
                    "https://api.openai.com/v1",
                    api_key=api_key,
                ),
                None,
            )

        if provider in {"groq", "openrouter"}:
            api_key = os.environ.get(PROVIDER_API_KEY_ENVS.get(provider, ""))

            if not api_key:
                return [], f"{_model_source_label(provider)} API key is required."

            return (
                _discover_openai_compatible_models(
                    base_url,
                    api_key=api_key,
                ),
                None,
            )

        return list(PROVIDER_MODEL_HINTS.get(provider, ())), None

    except RuntimeError as exc:
        return [], str(exc)


def _model_state_label(option: dict[str, Any]) -> str:
    state = option.get("state")

    labels = {
        ModelState.READY: "Ready",
        ModelState.AUTH_REQUIRED: "Add API key",
        ModelState.LOGIN_REQUIRED: "Sign in",
        ModelState.SERVER_OFFLINE: "Server offline",
        ModelState.MODEL_NOT_INSTALLED: "Not installed",
        ModelState.DOWNLOADING: "Downloading",
        ModelState.LOADING: "Loading",
        ModelState.UNAVAILABLE: "Unavailable",
        ModelState.INCOMPATIBLE: "Incompatible",
        ModelState.UNKNOWN: "Setup required",
    }

    return labels.get(state, "Unknown")


def _handle_model_setup(candidate: Any) -> str:
    provider = str(getattr(candidate, "provider", "") or "")
    model = str(getattr(candidate, "model", "") or "")
    error = str(
        getattr(candidate, "configuration_error", "")
        or "Model is not ready."
    )

    if provider == "ollama":
        return (
            f"`{model}` is not ready locally.\n"
            f"Start Ollama, then install it with: ollama pull {model}"
        )

    if provider == "lmstudio":
        return (
            f"`{model}` is not ready locally.\n"
            "Start the LM Studio server and load the model first."
        )

    if provider in {"llamacpp", "vllm", "localai"}:
        return (
            f"`{model}` is not ready locally.\n"
            f"Start the {_model_source_label(provider)} server with the model loaded; no login is required."
        )

    if provider in PROVIDER_API_KEY_ENVS:
        return (
            f"`{model}` needs an API key.\n"
            f"Details: {error}"
        )

    return f"Could not prepare `{model}`: {error}"    

def _model_switch_text(ctx: Any) -> str:
    provider = _active_provider(ctx)
    model = getattr(getattr(ctx, "llm", None), "model", None)
    if _llm_configuration_state(ctx) == "ready":
        return f"{_model_source_label(provider)} · {model}"
    return _provider_switch_text(ctx)


def _apply_saved_reasoning_effort(ctx: Any, llm: Any) -> None:
    effort = getattr(getattr(ctx, "session", None), "reasoning_effort", None)
    if effort in {"minimal", "low", "medium", "high"} and getattr(llm, "reasoning_effort", None) is not None:
        llm.reasoning_effort = effort


def _open_provider_setup_url(provider: str) -> tuple[bool, str] | None:
    if provider in LOCAL_PROVIDERS:
        return None
    url = PROVIDER_LOGIN_URLS.get(provider)
    if not url:
        return None
    try:
        opened = bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception:
        opened = False
    return opened, url


def _provider_for_model(model: str, active_provider: str) -> str:
    providers = _providers_for_model(model)
    if active_provider in providers:
        return active_provider
    if providers:
        local_matches = [provider for provider in providers if provider in LOCAL_PROVIDERS]
        if local_matches:
            return local_matches[0]
        return providers[0]
    return active_provider


def _providers_for_model(model: str) -> list[str]:
    normalized = model.casefold()
    providers: list[str] = []
    for provider in sorted(
        AVAILABLE_PROVIDERS,
        key=lambda item: PROVIDER_SORT_ORDER.get(item, 99),
    ):
        hints = PROVIDER_MODEL_HINTS.get(provider, ())
        if any(hint.casefold() == normalized for hint in hints):
            providers.append(provider)
    return providers


def _discovered_model_options(ctx: Any,) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []

    for provider in sorted(
        AVAILABLE_PROVIDERS,
        key=lambda item: PROVIDER_SORT_ORDER.get(item, 99),
    ):
        discovered, discovery_error = _discover_provider_models(
            ctx,
            provider,
        )

        discovered_set = set(discovered)
        suggestions = PROVIDER_MODEL_HINTS.get(provider, ())

        for model in discovered:
            options.append({
                "provider": provider,
                "model": model,
                "state": ModelState.READY,
                "selectable": True,
                "error": None,
            })

        for model in suggestions:
            if model in discovered_set:
                continue

            if provider in LOCAL_PROVIDERS:
                state = (
                    ModelState.SERVER_OFFLINE
                    if discovery_error
                    else ModelState.MODEL_NOT_INSTALLED
                )
            elif (
                discovery_error
                and _provider_configuration_state_from_environment(provider)
                == "api_key_required"
            ):
                state = ModelState.AUTH_REQUIRED
            elif discovery_error:
                state = ModelState.UNAVAILABLE
            else:
                state = ModelState.UNKNOWN

            options.append({
                "provider": provider,
                "model": model,
                "state": state,
                "selectable": state in {
                    ModelState.READY,
                    ModelState.AUTH_REQUIRED,
                },
                "error": discovery_error,
            })

    return options

def _model_options() -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []

    for provider in sorted(
        AVAILABLE_PROVIDERS,
        key=lambda item: MODEL_PICKER_SORT_ORDER.get(
            item,
            99,
        ),
    ):
        for model in PROVIDER_MODEL_HINTS.get(
            provider,
            (),
        ):
            options.append({
                "provider": provider,
                "model": model,
                "state": ModelState.UNKNOWN,
                "selectable": True,
                "error": None,
            })

    return options


def _model_options_for_display(ctx: Any) -> list[dict[str, Any]]:
    active_provider = _active_provider(ctx)
    active_model = str(getattr(getattr(ctx, "llm", None), "model", "") or "")
    options = _model_options()
    local_availability = _discover_local_provider_availability(ctx)
    known_options = {
        (option["provider"], option["model"])
        for option in options
    }
    for provider, (discovered, _error) in local_availability.items():
        for installed_model in discovered:
            if (provider, installed_model) in known_options:
                continue
            options.append({
                "provider": provider,
                "model": installed_model,
                "state": ModelState.READY,
                "selectable": True,
                "error": None,
            })
            known_options.add((provider, installed_model))

    for option in options:
        provider = option["provider"]
        model = option["model"]
        if provider in LOCAL_PROVIDERS:
            discovered, discovery_error = local_availability.get(provider, ([], "Runtime unavailable."))
            if discovery_error:
                option["state"] = ModelState.SERVER_OFFLINE
            elif _local_model_is_available(model, discovered):
                option["state"] = ModelState.READY
            else:
                option["state"] = ModelState.MODEL_NOT_INSTALLED
        else:
            option["state"] = _hinted_model_state(
                ctx,
                provider,
                model,
                active_provider,
                active_model,
            )
        option["active"] = provider == active_provider and model == active_model
        option["selectable"] = True
        option["error"] = None

    return sorted(options, key=_model_display_sort_key)


def _model_display_sort_key(option: dict[str, Any]) -> tuple[Any, ...]:
    provider = str(option.get("provider") or "")
    model = str(option.get("model") or "")
    state = option.get("state")
    state_order = {
        ModelState.READY: 0,
        ModelState.AUTH_REQUIRED: 1,
        ModelState.LOGIN_REQUIRED: 1,
        ModelState.UNKNOWN: 2,
        ModelState.MODEL_NOT_INSTALLED: 3,
        ModelState.SERVER_OFFLINE: 4,
        ModelState.DOWNLOADING: 4,
        ModelState.LOADING: 4,
        ModelState.UNAVAILABLE: 5,
        ModelState.INCOMPATIBLE: 5,
    }
    model_rank = _provider_model_rank(provider, model)
    return (
        MODEL_PICKER_SORT_ORDER.get(provider, 99),
        model_rank,
        _descending_text_key(model),
        state_order.get(state, 6),
    )


def _provider_model_rank(provider: str, model: str) -> int:
    normalized = model.casefold()
    for index, hinted_model in enumerate(PROVIDER_MODEL_HINTS.get(provider, ())):
        if hinted_model.casefold() == normalized:
            return index
    return len(PROVIDER_MODEL_HINTS.get(provider, ()))


def _descending_text_key(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value.casefold())


def _discover_local_provider_availability(
    ctx: Any,
) -> dict[str, tuple[list[str], str | None]]:
    results: dict[str, tuple[list[str], str | None]] = {}
    with ThreadPoolExecutor(max_workers=len(LOCAL_PROVIDERS)) as executor:
        pending = {
            executor.submit(_discover_provider_models, ctx, provider): provider
            for provider in LOCAL_PROVIDERS
        }
        for future in as_completed(pending):
            provider = pending[future]
            try:
                results[provider] = future.result()
            except Exception as exc:
                results[provider] = ([], str(exc))
    return results


def _local_model_is_available(model: str, discovered: list[str]) -> bool:
    normalized = model.casefold()
    return any(
        installed.casefold() == normalized
        or installed.casefold().startswith(f"{normalized}:")
        for installed in discovered
    )


def _hinted_model_state(
    ctx: Any,
    provider: str,
    model: str,
    active_provider: str,
    active_model: str,
) -> ModelState:
    if provider == active_provider and model == active_model and _llm_configuration(ctx) == "ready":
        return ModelState.READY

    if provider in LOCAL_PROVIDERS:
        return ModelState.UNKNOWN

    if provider in UNIMPLEMENTED_PROVIDER_TRANSPORTS:
        return ModelState.UNAVAILABLE

    if provider == "openai-compatible":
        return ModelState.UNKNOWN

    env_name = PROVIDER_API_KEY_ENVS.get(provider)
    if env_name and not os.environ.get(env_name):
        return ModelState.AUTH_REQUIRED

    return ModelState.UNKNOWN


def _provider_access_label(provider: str) -> str:
    if provider in LOCAL_PROVIDERS:
        return "open source · local runtime/install · no login"
    if provider == "openai-compatible":
        return "endpoint required"
    if provider == "copilot":
        return "GitHub sign-in"
    if provider in {"bedrock", "vertexai"}:
        return "cloud credentials"
    if provider == "azure":
        return "Azure endpoint and API key"
    return "sign in or API key"


def _model_source_label(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider)


def _voice_snapshot() -> dict[str, Any]:
    try:
        from .voice import voice_status

        status = voice_status()
        return status.as_dict()
    except Exception:
        return {
            "input_ready": False,
            "input_reason": "Voice module unavailable.",
            "tts_ready": False,
            "tts_reason": None,
            "auto_speak": False,
            "stt_provider": None,
            "tts_provider": None,
            "input_secret_provider": None,
        }


def _agent_display_name(ctx: Any) -> str:
    return str(getattr(ctx, "agent_name", DEFAULT_AGENT_NAME) or DEFAULT_AGENT_NAME)


def _unquote_agent_name_argument(value: str) -> str:
    name = value.strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in {"'", '"'}:
        return name[1:-1]
    return name


def _set_agent_name_command(ctx: Any, raw_name: str) -> str:
    try:
        agent_name = _persist_agent_name(raw_name)
    except ValueError as exc:
        return str(exc)
    ctx.agent_name = agent_name
    return f"Okay, my name is now {agent_name}."


def _status_text(ctx: AppContext) -> str:
    lines = ["Status", "", "Session:", str(ctx.session_id), ""]
    lines.extend(_status_context_lines(ctx))
    return "\n".join(lines)


def _local_runtime_status_lines(ctx: Any) -> list[str]:
    provider = _active_provider(ctx)
    if provider not in LOCAL_PROVIDERS:
        return []

    source = _model_source_label(provider)
    model = str(getattr(getattr(ctx, "llm", None), "model", "") or "")
    base_url = _provider_base_url(ctx, provider)
    lines = [
        "Local runtime:",
        f"- Source: {source}",
        f"- Endpoint: {base_url}",
    ]

    discovered, discovery_error = _discover_provider_models(ctx, provider)
    if discovery_error:
        lines.extend([
            "- Server: unreachable",
            f"- Details: {discovery_error}",
            "- Latency: server is offline; the next usable turn must wait for the runtime/model to start.",
        ])
        return lines

    installed = ", ".join(discovered[:8]) if discovered else "none"
    if len(discovered) > 8:
        installed = f"{installed}, +{len(discovered) - 8} more"
    active_installed = _local_model_is_available(model, discovered)
    lines.extend([
        "- Server: reachable",
        f"- Installed/API models: {installed}",
        f"- Active model installed: {'yes' if active_installed else 'no'}",
    ])

    if provider != "ollama":
        lines.append("- Loaded/warm models: not exposed by this provider's OpenAI-compatible API")
        return lines

    try:
        loaded = _discover_ollama_loaded_models(base_url)
    except RuntimeError as exc:
        lines.append(f"- Loaded/warm models: unavailable ({exc})")
        return lines

    loaded_names = [str(item["model"]) for item in loaded]
    active_loaded = _local_model_is_available(model, loaded_names)
    if not loaded:
        lines.extend([
            "- Loaded/warm models: none",
            "- Latency: first prompt may be slow because Ollama must load the model into memory.",
        ])
        return lines

    loaded_text = ", ".join(_format_loaded_local_model(item) for item in loaded[:5])
    if len(loaded) > 5:
        loaded_text = f"{loaded_text}, +{len(loaded) - 5} more"
    lines.extend([
        f"- Loaded/warm models: {loaded_text}",
        f"- Active model loaded: {'yes' if active_loaded else 'no'}",
    ])
    if not active_loaded:
        lines.append("- Latency: active model is installed but not warm; the next prompt may include load time.")
    return lines


def _format_loaded_local_model(item: dict[str, Any]) -> str:
    model = str(item.get("model") or "unknown")
    metadata = [
        str(value)
        for value in (item.get("parameters"), item.get("quantization"))
        if isinstance(value, str) and value
    ]
    suffix = f" ({', '.join(metadata)})" if metadata else ""
    return f"{model}{suffix}"



def _gateway_text(ctx: AppContext) -> str:
    gateway = getattr(ctx, "gateway", None)
    if gateway is None:
        return "Agent control plane is not available in this session."
    status = gateway.status()
    sources = status.get("config_sources") or []
    tool_allowlist = getattr(ctx, "tool_allowlist", None)
    tools_text = (
        "all parent tools"
        if tool_allowlist is None
        else ", ".join(tool_allowlist) or "none"
    )
    return "\n".join([
        "Agent control plane",
        f"Runtime: {status['control_plane']}",
        f"Session store: {status['session_store']}",
        f"Agent profile: {getattr(ctx, 'agent_id', status['default_agent'])}",
        f"Profile tools: {tools_text}",
        f"Session scope: {status['default_scope']}",
        f"Route: {getattr(ctx, 'route_key', None) or 'direct CLI session'}",
        f"Bindings: {status['bindings']}",
        f"Registered channels: {', '.join(status['channels']) or 'none'}",
        f"Config: {', '.join(sources) or 'built-in defaults'}",
        f"Execution: {status['execution_model']}",
    ])


def _gateway_control_snapshot(ctx: AppContext) -> dict[str, Any]:
    gateway = getattr(ctx, "gateway", None)
    if gateway is None:
        raise RuntimeError("Agent control plane is not available in this session.")
    snapshot = gateway.control_snapshot()
    overview = snapshot.get("overview")
    if isinstance(overview, dict):
        overview.update({
            "active_session": ctx.session_id,
            "active_agent": getattr(ctx, "agent_id", "main"),
            "active_route": getattr(ctx, "route_key", None),
            "workspace_root": str(ctx.workspace_root),
            "tool_policy": (
                "all parent tools"
                if getattr(ctx, "tool_allowlist", None) is None
                else ", ".join(ctx.tool_allowlist) or "none"
            ),
        })
    return snapshot


def _status_context_lines(ctx: Any) -> list[str]:
    model = str(getattr(getattr(ctx, "llm", None), "model", "") or "")
    context_limit = _context_window_for_model(model)
    session_info = _ctx_session_info(ctx)
    if session_info is None:
        total_tokens = 0
    else:
        total_tokens = _billable_token_total(session_info.tokens)

    lines = ["Context:"]
    if context_limit is None:
        lines.append("  Limit is not reported for this model")
    else:
        context_left = max(0, context_limit - total_tokens)
        percent_left = 100.0 - (_usage_percent(total_tokens, context_limit) or 0.0)
        lines.append(
            f"  {_format_percent(percent_left)} left "
            f"({_format_count(total_tokens)} used / {_format_count(context_limit)})"
        )
        lines.append(f"  {_context_meter(percent_left)}")
    return lines


def _context_meter(percent_left: float, width: int = 40) -> str:
    width = max(1, width)
    filled = min(width, max(0, round((percent_left / 100.0) * width)))
    return "█" * filled + "░" * (width - filled)


def _ctx_session_info(ctx: Any) -> SessionInfo | None:
    store = getattr(ctx, "store", None)
    session_id = getattr(ctx, "session_id", None)
    if store is None or not session_id or not hasattr(store, "get_session"):
        return None
    try:
        return store.get_session(session_id)
    except Exception:
        return None


def _setup_text(ctx: AppContext, selection: str | None) -> str:
    if selection is None:
        return "\n".join([
            "Connect a model",
            "",
            "Choose one in the menu above:",
            "• Ollama — run a private local model, no account needed",
            "• OpenAI, Anthropic, or Groq — connect with an API key",
            "",
            "You can change models any time with /model.",
        ])

    normalized = selection.casefold()
    if normalized in {"status", "check"}:
        provider = _active_provider(ctx)
        state = _llm_configuration_state(ctx)
        next_step = "You are ready to chat." if state == "ready" else "Choose a provider from /setup to continue."
        return "\n".join([
            "Connection status",
            f"Model: {_model_source_label(provider)} · {ctx.llm.model}",
            f"State: {'Ready' if state == 'ready' else 'Needs setup'}",
            next_step,
        ])
    if normalized == "local":
        normalized = "ollama"
    if normalized in LOCAL_PROVIDERS:
        runtime = _model_source_label(normalized)
        return "\n".join([
            f"Run a local model with {runtime}",
            "",
            f"Install and start {runtime}, then choose a model from /model.",
            "No account or API key is needed.",
            "",
            "Need another local runtime? Use /setup more.",
        ])
    if normalized in {"more", "other"}:
        providers = _setup_available_providers(include_local=True)
        names = ", ".join(_model_source_label(provider) for provider in providers)
        return f"More connection options\n\n{names}\n\nType /setup followed by the provider name."
    try:
        provider = _normalize_provider(normalized)
    except ValueError:
        return "That connection option is not available. Type /setup and choose one from the menu."
    if provider in LOCAL_PROVIDERS:
        return _setup_text(ctx, "ollama")
    if provider not in PROVIDER_API_KEY_ENVS or provider in UNIMPLEMENTED_PROVIDER_TRANSPORTS:
        return "That provider is not available in this build. Choose another option from /setup."

    model = PROVIDER_MODEL_HINTS.get(provider, (ctx.llm.model,))[0]
    previous_llm = ctx.llm
    try:
        ctx.llm = LLMClient(model=model, provider=provider)
        _apply_saved_reasoning_effort(ctx, ctx.llm)
        _persist_llm_config(ctx)
    except Exception as exc:
        ctx.llm = previous_llm
        return _command_text(
            f"Could not prepare {_model_source_label(provider)}: {exc}",
            code="unavailable",
            setup_required=True,
            error=True,
        )
    return _command_text(
        "\n".join([
            f"Connect {_model_source_label(provider)}",
            "Paste your API key in the protected field below.",
            "It is never shown in the conversation.",
            f"Need a key first? Use /login {provider}.",
        ]),
        code="api_key_required",
        setup_required=True,
        secret_provider=provider,
    )


def _setup_available_providers(*, include_local: bool = False) -> list[str]:
    providers = [
        provider
        for provider in AVAILABLE_PROVIDERS
        if provider in PROVIDER_API_KEY_ENVS
        and provider not in UNIMPLEMENTED_PROVIDER_TRANSPORTS
    ]
    if include_local:
        providers.extend(sorted(LOCAL_PROVIDERS))
    return sorted(set(providers), key=lambda provider: PROVIDER_SORT_ORDER.get(provider, 99))


def _llm_mode(ctx: AppContext) -> str:
    value = getattr(getattr(ctx, "llm", None), "mode", "")
    return str(value or "unknown")


def _llm_endpoint(ctx: AppContext) -> str:
    value = getattr(getattr(ctx, "llm", None), "endpoint", "")
    return str(value or "not configured")


def _llm_configuration(ctx: AppContext) -> str:
    error = getattr(getattr(ctx, "llm", None), "configuration_error", None)
    return str(error) if error else "ready"


def _llm_configuration_state(ctx: AppContext) -> str:
    llm = getattr(ctx, "llm", None)
    explicit = getattr(llm, "configuration_state", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    if _llm_configuration(ctx) == "ready":
        return "ready"
    return _provider_configuration_state_from_environment(_active_provider(ctx))


def _candidate_configuration_state(candidate: Any) -> str:
    explicit = getattr(candidate, "configuration_state", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    if not getattr(candidate, "configuration_error", None):
        return "ready"
    provider = str(getattr(candidate, "provider", "") or "")
    return _provider_configuration_state_from_environment(provider)


def _provider_configuration_state_from_environment(provider: str) -> str:
    if provider in LOCAL_PROVIDERS:
        return "runtime_unavailable"
    if provider == "openai-compatible" and not os.environ.get(
        "AGENT_OPENAI_COMPAT_BASE_URL"
    ):
        return "endpoint_required"
    env_name = PROVIDER_API_KEY_ENVS.get(provider)
    if env_name and not os.environ.get(env_name):
        return "api_key_required"
    if provider == "bedrock":
        has_credentials = bool(
            os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("AWS_PROFILE")
            or (
                os.environ.get("AWS_ACCESS_KEY_ID")
                and os.environ.get("AWS_SECRET_ACCESS_KEY")
            )
        )
        return "unavailable" if has_credentials else "credentials_required"
    if provider == "vertexai":
        has_project = bool(
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT_ID")
        )
        return "unavailable" if has_project else "credentials_required"
    if provider == "copilot":
        return "credentials_required"
    return "unavailable"


def _persist_llm_config(ctx: Any) -> None:
    store = getattr(ctx, "store", None)
    session_id = getattr(ctx, "session_id", None)
    if store is None or not session_id or not hasattr(store, "update_llm_config"):
        return
    store.update_llm_config(
        session_id,
        provider=getattr(getattr(ctx, "llm", None), "provider", None),
        model=getattr(getattr(ctx, "llm", None), "model", None),
    )


def _set_provider_api_key(ctx: Any, provider: str, api_key: str) -> str:
    try:
        normalized = _normalize_provider(provider)
    except ValueError as exc:
        return str(exc)

    env_name = PROVIDER_API_KEY_ENVS.get(normalized)
    if env_name is None:
        return f"{_model_source_label(normalized)} does not use an API key."

    os.environ[env_name] = api_key
    try:
        _persist_api_key(env_name, api_key)
    except OSError as exc:
        return f"{_model_source_label(normalized)} key could not be saved: {exc}"
    active_provider = _active_provider(ctx)
    pending_provider = getattr(ctx, "pending_provider", None)
    pending_model = getattr(ctx, "pending_model", None)
    reload_provider = pending_provider if pending_provider == normalized else active_provider

    if reload_provider == normalized:
        current_model = getattr(getattr(ctx, "llm", None), "model", None)
        reload_model = pending_model if pending_provider == normalized else current_model
        try:
            ctx.llm = LLMClient(model=reload_model, provider=reload_provider)
            _apply_saved_reasoning_effort(ctx, ctx.llm)
            ctx.pending_provider = None
            ctx.pending_model = None
            _persist_llm_config(ctx)
        except Exception as exc:
            return f"{_model_source_label(normalized)} key loaded, but reload failed: {exc}"

    return (
        f"{_model_source_label(normalized)} key loaded.\n"
        f"Configuration: {_llm_configuration(ctx)}"
    )


def _provider_api_key_needed(
    ctx: Any,
) -> str | None:
    pending_provider = getattr(
        ctx,
        "pending_provider",
        None,
    )

    if pending_provider:
        if pending_provider in LOCAL_PROVIDERS:
            return None
        if (
            pending_provider in PROVIDER_API_KEY_ENVS
            and _llm_configuration_state(ctx) == "api_key_required"
        ):
            return pending_provider

        return None

    provider = _active_provider(ctx)

    if provider in LOCAL_PROVIDERS:
        return None

    if (
        provider in PROVIDER_API_KEY_ENVS
        and _llm_configuration_state(ctx) == "api_key_required"
    ):
        return provider

    return None


def _login_provider(ctx: Any, provider: str) -> str:
    try:
        normalized = _normalize_provider(provider)
    except ValueError as exc:
        return str(exc)

    url = PROVIDER_LOGIN_URLS.get(normalized)
    display_name = PROVIDER_DISPLAY_NAMES.get(normalized, normalized)
    if url is None:
        if normalized in LOCAL_PROVIDERS:
            return f"{display_name} is local and needs no login. Start its server, then use /model {normalized} <model>."
        return f"No login URL is configured for {normalized}."

    opened = False
    try:
        opened = bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception:
        opened = False

    status = "Opened" if opened else "Open"
    active_hint = ""
    if normalized == _active_provider(ctx):
        active_hint = f"\nAfter creating a key, return here and run: /apikey {normalized}"
    return f"{status} {display_name} account/API-key page:\n{url}{active_hint}"


def _api_key_prompt_provider(text: str) -> str | None:
    parts = text.strip().split()
    if len(parts) != 2 or parts[0].casefold() not in {"/apikey", "/key"}:
        return None
    try:
        provider = _normalize_provider(parts[1])
        return None if provider in LOCAL_PROVIDERS else provider
    except ValueError:
        return None




def _starts_auth_candidate(text: str) -> bool:
    normalized = text.strip().casefold()
    return normalized.startswith("/provider ") or normalized.startswith("/model ")


def _redact_local_command(text: str) -> str:
    parts = text.strip().split()
    if len(parts) >= 3 and parts[0].casefold() in {"/apikey", "/key"}:
        return f"{parts[0]} {parts[1]} <redacted>"
    return text


def _is_exit_command(value: str) -> bool:
    return value.strip().casefold() in {"/exit", "/quit", "/q"}


def _active_provider(ctx: Any) -> str:
    provider = getattr(getattr(ctx, "llm", None), "provider", None)
    return str(provider or "openai")


def print_session_history(store: SessionStore, session_id: str) -> None:
    messages = store.list_messages(session_id, limit=None)
    if not messages:
        return

    print("Loaded session history")
    print()
    for message in messages:
        print(f"{message.role}:")
        print(message.content)
        print()


def list_sessions(store: SessionStore) -> int:
    sessions = store.list_sessions(limit=SESSION_LIST_LIMIT)
    if not sessions:
        print("No sessions found.")
        return 0

    print(f"{'ID':12}  {'UPDATED':>10}  TITLE")
    for item in sessions:
        title = item.last_prompt or item.title
        print(f"{item.id:12}  {format_age(item.updated_at):>10}  {truncate(title, 96)}")

    return 0


def format_age(timestamp: str) -> str:
    try:
        updated = datetime.fromisoformat(timestamp)
    except ValueError:
        return "unknown"

    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)

    seconds = max(0, int((datetime.now(timezone.utc) - updated).total_seconds()))

    if seconds < 60:
        return "now"

    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"

    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"

    days = hours // 24
    if days < 30:
        return f"{days}d ago"

    return updated.date().isoformat()


def truncate(value: str | None, limit: int) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _load_persisted_api_keys()
    try:
        store = SessionStore.default()
    except Exception as exc:
        if args.tui_bridge:
            payload = {"ok": False, "error": str(exc)}
            if args.tui_bridge == "stream-submit":
                payload["kind"] = "final"
                _bridge_emit(payload)
                return 0
            print(json.dumps(payload, ensure_ascii=False))
            return 1
        raise

    if args.tui_bridge:
        return _run_tui_bridge(args, store)

    command = args.session_command.strip() if args.session_command else None
    session_arg = args.session_id.strip() if args.session_id else None

    if command and command.lower() == "list":
        if session_arg:
            parser.error("'list' does not accept a session id.")
        return list_sessions(store)

    use_tui = args.tui or (command and command.lower() == "tui")

    if command and command.lower() == "tui":
        command = None

    if command and command.lower() == "resume":
        if session_arg:
            session_info = load_existing_session(session_arg, store)
        else:
            session_info = choose_session(store)
            if session_info is None:
                return 0
    else:
        if session_arg:
            parser.error(f"unrecognized arguments: {session_arg}")

        session_info = (
            load_existing_session(command, store)
            if command
            else create_new_session(args, store)
        )

    ctx = build_context(args, store=store, session_info=session_info)

    if use_tui:
        return run_tui(ctx)

    print_session_history(store, session_info.id)

    if ctx.debug:
        print(f"context={ctx}")

    return repl(ctx)


def run_tui(ctx: AppContext) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("TUI requires an interactive terminal.")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    command = [
        str(ctx.rust.rust_bin),
        "tui",
        "--python",
        sys.executable,
        "--repo-root",
        str(repo_root),
        "--session-id",
        ctx.session_id,
    ]
    for key in _load_tui_paste_keys():
        command.extend(["--paste-key", key])
    for key in _load_tui_copy_keys():
        command.extend(["--copy-key", key])
    if _load_tui_mouse_capture():
        command.append("--mouse-capture")
    env = os.environ.copy()
    store_db_path = getattr(getattr(ctx, "store", None), "db_path", None)
    if store_db_path is not None:
        env["AGENT_SESSION_DB"] = str(store_db_path)
    config_path = getattr(ctx, "config_path", None)
    if config_path is not None:
        env["AGENT_CONFIG"] = str(config_path)

    try:
        completed = subprocess.run(command, check=False, env=env)
    except OSError as exc:
        print(f"Failed to start Ratatui UI: {exc}")
        return 1
    finally:
        _stop_language_servers(ctx)

    return int(completed.returncode)


@contextmanager
def _exclusive_bridge_turn(session_id: str):
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
    lock_dir = Path(tempfile.gettempdir()) / "agent-bridge-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{digest}.lock"
    handle = lock_path.open("a+")
    if fcntl is None:
        handle.close()
        raise RuntimeError(
            "This runtime cannot enforce the single active bridge safety lock."
        )
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "A turn is already active for this session. Wait for it to finish before submitting another."
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _run_tui_stream_submit(ctx: AppContext, prompt: str) -> int:
    if not prompt:
        _bridge_emit({
            "kind": "final",
            "ok": False,
            "error": "Prompt cannot be empty.",
            "snapshot": _tui_bridge_snapshot(ctx),
        })
        return 0

    try:
        with _exclusive_bridge_turn(ctx.session_id):
            _bridge_emit({
                "kind": "submitted",
                "prompt": prompt,
                "snapshot": _tui_bridge_snapshot(ctx),
            })
            answer = _handle_local_command(
                ctx,
                prompt,
                install_progress=lambda summary: _tui_bridge_local_progress(
                    ctx,
                    "install_progress",
                    summary,
                ),
            )
            if answer is None:
                answer = handle_prompt(
                    ctx,
                    prompt,
                    stream_event=lambda event: _tui_bridge_stream_event(ctx, event),
                    approval_requester=lambda request: _tui_bridge_wait_for_approval(ctx, request),
                )
            else:
                _record_local_command_exchange(ctx, prompt, answer)
            if os.environ.get("AGENT_TTS_ENABLED") == "1" and answer:
                try:
                    from .voice import speak
                    import threading as _threading
                    _threading.Thread(target=speak, args=(answer,), daemon=True).start()
                except Exception:
                    pass
            _bridge_emit({
                "kind": "final",
                "ok": True,
                "answer": answer,
                "command_result": ctx.last_local_command_result,
                "snapshot": _tui_bridge_snapshot(ctx),
            })
            return 0
    except Exception as exc:
        _bridge_emit({
            "kind": "final",
            "ok": False,
            "error": str(exc),
            "snapshot": _tui_bridge_snapshot(ctx),
        })
        return 0


def _expire_orphaned_approvals(ctx: AppContext) -> int:
    try:
        with _exclusive_bridge_turn(ctx.session_id):
            return _expire_orphaned_approvals_unlocked(ctx)
    except RuntimeError as exc:
        if "A turn is already active for this session" in str(exc):
            return 0
        raise


def _expire_orphaned_approvals_unlocked(ctx: AppContext) -> int:
    expired_count = 0
    expired_ids: list[str] = []
    expired_at = datetime.now(timezone.utc).isoformat()
    for item in ctx.session.pending_approvals:
        if not isinstance(item, dict) or item.get("status") != "pending":
            continue
        item["status"] = "expired"
        item["decision"] = "denied"
        item["decision_at"] = expired_at
        item["expired_reason"] = "no_active_turn"
        expired_count += 1
        request_id = item.get("id")
        if isinstance(request_id, str) and request_id:
            expired_ids.append(request_id)
    if expired_count == 0:
        return 0
    persist_agent_state(ctx)
    ctx.store.add_event(
        ctx.session_id,
        event_type="approval_expired",
        summary=f"Expired {expired_count} orphaned approval request(s)",
        data={"request_ids": expired_ids, "reason": "no_active_turn"},
    )
    return expired_count


def _run_tui_bridge(args: argparse.Namespace, store: SessionStore) -> int:
    session_id = (args.bridge_session_id or "").strip()
    if not session_id:
        print(json.dumps({"ok": False, "error": "Missing bridge session id."}))
        return 1

    if args.tui_bridge == "complete":
        prompt = args.bridge_prompt or ""
        normalized_prompt = _normalized_command_prompt(prompt)
        prompt_parts = normalized_prompt.strip().split()
        completion_ctx = (
            _completion_context_from_session(store, session_id)
            if _is_model_palette_prompt(normalized_prompt, prompt_parts)
            else None
        )
        payload = {
            "ok": True,
            "completions": _tui_bridge_completions(prompt, ctx=completion_ctx),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    ctx: AppContext | None = None
    try:
        session_info = store.get_session(session_id)
        ctx = build_context(args, store=store, session_info=session_info)
        if args.tui_bridge == "snapshot":
            _expire_orphaned_approvals(ctx)
            payload = {"ok": True, "snapshot": _tui_bridge_snapshot(ctx)}
        elif args.tui_bridge == "gateway":
            payload = {"ok": True, "gateway": _gateway_control_snapshot(ctx)}
        elif args.tui_bridge in {"approve", "deny"}:
            request_id = (args.bridge_request_id or "").strip()
            if not request_id:
                payload = {"ok": False, "error": "Missing approval request id.", "snapshot": _tui_bridge_snapshot(ctx)}
            else:
                decision = "approved" if args.tui_bridge == "approve" else "denied"
                payload = _tui_bridge_apply_approval_decision(ctx, request_id, decision)
        elif args.tui_bridge == "voice-record":
            from .voice import bridge_voice_record
            payload = bridge_voice_record()
        elif args.tui_bridge == "voice-stream":
            from .voice import bridge_voice_stream
            return bridge_voice_stream()
        elif args.tui_bridge == "voice-speak":
            from .voice import bridge_voice_speak
            text = (args.bridge_prompt or "").strip()
            payload = bridge_voice_speak(text)
        elif args.tui_bridge == "stream-submit":
            prompt = (args.bridge_prompt or "").strip()
            return _run_tui_stream_submit(ctx, prompt)
        else:
            prompt = (args.bridge_prompt or "").strip()
            if not prompt:
                payload = {"ok": False, "error": "Prompt cannot be empty.", "snapshot": _tui_bridge_snapshot(ctx)}
            else:
                try:
                    answer = _handle_local_command(ctx, prompt)
                    if answer is None:
                        answer = handle_prompt(ctx, prompt)
                    else:
                        _record_local_command_exchange(ctx, prompt, answer)
                    payload = {
                        "ok": True,
                        "answer": answer,
                        "command_result": ctx.last_local_command_result,
                        "snapshot": _tui_bridge_snapshot(ctx),
                    }
                except Exception as exc:
                    payload = {"ok": False, "error": str(exc), "snapshot": _tui_bridge_snapshot(ctx)}
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    finally:
        try:
            if ctx is not None:
                _stop_language_servers(ctx)
        except Exception:
            pass


def _tui_bridge_snapshot(ctx: AppContext) -> dict[str, Any]:
    session = ctx.store.get_session(ctx.session_id)
    messages = ctx.store.list_messages(ctx.session_id, limit=TUI_TRANSCRIPT_LIMIT)
    return {
        "session": {
            "id": session.id,
            "title": session.title,
            "workspace_root": session.workspace_root,
            "updated_at": session.updated_at,
            "provider": _active_provider(ctx),
            "model": ctx.llm.model,
            "mode": _llm_mode(ctx),
            "configuration": _llm_configuration(ctx),
            "configuration_state": _llm_configuration_state(ctx),
            "context_limit": _context_window_for_model(ctx.llm.model),
            "cost_usd": session.cost_usd,
            "pending_attachments": [
                {
                    "filename": attachment.filename,
                    "mime": attachment.mime,
                    "size_bytes": attachment.size_bytes,
                    "storage_path": attachment.storage_path,
                }
                for attachment in getattr(ctx, "pending_attachments", ())
            ],
            "tokens": {
                "input": session.tokens.input,
                "output": session.tokens.output,
                # Keep the TUI bridge token schema compatible with both current
                # and previously built Rust front ends.  Older binaries require
                # these counters even when no reasoning or cache tokens exist.
                "reasoning": session.tokens.reasoning,
                "cache_read": session.tokens.cache_read,
                "cache_write": session.tokens.cache_write,
            },
        },
        "agent_name": _agent_display_name(ctx),
        "voice": _voice_snapshot(),
        "approvals": [
            dict(item)
            for item in ctx.session.pending_approvals
            if isinstance(item, dict) and item.get("status") == "pending"
        ],
        "messages": [
            {
                "role": item.role,
                "content": item.content,
                "created_at": item.created_at,
                "attachments": [
                    {
                        "filename": attachment.filename,
                        "mime": attachment.mime,
                        "size_bytes": attachment.size_bytes,
                        "storage_path": attachment.storage_path,
                    }
                    for attachment in getattr(item, "attachments", ())
                ],
            }
            for item in messages
        ],
    }


def _tui_bridge_completions(prompt: str, *, ctx: Any | None = None) -> dict[str, Any]:
    entries = _slash_palette_entries(prompt, ctx=ctx)
    return {
        "title": _slash_palette_title(prompt) if entries else "",
        "selected_index": _palette_selected_index(prompt, entries, ctx=ctx),
        "entries": [
            {
                "value": entry.value,
                "label": entry.label,
                "description": entry.description,
                "complete_to": entry.complete_to,
                "execute": entry.execute,
            }
            for entry in entries
        ],
    }


def _palette_selected_index(
    prompt: str,
    entries: list[PaletteEntry],
    *,
    ctx: Any | None = None,
) -> int:
    if not entries:
        return 0
    if _normalized_command_prompt(prompt).startswith("/model"):
        if ctx is not None:
            provider = _active_provider(ctx)
            model = str(getattr(getattr(ctx, "llm", None), "model", "") or "")
            active_value = f"{provider}/{model}"
            for index, entry in enumerate(entries):
                if entry.value == active_value and _palette_entry_selectable(entry):
                    return index
        for index, entry in enumerate(entries):
            if _palette_entry_selectable(entry):
                return index
    return 0


def _completion_context_from_session(store: Any, session_id: str) -> Any | None:
    try:
        session_info = store.get_session(session_id)
    except Exception:
        return None
    provider = getattr(session_info, "provider", None)
    model = getattr(session_info, "model", None)
    if not provider and not model:
        return None
    return argparse.Namespace(
        llm=argparse.Namespace(
            provider=provider or "openai",
            model=model or "",
            configuration_error=None,
            mode="unknown",
            endpoint="",
        )
    )


def _bridge_emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _tui_bridge_stream_event(ctx: AppContext, event: dict[str, Any]) -> None:
    persist_agent_state(ctx)
    _bridge_emit({"kind": "stream_event", "event": event, "snapshot": _tui_bridge_snapshot(ctx)})


def _tui_bridge_local_progress(
    ctx: AppContext,
    kind: str,
    summary: str,
) -> None:
    _bridge_emit({
        "kind": "stream_event",
        "event": {"kind": kind, "summary": summary},
        "snapshot": _tui_bridge_snapshot(ctx),
    })


def _tui_bridge_wait_for_approval(ctx: AppContext, request: dict[str, Any], *, poll_interval: float = 0.1, timeout: float = 1800.0) -> str:
    request_id = str(request.get("id") or "")
    if not request_id:
        return "denied"

    persist_agent_state(ctx)
    deadline = time.time() + timeout
    while time.time() < deadline:
        session_info = ctx.store.get_session(ctx.session_id)
        refreshed = agent_session_from_dict(session_info.state)
        ctx.session.pending_approvals = refreshed.pending_approvals
        for item in ctx.session.pending_approvals:
            if not isinstance(item, dict) or item.get("id") != request_id:
                continue
            status = str(item.get("status") or "pending")
            decision = str(item.get("decision") or "")
            if status != "pending" and decision in {"approved", "denied"}:
                persist_agent_state(ctx)
                return decision
        time.sleep(poll_interval)
    return "denied"


def _tui_bridge_apply_approval_decision(ctx: AppContext, request_id: str, decision: str) -> dict[str, Any]:
    matched = False
    for item in ctx.session.pending_approvals:
        if not isinstance(item, dict) or item.get("id") != request_id:
            continue
        item["status"] = decision
        item["decision"] = decision
        item["decision_at"] = datetime.now(timezone.utc).isoformat()
        matched = True
        break

    if not matched:
        session_info = ctx.store.get_session(ctx.session_id)
        refreshed = agent_session_from_dict(session_info.state)
        ctx.session.pending_approvals = refreshed.pending_approvals
        for item in ctx.session.pending_approvals:
            if not isinstance(item, dict) or item.get("id") != request_id:
                continue
            item["status"] = decision
            item["decision"] = decision
            item["decision_at"] = datetime.now(timezone.utc).isoformat()
            matched = True
            break

    if not matched:
        return {"ok": False, "error": f"Approval request not found: {request_id}", "snapshot": _tui_bridge_snapshot(ctx)}

    persist_agent_state(ctx)
    return {"ok": True, "snapshot": _tui_bridge_snapshot(ctx)}


def _stop_language_servers(ctx: Any) -> None:
    manager = getattr(ctx, "language_servers", None)
    stop_all = getattr(manager, "stop_all", None)
    if callable(stop_all):
        stop_all()


def _run_tui(stdscr: Any, ctx: AppContext) -> None:
    curses.curs_set(1)
    stdscr.keypad(True)
    stdscr.nodelay(True)
    _setup_tui_colors(stdscr)

    prompt = ""
    prompt_history: list[str] = []
    history_index: int | None = None
    transcript_scroll = 0
    transcript_at_bottom = True
    event_scroll = 0
    status = "Ready"
    live_turn = LiveTurnState()
    approval_queue = ApprovalQueueState(ctx)
    worker: threading.Thread | None = None
    worker_error: str | None = None
    active_prompt: str | None = None
    queued_prompts: deque[str] = deque()
    palette_index = 0
    secret_provider: str | None = None
    secret_value = ""

    def launch_turn(user_prompt: str, *, queue_if_busy: bool = True) -> None:
        nonlocal worker, worker_error, active_prompt, status

        if worker is not None and worker.is_alive():
            if queue_if_busy:
                queued_prompts.append(user_prompt)
                status = f"Queued {len(queued_prompts)} prompt(s)"
            else:
                status = "Agent is still working"
            return

        active_prompt = user_prompt
        worker_error = None
        status = "Thinking..."
        live_turn.start(user_prompt)

        def _target() -> None:
            nonlocal worker_error, active_prompt, status
            try:
                handle_prompt(
                    ctx,
                    user_prompt,
                    stream_event=live_turn.update,
                    approval_requester=approval_queue.request,
                )
                status = "Ready"
                live_turn.finish()
            except Exception as exc:  # pragma: no cover - surfaced in UI
                worker_error = str(exc)
                status = "Error"
                live_turn.finish(str(exc))
            finally:
                active_prompt = None

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()

    while True:
        height, width = stdscr.getmaxyx()
        messages = ctx.store.list_messages(ctx.session_id, limit=TUI_TRANSCRIPT_LIMIT)
        session_info = ctx.store.get_session(ctx.session_id)
        show_usage_panel = width >= USAGE_PANEL_MIN_TERMINAL_WIDTH
        panel_width = USAGE_PANEL_WIDTH if show_usage_panel else 0
        panel_gap = 1 if show_usage_panel else 0
        content_width = max(20, width - panel_width - panel_gap)
        approval_snapshot = approval_queue.snapshot()

        if worker is not None and not worker.is_alive() and active_prompt is None and status == "Thinking...":
            status = "Ready"

        if worker is not None and not worker.is_alive() and active_prompt is None and queued_prompts:
            launch_turn(queued_prompts.popleft(), queue_if_busy=False)
            continue

        rendered_messages = _render_tui_transcript(
            messages,
            live_turn.snapshot(),
            max(20, content_width - 2),
            agent_name=_agent_display_name(ctx),
        )
        palette_entries = [] if secret_provider else _slash_palette_entries(prompt)
        if palette_entries:
            palette_index = min(palette_index, len(palette_entries) - 1)
        else:
            palette_index = 0
        command_palette = _slash_command_lines(prompt, max(20, content_width - 2), selected_index=palette_index)
        palette_height = min(9, len(command_palette))

        header_height = 3
        status_y = max(header_height + 1, height - 3)
        input_y = max(header_height + 2, height - 2)
        palette_y = max(header_height + 1, status_y - palette_height)
        transcript_height = max(5, (palette_y if palette_height else status_y) - header_height)

        max_transcript_scroll = max(0, len(rendered_messages) - transcript_height)
        if transcript_at_bottom:
            transcript_scroll = 0
        else:
            transcript_scroll = min(transcript_scroll, max_transcript_scroll)
            transcript_scroll = max(0, transcript_scroll)

        stdscr.erase()
        _draw_header(stdscr, ctx, width)
        _draw_box_title(stdscr, 2, "Conversation", content_width)
        _draw_lines(
            stdscr,
            header_height,
            transcript_height,
            rendered_messages,
            transcript_scroll,
            content_width,
        )
        if palette_height:
            _draw_command_palette(stdscr, palette_y, command_palette[:palette_height], content_width)
        if show_usage_panel:
            _draw_usage_panel(
                stdscr,
                x=content_width,
                y=2,
                height=max(3, input_y - 2),
                width=panel_width,
                session=session_info,
                model=ctx.llm.model,
                provider=_active_provider(ctx),
                approvals=approval_snapshot,
            )
        _draw_status_line(
            stdscr,
            status_y,
            width,
            status=_queue_status(status, len(queued_prompts), len(approval_snapshot.get("pending", []))),
            live_turn=live_turn.snapshot(),
            worker_alive=worker is not None and worker.is_alive(),
            error=worker_error,
            usage_summary=None if show_usage_panel else _compact_usage_text(session_info, ctx.llm.model, _active_provider(ctx)),
            auth_active=secret_provider is not None,
            approval_active=bool(approval_snapshot.get("pending")),
        )
        _draw_input_line(
            stdscr,
            input_y,
            width,
            "*" * len(secret_value) if secret_provider else prompt,
            label=f" {secret_provider} key> " if secret_provider else " agent> ",
        )
        stdscr.refresh()

        key = stdscr.getch()
        if key == -1:
            time.sleep(0.05)
            continue

        if key == 3:  # Ctrl+C
            if secret_provider:
                secret_provider = None
                secret_value = ""
                status = "API key entry cancelled"
                continue
            if worker is not None and worker.is_alive():
                canceled = ctx.rust.cancel_active()
                status = "Cancelling..." if canceled else "Cancellation requested"
                continue
            break

        if key == 27:  # Esc
            if secret_provider:
                secret_provider = None
                secret_value = ""
                status = "API key entry cancelled"
            elif approval_snapshot.get("pending"):
                if approval_queue.deny_selected():
                    status = "Approval denied"
            continue

        if key == 17:  # Ctrl+Q
            break

        if key == 15:  # Ctrl+O
            if secret_provider:
                login_answer = _login_provider(ctx, secret_provider)
                status = truncate(login_answer, 80)
            continue

        if key == 1:  # Ctrl+A
            if approval_queue.approve_selected():
                status = "Approval granted"
            continue

        if key == 4:  # Ctrl+D
            if approval_queue.deny_selected():
                status = "Approval denied"
            continue

        if key == 14:  # Ctrl+N
            approval_queue.next_item()
            continue

        if key == 16:  # Ctrl+P
            if approval_snapshot.get("pending"):
                approval_queue.previous_item()
            elif not prompt:
                prompt = "/"
                palette_index = 0
            continue

        if key in (10, 13):  # Enter
            if secret_provider:
                provider = secret_provider
                api_key = secret_value.strip()
                secret_provider = None
                secret_value = ""
                if not api_key:
                    status = "API key entry cancelled"
                    continue
                local_answer = _set_provider_api_key(ctx, provider, api_key)
                logged_candidate = f"/apikey {provider} <redacted>"
                prompt_history.append(logged_candidate)
                history_index = None
                transcript_at_bottom = True
                transcript_scroll = 0
                _record_local_command_exchange(ctx, logged_candidate, local_answer)
                status = truncate(local_answer, 80)
                continue

            if approval_snapshot.get("pending"):
                if approval_queue.approve_selected():
                    status = "Approval granted"
                continue

            candidate = prompt.strip()
            if candidate:
                selected = _selected_palette_entry(prompt, palette_index)
                if selected is not None:
                    if selected.execute:
                        candidate = selected.complete_to.strip()
                    else:
                        prompt = selected.complete_to
                        palette_index = 0
                        continue
                if _is_exit_command(candidate):
                    break
                prompt = ""
                transcript_at_bottom = True
                transcript_scroll = 0
                api_key_provider = _api_key_prompt_provider(candidate)
                if api_key_provider:
                    secret_provider = api_key_provider
                    secret_value = ""
                    prompt_history.append(candidate)
                    history_index = None
                    status = f"Paste {api_key_provider} API key. Input is hidden."
                    continue
                local_answer = _handle_local_command(ctx, candidate)
                if local_answer is not None:
                    logged_candidate = _redact_local_command(candidate)
                    prompt_history.append(logged_candidate)
                    history_index = None
                    _record_local_command_exchange(ctx, logged_candidate, local_answer)
                    auth_provider = _provider_api_key_needed(ctx)
                    if auth_provider and _starts_auth_candidate(candidate):
                        secret_provider = auth_provider
                        secret_value = ""
                        status = f"{auth_provider} needs a key. Paste it here; Ctrl+O opens account/API keys."
                    else:
                        status = truncate(local_answer, 80)
                else:
                    prompt_history.append(candidate)
                    history_index = None
                    launch_turn(candidate)
            continue

        if key in (curses.KEY_BACKSPACE, 127, 8):
            if secret_provider:
                secret_value = secret_value[:-1]
            else:
                prompt = prompt[:-1]
            palette_index = 0
            continue

        if key == 9:  # Tab
            if secret_provider:
                continue
            completed = _complete_slash_command(prompt)
            if completed is not None:
                prompt = completed
                palette_index = 0
            continue

        if key == curses.KEY_UP:
            if secret_provider:
                continue
            if palette_entries:
                palette_index = max(0, palette_index - 1)
            elif prompt:
                if prompt_history:
                    if history_index is None:
                        history_index = len(prompt_history) - 1
                    else:
                        history_index = max(0, history_index - 1)
                    prompt = prompt_history[history_index]
            else:
                step = max(1, transcript_height // 2)
                if transcript_at_bottom:
                    transcript_scroll = step
                    transcript_at_bottom = False
                else:
                    transcript_scroll = min(max_transcript_scroll, transcript_scroll + step)
            continue

        if key == curses.KEY_DOWN:
            if secret_provider:
                continue
            if palette_entries:
                palette_index = min(len(palette_entries) - 1, palette_index + 1)
            elif prompt:
                if prompt_history and history_index is not None:
                    if history_index >= len(prompt_history) - 1:
                        history_index = None
                        prompt = ""
                    else:
                        history_index += 1
                        prompt = prompt_history[history_index]
            else:
                step = max(1, transcript_height // 2)
                transcript_scroll = max(0, transcript_scroll - step)
                if transcript_scroll == 0:
                    transcript_at_bottom = True
            continue

        if key == curses.KEY_PPAGE:
            step = max(1, transcript_height // 2)
            if transcript_at_bottom:
                transcript_scroll = step
                transcript_at_bottom = False
            else:
                transcript_scroll = min(max_transcript_scroll, transcript_scroll + step)
            continue

        if key == curses.KEY_NPAGE:
            step = max(1, transcript_height // 2)
            transcript_scroll = max(0, transcript_scroll - step)
            if transcript_scroll == 0:
                transcript_at_bottom = True
            continue

        if 32 <= key <= 126:
            if secret_provider:
                secret_value += chr(key)
            elif approval_snapshot.get("pending") and chr(key).casefold() in {"y", "n"}:
                approved = chr(key).casefold() == "y"
                decided = (
                    approval_queue.approve_selected()
                    if approved
                    else approval_queue.deny_selected()
                )
                if decided:
                    status = "Approval granted" if approved else "Approval denied"
            else:
                char = chr(key)
                prompt += "/" if char == "\\" and not prompt else char
            palette_index = 0
            continue


def _setup_tui_colors(stdscr: Any | None = None) -> None:
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    if not _has_tui_colors():
        return
    curses.start_color()
    theme = os.environ.get("AGENT_TUI_THEME", "").strip().casefold()
    light_theme = theme == "light"
    background = curses.COLOR_WHITE if light_theme else -1
    text = curses.COLOR_BLACK if light_theme else curses.COLOR_WHITE
    muted = curses.COLOR_BLUE if light_theme else curses.COLOR_WHITE
    pairs = [
        (COLOR_HEADER, curses.COLOR_BLUE if light_theme else curses.COLOR_CYAN, background),
        (COLOR_USER, text, background),
        (COLOR_ASSISTANT, text, background),
        (COLOR_THINKING, curses.COLOR_MAGENTA if light_theme else curses.COLOR_YELLOW, background),
        (COLOR_GUARDRAIL, curses.COLOR_MAGENTA if light_theme else curses.COLOR_YELLOW, background),
        (COLOR_MUTED, muted, background),
        (COLOR_ERROR, curses.COLOR_RED, background),
    ]
    for pair, foreground, background in pairs:
        try:
            curses.init_pair(pair, foreground, background)
        except curses.error:
            continue


def _tui_attr(pair: int, *flags: int) -> int:
    attr = curses.color_pair(pair) if _has_tui_colors() else 0
    for flag in flags:
        attr |= flag
    return attr


def _has_tui_colors() -> bool:
    try:
        return curses.has_colors()
    except curses.error:
        return False


def _draw_header(stdscr: Any, ctx: AppContext, width: int) -> None:
    root_budget = max(12, width - 58)
    title = " agent "
    detail = (
        f"session {ctx.session_id}  source {_model_source_label(_active_provider(ctx))}  model {ctx.llm.model}  "
        f"root {truncate(str(ctx.workspace_root), root_budget)}"
    )
    stdscr.addnstr(0, 0, title, width - 1, _tui_attr(COLOR_HEADER, curses.A_BOLD))
    if width > len(title):
        stdscr.addnstr(
            0,
            len(title),
            detail.ljust(max(0, width - len(title))),
            max(0, width - len(title) - 1),
            _tui_attr(COLOR_MUTED),
        )
    stdscr.addnstr(1, 0, ("-" * max(0, width - 1)), width - 1, _tui_attr(COLOR_MUTED))


def _queue_status(status: str, queued_count: int, approval_count: int = 0) -> str:
    if queued_count <= 0 and approval_count <= 0:
        return status
    parts = [status]
    if queued_count > 0:
        parts.append(f"queued {queued_count}")
    if approval_count > 0:
        parts.append(f"approvals {approval_count}")
    return " | ".join(parts)


def _draw_box_title(stdscr: Any, y: int, title: str, width: int) -> None:
    line = f" {title} "
    stdscr.addnstr(y, 0, line.ljust(width), width - 1, _tui_attr(COLOR_HEADER, curses.A_BOLD))


def _draw_lines(stdscr: Any, start_y: int, height: int, lines: list[str], scroll: int, width: int) -> None:
    if scroll <= 0:
        visible = lines[-height:]
    else:
        end = max(0, len(lines) - scroll)
        start = max(0, end - height)
        visible = lines[start:end]
    for offset, line in enumerate(visible):
        stdscr.addnstr(start_y + offset, 0, line.ljust(width), width - 1, _line_attr(line))
    for offset in range(len(visible), height):
        stdscr.addnstr(start_y + offset, 0, " ".ljust(width), width - 1)


def _slash_command_lines(prompt: str, width: int, *, selected_index: int = 0) -> list[str]:
    entries = _slash_palette_entries(prompt)
    if not entries:
        return []
    title = _slash_palette_title(prompt)
    lines = [_clip_line(title, width)]
    visible_count = 8
    selected_index = _selectable_palette_index(entries, selected_index)
    start = min(
        max(0, selected_index - visible_count + 1),
        max(0, len(entries) - visible_count),
    )
    for index, entry in enumerate(entries[start : start + visible_count], start=start):
        marker = ">" if index == selected_index else " "
        if _palette_entry_is_model(entry):
            lines.append(f"{marker} {entry.label}")
            if entry.description:
                lines.append(f"    {entry.description}")
        else:
            lines.append(_clip_line(f"{marker} {entry.label:<16} {entry.description}", width))
    return lines


def _slash_palette_entries(prompt: str, *, ctx: Any | None = None) -> list[PaletteEntry]:
    prompt = _normalized_command_prompt(prompt)
    if not prompt.startswith("/"):
        return []

    stripped = prompt.strip()
    parts = stripped.split()
    provider_command = _provider_argument_command(prompt, parts)
    if provider_command is not None:
        return _provider_palette_entries(provider_command, parts[1] if len(parts) >= 2 else "")
    if _is_install_palette_prompt(prompt, parts):
        return _install_palette_entries(parts[1] if len(parts) >= 2 else "")
    if _is_model_palette_prompt(prompt, parts):
        return _model_palette_entries(parts[1] if len(parts) >= 2 else "", ctx=ctx)
    if _is_reasoning_palette_prompt(prompt, parts):
        return _reasoning_palette_entries(parts[1] if len(parts) >= 2 else "")
    if _is_setup_palette_prompt(prompt, parts):
        return _setup_palette_entries(parts[1] if len(parts) >= 2 else "")

    query = parts[0].casefold() if parts else "/"
    matches = [
        PaletteEntry(
            value=name,
            label=name,
            description=description,
            complete_to=_slash_command_complete_to(name),
            execute=_slash_command_executes(name),
        )
        for name, description in LOCAL_COMMANDS
        if name.casefold().startswith(query)
    ]
    matches.sort(key=lambda entry: entry.value.casefold() != query)
    if not matches:
        matches = [
            PaletteEntry(
                value=name,
                label=name,
                description=description,
                complete_to=_slash_command_complete_to(name),
                execute=_slash_command_executes(name),
            )
            for name, description in LOCAL_COMMANDS
        ]
    return matches


def _slash_command_complete_to(name: str) -> str:
    if name in PROVIDER_ARGUMENT_COMMANDS | {"/model", "/name", "/install", "/reasoning", "/setup"}:
        return f"{name} "
    return name


def _slash_command_executes(name: str) -> bool:
    return name not in PROVIDER_ARGUMENT_COMMANDS | {"/model", "/name", "/install", "/reasoning", "/setup"}


def _is_setup_palette_prompt(prompt: str, parts: list[str]) -> bool:
    return (
        len(parts) <= 2
        and bool(parts)
        and parts[0].casefold() in {"/setup", "/connect"}
        and (prompt.endswith(" ") or len(parts) == 2)
    )


def _setup_palette_entries(query: str) -> list[PaletteEntry]:
    choices = [
        ("ollama", "Ollama", "private local model · no account"),
        *[
            (provider, _model_source_label(provider), "connect with an API key")
            for provider in _setup_available_providers()[:3]
        ],
        ("more", "More options", "other providers and local runtimes"),
        ("status", "Connection status", "check the active model")
    ]
    normalized = query.casefold()
    matches = [choice for choice in choices if choice[0].startswith(normalized) or choice[1].casefold().startswith(normalized)]
    return [
        PaletteEntry(
            value=value,
            label=label,
            description=description,
            complete_to=f"/setup {value}",
            execute=True,
        )
        for value, label, description in (matches or choices)
    ]


def _provider_palette_entries(command: str, query: str) -> list[PaletteEntry]:
    normalized = query.casefold()
    providers = sorted(
        AVAILABLE_PROVIDERS,
        key=lambda item: PROVIDER_SORT_ORDER.get(item, 99),
    )
    matches = [provider for provider in providers if provider.casefold().startswith(normalized)]
    if not matches:
        matches = providers
    return [
        PaletteEntry(
            value=provider,
            label=_model_source_label(provider),
            description=_provider_palette_description(command, provider),
            complete_to=f"{command} {provider}",
            execute=True,
        )
        for provider in matches
    ]


def _provider_palette_description(command: str, provider: str) -> str:
    if command in {"/apikey", "/key"}:
        env_name = PROVIDER_API_KEY_ENVS.get(provider)
        return f"load {env_name}" if env_name else "no API key"
    if command in {"/login", "/auth"}:
        return "open account/API keys" if provider in PROVIDER_LOGIN_URLS else "local app"
    return f"default model: {PROVIDER_MODEL_HINTS.get(provider, ('custom-model',))[0]}"


def _model_palette_entries(
    query: str,
    *,
    ctx: Any | None = None,
) -> list[PaletteEntry]:
    normalized = query.casefold()
    options = _model_options_for_display(ctx) if ctx is not None else _model_options()
    matches = [
        option for option in options
        if option["model"].casefold().startswith(normalized)
        or option["provider"].casefold().startswith(normalized)
    ]
    if not matches:
        matches = options

    entries = [_model_palette_entry(option, with_state=ctx is not None) for option in matches]
    if ctx is None:
        return entries

    if normalized:
        return entries

    grouped: list[PaletteEntry] = []
    current_provider: str | None = None
    for entry, option in zip(entries, matches):
        provider = option["provider"]
        if provider != current_provider:
            current_provider = provider
            grouped.append(_palette_section(_model_source_label(provider), "/model "))
        grouped.append(entry)
    return grouped


def _model_palette_entry(option: dict[str, Any], *, with_state: bool) -> PaletteEntry:
    provider = option["provider"]
    model = option["model"]
    return PaletteEntry(
        value=f"{provider}/{model}",
        label=model,
        description=(
            f"{_model_source_label(provider)} · {_model_state_label(option)}"
            f"{_model_metadata_suffix(provider, model)}"
            if with_state
            else (
                f"{_model_source_label(provider)}: {_provider_access_label(provider)}"
                f"{_model_metadata_suffix(provider, model)}"
            )
        ),
        complete_to=f"/model {provider} {model}",
        execute=True,
    )


def _palette_section(label: str, complete_to: str) -> PaletteEntry:
    return PaletteEntry(
        value=f"section:{label.casefold().replace(' ', '-')}",
        label=f"── {label} ──",
        description="",
        complete_to=complete_to,
        execute=False,
    )


def _install_palette_entries(query: str) -> list[PaletteEntry]:
    normalized = query.casefold()
    entries = [
        entry
        for entry in LOCAL_INSTALL_CATALOG
        if entry["model"].casefold().startswith(normalized)
        or entry["provider"].casefold().startswith(normalized)
    ]
    if not entries:
        entries = list(LOCAL_INSTALL_CATALOG)
    return [
        PaletteEntry(
            value=f"{entry['provider']}/{entry['model']}",
            label=entry["model"],
            description=(
                "Open-source/open-weight · "
                f"Provider: {_model_source_label(entry['provider'])} · "
                f"{entry['parameters']} params · {entry['size']} · {entry['memory']} · "
                "preview first · installs locally · no login"
            ),
            complete_to=f"/install {entry['provider']} {entry['model']}",
            execute=True,
        )
        for entry in entries
    ]


def _reasoning_palette_entries(query: str) -> list[PaletteEntry]:
    descriptions = {
        "minimal": "fastest · smallest reasoning budget",
        "low": "quick tasks and small edits",
        "medium": "balanced default",
        "high": "complex debugging and architecture",
    }
    normalized = query.casefold()
    efforts = [effort for effort in descriptions if effort.startswith(normalized)] or list(descriptions)
    return [
        PaletteEntry(
            value=effort,
            label=effort,
            description=descriptions[effort],
            complete_to=f"/reasoning {effort}",
            execute=True,
        )
        for effort in efforts
    ]


def _model_metadata_suffix(provider: str, model: str) -> str:
    entry = _local_install_entry(provider, model)
    if entry is None:
        return ""
    return f" · {entry['parameters']} params · {entry['size']} · {entry['context']} ctx"


def _slash_palette_title(prompt: str) -> str:
    prompt = _normalized_command_prompt(prompt)
    stripped = prompt.strip()
    parts = stripped.split()
    if _provider_argument_command(prompt, parts) is not None:
        return "Model sources"
    if _is_install_palette_prompt(prompt, parts):
        return "Open-source/open-weight models · choose local provider"
    if _is_model_palette_prompt(prompt, parts):
        return "Models"
    if _is_reasoning_palette_prompt(prompt, parts):
        return "Reasoning effort · raw chain-of-thought stays private"
    return "Commands"


def _provider_argument_command(prompt: str, parts: list[str]) -> str | None:
    if not parts:
        return None
    command = parts[0].casefold()
    if command not in PROVIDER_ARGUMENT_COMMANDS or len(parts) > 2:
        return None
    if prompt.startswith(f"{command} ") or prompt.endswith(" "):
        return command
    return None


def _is_model_palette_prompt(prompt: str, parts: list[str]) -> bool:
    return (
        len(parts) <= 2
        and (prompt.startswith("/model ") or (len(parts) >= 1 and parts[0].casefold() == "/model" and prompt.endswith(" ")))
    )


def _is_install_palette_prompt(prompt: str, parts: list[str]) -> bool:
    return (
        len(parts) <= 2
        and (
            prompt.startswith("/install ")
            or (
                len(parts) >= 1
                and parts[0].casefold() == "/install"
                and prompt.endswith(" ")
            )
        )
    )


def _is_reasoning_palette_prompt(prompt: str, parts: list[str]) -> bool:
    return (
        len(parts) <= 2
        and (
            prompt.startswith("/reasoning ")
            or (parts and parts[0].casefold() == "/reasoning" and prompt.endswith(" "))
        )
    )


def _selected_palette_entry(prompt: str, selected_index: int) -> PaletteEntry | None:
    entries = _slash_palette_entries(prompt)
    if not entries:
        return None
    return entries[_selectable_palette_index(entries, selected_index)]


def _selectable_palette_index(entries: list[PaletteEntry], selected_index: int) -> int:
    selected_index = min(max(0, selected_index), len(entries) - 1)
    if _palette_entry_selectable(entries[selected_index]):
        return selected_index
    for index in range(selected_index + 1, len(entries)):
        if _palette_entry_selectable(entries[index]):
            return index
    for index in range(selected_index - 1, -1, -1):
        if _palette_entry_selectable(entries[index]):
            return index
    return selected_index


def _palette_entry_selectable(entry: PaletteEntry) -> bool:
    return not entry.value.startswith("section:")


def _palette_entry_is_model(entry: PaletteEntry) -> bool:
    return entry.execute and entry.complete_to.startswith("/model ")


def _complete_slash_command(prompt: str) -> str | None:
    prompt = _normalized_command_prompt(prompt)
    selected = _selected_palette_entry(prompt, 0)
    parts = prompt.strip().split()
    if selected is not None and (
        _provider_argument_command(prompt, parts) is not None
        or prompt.startswith("/model ")
        or prompt.startswith("/install ")
        or prompt.startswith("/reasoning ")
    ):
        return selected.complete_to
    if not prompt.startswith("/") or " " in prompt:
        return None
    query = prompt.casefold()
    matches = [name for name, _description in LOCAL_COMMANDS if name.casefold().startswith(query)]
    if len(matches) == 1:
        return f"{matches[0]} "
    return None


def _normalized_command_prompt(prompt: str) -> str:
    if prompt.startswith("\\"):
        return f"/{prompt[1:]}"
    return prompt


def _draw_command_palette(stdscr: Any, y: int, lines: list[str], width: int) -> None:
    for offset, line in enumerate(lines):
        attr = _tui_attr(COLOR_HEADER, curses.A_BOLD) if offset == 0 else _tui_attr(COLOR_MUTED)
        stdscr.addnstr(y + offset, 0, line.ljust(width), width - 1, attr)


def _line_attr(line: str) -> int:
    stripped = line.strip()
    if stripped.startswith("You"):
        return _tui_attr(COLOR_USER, curses.A_BOLD)
    if stripped.startswith("Agent"):
        return _tui_attr(COLOR_ASSISTANT, curses.A_BOLD)
    if stripped.startswith("Thinking") or stripped.startswith("Tool") or stripped.startswith("Result"):
        return _tui_attr(COLOR_THINKING)
    if stripped.startswith("Guardrail"):
        return _tui_attr(COLOR_GUARDRAIL, curses.A_BOLD)
    if stripped.startswith("Error"):
        return _tui_attr(COLOR_ERROR, curses.A_BOLD)
    if stripped.startswith("Activity"):
        return _tui_attr(COLOR_MUTED, curses.A_BOLD)
    return 0


def _draw_usage_panel(
    stdscr: Any,
    *,
    x: int,
    y: int,
    height: int,
    width: int,
    session: SessionInfo,
    model: str,
    provider: str,
    approvals: dict[str, Any] | None = None,
) -> None:
    if width <= 3 or height <= 0:
        return
    separator_attr = _tui_attr(COLOR_MUTED)
    for offset in range(height):
        stdscr.addnstr(y + offset, x, "|", 1, separator_attr)

    panel_x = x + 2
    panel_width = max(1, width - 2)
    lines = _usage_panel_lines(session, model, provider, panel_width)
    approval_lines = _approval_panel_lines(approvals or {"pending": [], "selected_index": 0}, panel_width)
    lines.extend([""] + approval_lines)
    for offset in range(height):
        text = lines[offset] if offset < len(lines) else ""
        attr = _panel_line_attr(text)
        stdscr.addnstr(y + offset, panel_x, text.ljust(panel_width), panel_width - 1, attr)


def _panel_line_attr(line: str) -> int:
    if line.startswith("Usage"):
        return _tui_attr(COLOR_HEADER, curses.A_BOLD)
    if line.startswith("Guardrails"):
        return _tui_attr(COLOR_GUARDRAIL, curses.A_BOLD)
    if line.startswith("Approvals"):
        return _tui_attr(COLOR_HEADER, curses.A_BOLD)
    if line.startswith("> "):
        return _tui_attr(COLOR_HEADER, curses.A_BOLD)
    if line.startswith("Source") or line.startswith("Provider") or line.startswith("Model"):
        return _tui_attr(COLOR_ASSISTANT, curses.A_BOLD)
    if line.startswith("Context") or line.startswith("Tokens") or line.startswith("Cost"):
        return _tui_attr(COLOR_ASSISTANT, curses.A_BOLD)
    if line.startswith("Approve ") or line.startswith("Deny ") or line.startswith("Pending"):
        return _tui_attr(COLOR_GUARDRAIL)
    return _tui_attr(COLOR_MUTED)


def _usage_panel_lines(session: SessionInfo, model: str, provider: str, width: int) -> list[str]:
    usage = session.tokens
    total_tokens = _billable_token_total(usage)
    context_limit = _context_window_for_model(model)
    percent = _usage_percent(total_tokens, context_limit)
    lines = [
        "Usage",
        "",
        f"Source     {_model_source_label(provider)}",
        f"Model      {model}",
        "",
        f"Tokens     {_format_count(total_tokens)}",
        f"Context    {_format_percent(percent)}",
        f"Cost       {_format_cost(session.cost_usd)}",
        "",
        f"Input      {_format_count(usage.input)}",
        f"Output     {_format_count(usage.output)}",
        f"Reasoning  {_format_count(usage.reasoning)}",
        f"Cached     {_format_count(usage.cache_read)}",
        "",
        "Guardrails",
        "Visible when tools are",
        "blocked or need approval.",
    ]
    if context_limit is None:
        lines.insert(5, "Context limit unknown")
    return [_clip_line(line, width) for line in lines]


def _approval_panel_lines(approvals: dict[str, Any], width: int) -> list[str]:
    pending = approvals.get("pending") if isinstance(approvals, dict) else []
    selected_index = approvals.get("selected_index") if isinstance(approvals, dict) else 0
    pending_items = pending if isinstance(pending, list) else []
    if not pending_items:
        return [_clip_line("Approvals", width), "", _clip_line("None pending", width)]

    lines = [_clip_line("Approvals", width), ""]
    for index, item in enumerate(pending_items[:6], start=1):
        tool = _approval_text(item.get("tool"))
        path = _approval_display_text(item)
        reason = _approval_text(item.get("reason"))
        prefix = ">" if index - 1 == selected_index else " "
        lines.append(_clip_line(f"{prefix} {index}. {tool}", width))
        if path:
            lines.append(_clip_line(f"   {path}", width))
        if reason:
            lines.append(_clip_line(f"   {reason}", width))
    lines.extend([
        "",
        _clip_line("Enter/Y approve", width),
        _clip_line("N/Esc deny", width),
    ])
    return lines


def _approval_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def _approval_display_text(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    raw = _approval_text(
        item.get("display_path")
        or item.get("translated_path")
        or item.get("resolved_path")
        or item.get("requested_path")
    ).strip()
    if not raw:
        return ""
    parts = raw.split()
    if len(parts) >= 3 and parts[0] == "desktop":
        action = parts[1]
        target = parts[2]
        if target.startswith(("windows-app:", "windows-shortcut:")):
            label = parts[-1] if ":" not in parts[-1] else "selected app"
            return f"desktop {action} {label}"
        if action in {"focus_window", "close_window", "minimize_window", "maximize_window", "restore_window"}:
            try:
                int(target[2:] if target.lower().startswith("0x") else target, 16 if target.lower().startswith("0x") else 10)
            except ValueError:
                pass
            else:
                return f"desktop {action} selected window"
    return raw


def _compact_usage_text(session: SessionInfo, model: str, provider: str = "openai") -> str:
    total_tokens = _billable_token_total(session.tokens)
    percent = _usage_percent(total_tokens, _context_window_for_model(model))
    return (
        f"{provider}/{model}"
        f" tokens {_format_count(total_tokens)}"
        f" ({_format_percent(percent)})"
        f" cost {_format_cost(session.cost_usd)}"
    )


def _billable_token_total(usage: TokenUsage) -> int:
    return max(0, usage.input) + max(0, usage.output)


def _context_window_for_model(model: str) -> int | None:
    env_value = os.environ.get("AGENT_CONTEXT_WINDOW_TOKENS", "").strip()
    if env_value:
        try:
            parsed = int(env_value.replace("_", ""))
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    normalized = model.strip().casefold()
    if normalized in DEFAULT_CONTEXT_WINDOWS:
        return DEFAULT_CONTEXT_WINDOWS[normalized]
    for prefix, limit in DEFAULT_CONTEXT_WINDOWS.items():
        if normalized.startswith(prefix.casefold()):
            return limit
    return None


def _usage_percent(tokens: int, context_limit: int | None) -> float | None:
    if context_limit is None or context_limit <= 0:
        return None
    return max(0.0, (tokens / context_limit) * 100.0)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.1:
        return "<0.1%"
    if value < 10:
        return f"{value:.1f}%"
    return f"{value:.0f}%"


def _format_count(value: int) -> str:
    value = max(0, int(value))
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def _format_cost(value: float) -> str:
    value = max(0.0, float(value))
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _clip_line(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return f"{value[: width - 1]}..."


def _draw_status_line(
    stdscr: Any,
    y: int,
    width: int,
    *,
    status: str,
    live_turn: dict[str, Any],
    worker_alive: bool,
    error: str | None,
    usage_summary: str | None = None,
    auth_active: bool = False,
    approval_active: bool = False,
) -> None:
    footer = f" {status}"
    if worker_alive:
        footer = f"{footer}  working"
    phase = live_turn.get("phase")
    if isinstance(phase, str) and phase not in {"idle", "completed"}:
        footer = f"{footer}  {phase}"
    if error:
        footer = f"{footer}  {truncate(error, max(0, width - 18))}"
    if usage_summary:
        footer = f"{footer}  {usage_summary}"
    help_text = (
        "Enter save key  Ctrl+O open account  Ctrl+C cancel"
        if auth_active
        else "Enter/Y approve  N/Esc deny"
        if approval_active
        else "Ctrl+C cancel/exit  PgUp/PgDn scroll"
    )
    gap = max(1, width - len(footer) - len(help_text) - 1)
    line = f"{footer}{' ' * gap}{help_text}"
    attr = _tui_attr(COLOR_ERROR, curses.A_BOLD) if error else _tui_attr(COLOR_MUTED)
    stdscr.addnstr(y, 0, line.ljust(width), width - 1, attr)


def _draw_input_line(
    stdscr: Any,
    y: int,
    width: int,
    prompt: str,
    *,
    label: str = " agent> ",
) -> None:
    body_width = max(0, width - len(label))
    visible_prompt = prompt[-body_width:]
    line = f"{label}{visible_prompt}"
    stdscr.addnstr(y, 0, line.ljust(width), width - 1, _tui_attr(COLOR_ASSISTANT, curses.A_REVERSE))
    cursor_x = min(width - 1, len(label) + len(visible_prompt))
    stdscr.move(y, cursor_x)


def _render_tui_transcript(
    messages: list[Any],
    live_turn: dict[str, Any],
    width: int,
    *,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> list[str]:
    active = bool(live_turn.get("active"))
    error = live_turn.get("error")
    prompt = live_turn.get("prompt", "")
    visible_messages = list(messages)

    if active and prompt and visible_messages:
        last = visible_messages[-1]
        if (
            getattr(last, "role", None) == "user"
            and getattr(last, "content", "").strip() == str(prompt).strip()
        ):
            visible_messages = visible_messages[:-1]

    lines = _render_messages(visible_messages, width, agent_name=agent_name) if visible_messages else []
    has_recent_subagents = any(
        isinstance(item, (list, tuple))
        and len(item) == 2
        and item[0] == "subagent"
        for item in live_turn.get("feed", [])
    )
    live_lines = (
        _render_live_turn(live_turn, width, agent_name=agent_name)
        if active or error or has_recent_subagents
        else []
    )

    if lines and live_lines:
        return [*lines, *live_lines]
    if lines:
        return lines
    if live_lines:
        return live_lines
    return ["No messages yet."]


def _render_messages(
    messages: list[Any], width: int, *, agent_name: str = DEFAULT_AGENT_NAME
) -> list[str]:
    lines: list[str] = []
    for message in messages:
        speaker = "You" if message.role == "user" else agent_name if message.role == "assistant" else message.role.title()
        lines.append(f"{speaker}  {message.created_at}")
        body = message.content.strip() or "<empty>"
        for paragraph in body.splitlines() or [""]:
            wrapped = textwrap.wrap(
                paragraph,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            if wrapped:
                lines.extend(f"  {line}" for line in wrapped)
            else:
                lines.append("")
        for attachment in getattr(message, "attachments", ()):
            lines.extend(_wrap_lines(
                f"Attachment: {attachment.filename} ({attachment.mime}, {attachment.size_bytes} bytes)",
                width,
                indent="  ",
            ))
        lines.append("")
    return lines or ["No messages yet."]


def _render_live_turn(
    live_turn: dict[str, Any], width: int, *, agent_name: str = DEFAULT_AGENT_NAME
) -> list[str]:
    phase = live_turn.get("phase")
    feed = live_turn.get("feed", [])
    active = live_turn.get("active", False)

    if not active and phase in {"idle", None} and not live_turn.get("error"):
        return []

    lines: list[str] = []
    prompt = live_turn.get("prompt", "")
    if prompt:
        lines.append(f"You")
        lines.extend(_wrap_lines(prompt, width, indent="  "))
        lines.append("")
        lines.append(agent_name)

    prev_kind: str | None = None
    for kind, content in feed:
        if kind in {"thinking", "reasoning"}:
            if prev_kind not in {"thinking", "reasoning"}:
                lines.append("  Activity")
            lines.extend(_wrap_lines(str(content), width, indent="    · "))
        elif kind == "text":
            for para in content.splitlines() or [""]:
                pieces = textwrap.wrap(para, width=max(1, width - 2), break_long_words=True) or [""]
                lines.extend(f"  {p}" for p in pieces)
        elif kind == "tool":
            if prev_kind != "tool":
                lines.append("  Activity")
            lines.extend(_wrap_lines(str(content), width, indent="    "))
        elif kind == "tool_result":
            lines.extend(_wrap_lines(str(content), width, indent="    Result: "))
        elif kind == "guardrail":
            lines.extend(_wrap_lines(str(content), width, indent="    Guardrail: "))
        elif kind == "subagent":
            if prev_kind != "subagent":
                lines.append("  Subagents")
            lines.extend(_wrap_lines(str(content), width, indent="    ↳ "))
        elif kind == "tool_error":
            lines.extend(_wrap_lines(str(content), width, indent="    Error: "))
        prev_kind = kind

    if active and not feed:
        lines.append("  Activity")
        lines.append("    · Working")

    error = live_turn.get("error")
    if error:
        lines.append(f"  Error: {truncate(error, width - 9)}")

    lines.append("")
    return lines


def _wrap_lines(text: str, width: int, *, indent: str = "") -> list[str]:
    wrapped: list[str] = []
    for paragraph in text.splitlines() or [""]:
        pieces = textwrap.wrap(
            paragraph,
            width=max(1, width - len(indent)),
            break_long_words=False,
            break_on_hyphens=False,
        )
        if pieces:
            wrapped.extend(f"{indent}{piece}" for piece in pieces)
        else:
            wrapped.append(indent.rstrip())
    return wrapped


if __name__ == "__main__":
    raise SystemExit(main())
