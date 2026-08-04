from __future__ import annotations

import os
from pathlib import Path


def prompt_directory() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def default_system_prompt_path() -> Path:
    return prompt_directory() / "system.txt"


def system_prompt_path() -> Path:
    override = os.environ.get("AGENT_SYSTEM_PROMPT_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return default_system_prompt_path()


def _read_prompt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"System prompt file not found: {path}") from exc

    if not text:
        raise RuntimeError(f"System prompt file is empty: {path}")

    return text


def _model_prompt_paths(provider: str | None, model: str | None) -> tuple[Path, ...]:
    paths = [prompt_directory() / "default.txt"]
    provider_name = (provider or "").strip().casefold()
    model_name = (model or "").strip().casefold()
    if model_name.startswith("gpt-") or (
        provider_name == "openai" and "gpt" in model_name
    ):
        paths.append(prompt_directory() / "gpt.txt")
    return tuple(paths)


def load_system_prompt(
    *, provider: str | None = None, model: str | None = None
) -> str:
    override = os.environ.get("AGENT_SYSTEM_PROMPT_PATH")
    if override:
        return _read_prompt(Path(override).expanduser().resolve())

    paths = (*_model_prompt_paths(provider, model), default_system_prompt_path())
    return "\n\n".join(_read_prompt(path) for path in paths)
