from __future__ import annotations

import os
from pathlib import Path


def default_system_prompt_path() -> Path:
    return Path(__file__).resolve().parent / "prompts" / "system.txt"


def system_prompt_path() -> Path:
    override = os.environ.get("NYM_SYSTEM_PROMPT_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return default_system_prompt_path()


def load_system_prompt() -> str:
    path = system_prompt_path()
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"System prompt file not found: {path}") from exc

    if not text:
        raise RuntimeError(f"System prompt file is empty: {path}")

    return text
