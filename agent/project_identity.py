from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectIdentity:
    workspace_root: Path
    aliases: tuple[str, ...] = field(default_factory=tuple)
    sources: tuple[str, ...] = field(default_factory=tuple)

    def matches(self, value: str) -> bool:
        normalized = _normalize_scope_text(value)
        if not normalized:
            return False
        return normalized in {_normalize_scope_text(alias) for alias in self.aliases}


def build_project_identity(workspace_root: Path) -> ProjectIdentity:
    root = workspace_root.expanduser().resolve()
    aliases: list[str] = []
    sources: list[str] = []

    def add_alias(value: str | None, source: str) -> None:
        if not value:
            return
        cleaned = _clean_alias(value)
        if not cleaned:
            return
        if cleaned.casefold() in {_clean_alias(existing).casefold() for existing in aliases}:
            return
        aliases.append(cleaned)
        sources.append(source)

    add_alias(root.name, "workspace_root")
    add_alias(_git_root_name(root), "git_root")
    add_alias(_pyproject_name(root / "pyproject.toml"), "pyproject.toml")
    add_alias(_package_json_name(root / "package.json"), "package.json")
    add_alias(_cargo_name(root / "Cargo.toml"), "Cargo.toml")
    add_alias(_readme_heading(root), "README")

    return ProjectIdentity(
        workspace_root=root,
        aliases=tuple(aliases),
        sources=tuple(sources),
    )


def resolve_workspace_alias(scope: str, workspace_root: Path) -> Path | None:
    identity = build_project_identity(workspace_root)
    if identity.matches(scope):
        return identity.workspace_root
    return None


def identity_text(workspace_root: Path) -> str:
    identity = build_project_identity(workspace_root)
    if not identity.aliases:
        return ""
    lines = [
        "Workspace identity:",
        f"  root: {identity.workspace_root}",
        f"  aliases: {', '.join(identity.aliases)}",
    ]
    if identity.sources:
        lines.append(f"  sources: {', '.join(identity.sources)}")
    return "\n".join(lines)


def _git_root_name(root: Path) -> str | None:
    current = root if root.is_dir() else root.parent
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate.name
    return None


def _pyproject_name(path: Path) -> str | None:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project")
    if isinstance(project, dict):
        name = project.get("name")
        if isinstance(name, str):
            return name
    tool = data.get("tool")
    if isinstance(tool, dict):
        poetry = tool.get("poetry")
        if isinstance(poetry, dict):
            name = poetry.get("name")
            if isinstance(name, str):
                return name
    return None


def _package_json_name(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        name = data.get("name")
        if isinstance(name, str):
            return name
    return None


def _cargo_name(path: Path) -> str | None:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return None
    package = data.get("package")
    if isinstance(package, dict):
        name = package.get("name")
        if isinstance(name, str):
            return name
    return None


def _readme_heading(root: Path) -> str | None:
    for candidate in (root / "README.md", root / "README.txt", root / "README"):
        try:
            lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
                if heading:
                    return heading
                break
    return None


def _clean_alias(value: str) -> str:
    text = " ".join(value.replace("_", " ").replace("-", " ").split())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_scope_text(value: str) -> str:
    text = _clean_alias(value).casefold()
    text = re.sub(r"^(my|the|this|that|current)\s+", "", text)
    return text.strip()
