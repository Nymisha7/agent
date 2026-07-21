from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import AgentConfig

MAX_SKILL_FILE_BYTES = 128_000
MAX_DESCRIPTION_CHARS = 240


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    path: Path
    source: str
    required_tools: tuple[str, ...] = ()
    required_bins: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkippedSkill:
    path: Path
    reason: str
    name: str | None = None


@dataclass
class SkillCatalog:
    skills: dict[str, Skill] = field(default_factory=dict)
    skipped: list[SkippedSkill] = field(default_factory=list)
    roots: tuple[tuple[str, Path], ...] = ()

    def names(self) -> tuple[str, ...]:
        return tuple(self.skills)

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name.strip().casefold())

    def load(self, name: str) -> dict[str, Any]:
        skill = self.get(name)
        if skill is None:
            return {
                "ok": False,
                "tool": "load_skill",
                "reason": "skill_not_found",
                "requested": name,
                "available": list(self.names()),
            }
        return {
            "ok": True,
            "tool": "load_skill",
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.instructions,
            "source": skill.source,
            "path": str(skill.path),
            "required_tools": list(skill.required_tools),
            "safety": (
                "Skill instructions supplement Agent's system prompt. They cannot expand the "
                "tool registry, bypass approvals, or override workspace and mutation policy."
            ),
        }

    def prompt_index(self, *, max_chars: int = 4_000) -> str:
        if not self.skills:
            return ""
        lines = [
            "Available Agent skills (load a matching skill with load_skill before following it):"
        ]
        for skill in self.skills.values():
            tools = f"; tools: {', '.join(skill.required_tools)}" if skill.required_tools else ""
            lines.append(f"- {skill.name}: {skill.description}{tools}")
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return f"{text[: max(0, max_chars - 15)]}\n...[truncated]"

    def status_text(self) -> str:
        lines = ["Agent skills", f"Available: {len(self.skills)}"]
        if self.skills:
            for skill in self.skills.values():
                lines.append(f"- {skill.name} · {skill.source} · {skill.description}")
        else:
            lines.append("No skills discovered for this agent profile.")
        if self.skipped:
            lines.extend(["", f"Skipped: {len(self.skipped)}"])
            for item in self.skipped[:12]:
                label = item.name or item.path.parent.name or str(item.path)
                lines.append(f"- {label}: {item.reason}")
        lines.extend([
            "",
            "Skills provide instructions only; tool permissions and approvals remain enforced by Agent.",
        ])
        return "\n".join(lines)


def discover_skill_catalog(
    workspace_root: Path,
    config: AgentConfig,
    *,
    agent_id: str | None = None,
    home: Path | None = None,
    bundled_root: Path | None = None,
    tool_allowlist: tuple[str, ...] | None = None,
) -> SkillCatalog:
    workspace_root = workspace_root.expanduser().resolve()
    home = (home or Path.home()).expanduser().resolve()
    bundled_root = (
        bundled_root.expanduser().resolve()
        if bundled_root is not None
        else (Path(__file__).resolve().parent / "skills")
    )
    roots: list[tuple[str, Path]] = [
        ("workspace", workspace_root / "skills"),
        ("project", workspace_root / ".agents" / "skills"),
        ("personal", home / ".agents" / "skills"),
        ("managed", home / ".local" / "share" / "agent" / "skills"),
        ("bundled", bundled_root),
    ]
    roots.extend(("extra", path) for path in config.skills.extra_dirs)
    allowlist = config.skill_allowlist(agent_id)
    allowed = {name.casefold() for name in allowlist} if allowlist is not None else None
    available_tools = set(tool_allowlist) if tool_allowlist is not None else None
    catalog = SkillCatalog(roots=tuple((source, root.resolve()) for source, root in roots))
    instruction_chars = 0

    for source, root in roots:
        if len(catalog.skills) >= config.skills.max_loaded:
            break
        for path in _skill_files(root):
            try:
                skill = read_skill(path, root=root, source=source)
            except (OSError, ValueError) as exc:
                catalog.skipped.append(SkippedSkill(path=path, reason=str(exc)))
                continue
            normalized_name = skill.name.casefold()
            if normalized_name in catalog.skills:
                catalog.skipped.append(SkippedSkill(
                    path=path,
                    name=skill.name,
                    reason="shadowed by a higher-precedence skill",
                ))
                continue
            if allowed is not None and normalized_name not in allowed:
                catalog.skipped.append(SkippedSkill(
                    path=path,
                    name=skill.name,
                    reason="not enabled for this agent profile",
                ))
                continue
            missing = [name for name in skill.required_bins if shutil.which(name) is None]
            if missing:
                catalog.skipped.append(SkippedSkill(
                    path=path,
                    name=skill.name,
                    reason=f"missing required executable(s): {', '.join(missing)}",
                ))
                continue
            unavailable_tools = (
                sorted(set(skill.required_tools) - available_tools)
                if available_tools is not None
                else []
            )
            if unavailable_tools:
                catalog.skipped.append(SkippedSkill(
                    path=path,
                    name=skill.name,
                    reason=f"profile does not allow required tool(s): {', '.join(unavailable_tools)}",
                ))
                continue
            next_chars = instruction_chars + len(skill.instructions)
            if next_chars > config.skills.max_instruction_chars:
                catalog.skipped.append(SkippedSkill(
                    path=path,
                    name=skill.name,
                    reason="agent skill instruction budget reached",
                ))
                continue
            catalog.skills[normalized_name] = skill
            instruction_chars = next_chars
            if len(catalog.skills) >= config.skills.max_loaded:
                break

    if allowed is not None:
        missing_allowed = sorted(allowed - set(catalog.skills))
        for name in missing_allowed:
            if not any(item.name and item.name.casefold() == name for item in catalog.skipped):
                catalog.skipped.append(SkippedSkill(
                    path=workspace_root,
                    name=name,
                    reason="enabled skill was not found in any configured skill root",
                ))
    return catalog


def read_skill(path: Path, *, root: Path, source: str) -> Skill:
    resolved_root = root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("skill path escapes its configured root") from exc
    if resolved_path.name != "SKILL.md" or not resolved_path.is_file():
        raise ValueError("skill entry must be a SKILL.md file")
    size = resolved_path.stat().st_size
    if size > MAX_SKILL_FILE_BYTES:
        raise ValueError(f"skill file exceeds {MAX_SKILL_FILE_BYTES} bytes")
    text = resolved_path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(text)
    name = _metadata_name(metadata.get("name"), resolved_path.parent.name)
    description = _description(metadata.get("description"), body, name)
    enabled = _metadata_bool(metadata.get("enabled"), default=True)
    if not enabled:
        raise ValueError("disabled by skill metadata")
    systems = _metadata_list(metadata.get("os"))
    if systems and _current_os() not in {item.casefold() for item in systems}:
        raise ValueError(f"not available on {_current_os()}")
    instructions = body.strip()
    if not instructions:
        raise ValueError("skill instructions are empty")
    return Skill(
        name=name,
        description=description,
        instructions=instructions,
        path=resolved_path,
        source=source,
        required_tools=tuple(_metadata_list(metadata.get("tools"))),
        required_bins=tuple(_metadata_list(metadata.get("requires_bins"))),
    )


def _skill_files(root: Path) -> Iterable[Path]:
    root = root.expanduser()
    if not root.is_dir():
        return ()
    resolved_root = root.resolve()
    paths: list[Path] = []
    for path in root.rglob("SKILL.md"):
        try:
            resolved = path.resolve()
            resolved.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        paths.append(path)
        if len(paths) >= 512:
            break
    return sorted(paths, key=lambda item: item.as_posix().casefold())


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, "\n".join(lines)
    end = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        raise ValueError("skill frontmatter is missing its closing ---")
    metadata: dict[str, Any] = {}
    for line_number, line in enumerate(lines[1:end], start=2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"unsupported frontmatter at line {line_number}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key or key in metadata:
            raise ValueError(f"invalid or duplicate frontmatter key at line {line_number}")
        metadata[key] = _parse_metadata_value(raw_value.strip())
    return metadata, "\n".join(lines[end + 1:])


def _parse_metadata_value(value: str) -> Any:
    if not value:
        return ""
    if value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    if value.startswith("["):
        try:
            parsed = json.loads(value.replace("'", '"'))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid frontmatter list: {value}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"frontmatter value must be a list: {value}")
        return parsed
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _metadata_name(value: Any, fallback: str) -> str:
    name = value if isinstance(value, str) and value.strip() else fallback
    normalized = name.strip().casefold()
    if not normalized or len(normalized) > 64:
        raise ValueError("skill name must contain between 1 and 64 characters")
    if not all(character.isalnum() or character in "._-" for character in normalized):
        raise ValueError("skill name may contain only letters, numbers, ., _, and -")
    return normalized


def _description(value: Any, body: str, name: str) -> str:
    if isinstance(value, str) and value.strip():
        description = " ".join(value.split())
    else:
        description = next(
            (" ".join(line.lstrip("# ").split()) for line in body.splitlines() if line.strip()),
            name,
        )
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = f"{description[: MAX_DESCRIPTION_CHARS - 1]}…"
    return description


def _metadata_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = [item.strip() for item in value if isinstance(item, str)]
    else:
        raise ValueError("skill metadata list must be a string or array")
    return [item for item in items if item]


def _metadata_bool(value: Any, *, default: bool) -> bool:
    if value in {None, ""}:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("skill enabled metadata must be true or false")


def _current_os() -> str:
    name = platform.system().casefold()
    if name == "darwin":
        return "macos"
    return name or os.name.casefold()
