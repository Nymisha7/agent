from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SESSION_SCOPES = frozenset({"per-sender", "shared", "global"})
PEER_KINDS = frozenset({"direct", "group", "channel"})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AgentProfileConfig:
    id: str
    skills: tuple[str, ...] | None = None
    tools: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RouteBindingConfig:
    agent_id: str
    channel: str
    scope: str | None = None
    account_id: str | None = None
    peer_kind: str | None = None
    peer_id: str | None = None
    guild_id: str | None = None
    team_id: str | None = None


@dataclass(frozen=True)
class SkillsConfig:
    extra_dirs: tuple[Path, ...] = ()
    max_loaded: int = 32
    max_instruction_chars: int = 24_000


@dataclass(frozen=True)
class SessionConfig:
    default_scope: str = "per-sender"


@dataclass(frozen=True)
class AgentConfig:
    default_agent_id: str = "main"
    default_skills: tuple[str, ...] | None = None
    default_tools: tuple[str, ...] | None = None
    agents: Mapping[str, AgentProfileConfig] = field(
        default_factory=lambda: {"main": AgentProfileConfig(id="main")}
    )
    bindings: tuple[RouteBindingConfig, ...] = ()
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    source_paths: tuple[Path, ...] = ()

    def agent(self, agent_id: str | None = None) -> AgentProfileConfig:
        selected = agent_id or self.default_agent_id
        try:
            return self.agents[selected]
        except KeyError as exc:
            raise ConfigError(f"Unknown agent profile: {selected}") from exc

    def skill_allowlist(self, agent_id: str | None = None) -> tuple[str, ...] | None:
        agent = self.agent(agent_id)
        return self.default_skills if agent.skills is None else agent.skills

    def tool_allowlist(self, agent_id: str | None = None) -> tuple[str, ...] | None:
        agent = self.agent(agent_id)
        return self.default_tools if agent.tools is None else agent.tools


def load_agent_config(
    workspace_root: Path,
    *,
    explicit_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> AgentConfig:
    workspace_root = workspace_root.expanduser().resolve()
    environ = environ if environ is not None else os.environ
    home = (home or Path.home()).expanduser()
    env_path = environ.get("AGENT_CONFIG")
    if explicit_path is not None or env_path:
        path = (explicit_path or Path(str(env_path))).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"Agent config file not found: {path}")
        return parse_agent_config(_read_json_object(path), workspace_root, source_paths=(path,))

    xdg_config = Path(environ.get("XDG_CONFIG_HOME", home / ".config")).expanduser()
    candidates = (
        xdg_config / "agent" / "config.json",
        workspace_root / ".agent" / "config.json",
    )
    merged: dict[str, Any] = {}
    loaded: list[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        merged = _deep_merge(merged, _read_json_object(path))
        loaded.append(path.resolve())
    return parse_agent_config(merged, workspace_root, source_paths=tuple(loaded))


def parse_agent_config(
    value: Mapping[str, Any],
    workspace_root: Path,
    *,
    source_paths: tuple[Path, ...] = (),
) -> AgentConfig:
    root = _object(value, "config")
    _reject_unknown(root, {"agents", "bindings", "session", "skills"}, "config")

    agents_raw = _object(root.get("agents", {}), "agents")
    _reject_unknown(agents_raw, {"default", "defaults", "list"}, "agents")
    default_agent_id = _identifier(agents_raw.get("default", "main"), "agents.default")
    defaults_raw = _object(agents_raw.get("defaults", {}), "agents.defaults")
    _reject_unknown(defaults_raw, {"skills", "tools"}, "agents.defaults")
    default_skills = _optional_identifier_list(
        defaults_raw.get("skills"),
        "agents.defaults.skills",
    )
    default_tools = _optional_identifier_list(
        defaults_raw.get("tools"),
        "agents.defaults.tools",
    )

    agents: dict[str, AgentProfileConfig] = {}
    list_raw = agents_raw.get("list", [{"id": default_agent_id}])
    if not isinstance(list_raw, list):
        raise ConfigError("agents.list must be an array.")
    for index, item in enumerate(list_raw):
        entry = _object(item, f"agents.list[{index}]")
        _reject_unknown(entry, {"id", "skills", "tools"}, f"agents.list[{index}]")
        agent_id = _identifier(entry.get("id"), f"agents.list[{index}].id")
        if agent_id in agents:
            raise ConfigError(f"Duplicate agent profile: {agent_id}")
        skills = (
            _optional_identifier_list(entry.get("skills"), f"agents.list[{index}].skills")
            if "skills" in entry
            else None
        )
        tools = (
            _optional_identifier_list(entry.get("tools"), f"agents.list[{index}].tools")
            if "tools" in entry
            else None
        )
        agents[agent_id] = AgentProfileConfig(id=agent_id, skills=skills, tools=tools)
    if not agents:
        agents[default_agent_id] = AgentProfileConfig(id=default_agent_id)
    if default_agent_id not in agents:
        raise ConfigError(
            f"agents.default names '{default_agent_id}', but that profile is not in agents.list."
        )

    session_raw = _object(root.get("session", {}), "session")
    _reject_unknown(session_raw, {"default_scope"}, "session")
    default_scope = _scope(session_raw.get("default_scope", "per-sender"), "session.default_scope")

    skills_raw = _object(root.get("skills", {}), "skills")
    _reject_unknown(
        skills_raw,
        {"extra_dirs", "max_loaded", "max_instruction_chars"},
        "skills",
    )
    extra_dirs_raw = skills_raw.get("extra_dirs", [])
    if not isinstance(extra_dirs_raw, list):
        raise ConfigError("skills.extra_dirs must be an array of paths.")
    extra_dirs: list[Path] = []
    for index, raw_path in enumerate(extra_dirs_raw):
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ConfigError(f"skills.extra_dirs[{index}] must be a non-empty path.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = workspace_root / path
        extra_dirs.append(path.resolve())
    max_loaded = _bounded_int(skills_raw.get("max_loaded", 32), "skills.max_loaded", 0, 256)
    max_chars = _bounded_int(
        skills_raw.get("max_instruction_chars", 24_000),
        "skills.max_instruction_chars",
        1_000,
        250_000,
    )

    bindings_raw = root.get("bindings", [])
    if not isinstance(bindings_raw, list):
        raise ConfigError("bindings must be an array.")
    bindings: list[RouteBindingConfig] = []
    for index, item in enumerate(bindings_raw):
        entry = _object(item, f"bindings[{index}]")
        _reject_unknown(entry, {"agent", "scope", "match"}, f"bindings[{index}]")
        agent_id = _identifier(entry.get("agent"), f"bindings[{index}].agent")
        if agent_id not in agents:
            raise ConfigError(f"bindings[{index}] references unknown agent '{agent_id}'.")
        match = _object(entry.get("match"), f"bindings[{index}].match")
        _reject_unknown(
            match,
            {"channel", "account_id", "peer", "guild_id", "team_id"},
            f"bindings[{index}].match",
        )
        channel = _identifier(match.get("channel"), f"bindings[{index}].match.channel")
        account_id = _optional_text(match.get("account_id"), f"bindings[{index}].match.account_id")
        guild_id = _optional_text(match.get("guild_id"), f"bindings[{index}].match.guild_id")
        team_id = _optional_text(match.get("team_id"), f"bindings[{index}].match.team_id")
        peer_kind: str | None = None
        peer_id: str | None = None
        if "peer" in match:
            peer = _object(match["peer"], f"bindings[{index}].match.peer")
            _reject_unknown(peer, {"kind", "id"}, f"bindings[{index}].match.peer")
            peer_kind = _peer_kind(peer.get("kind"), f"bindings[{index}].match.peer.kind")
            peer_id = _required_text(peer.get("id"), f"bindings[{index}].match.peer.id")
        bindings.append(RouteBindingConfig(
            agent_id=agent_id,
            channel=channel,
            scope=_scope(entry["scope"], f"bindings[{index}].scope") if "scope" in entry else None,
            account_id=account_id,
            peer_kind=peer_kind,
            peer_id=peer_id,
            guild_id=guild_id,
            team_id=team_id,
        ))

    return AgentConfig(
        default_agent_id=default_agent_id,
        default_skills=default_skills,
        default_tools=default_tools,
        agents=agents,
        bindings=tuple(bindings),
        skills=SkillsConfig(
            extra_dirs=tuple(extra_dirs),
            max_loaded=max_loaded,
            max_instruction_chars=max_chars,
        ),
        session=SessionConfig(default_scope=default_scope),
        source_paths=source_paths,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read Agent config {path}: {exc}") from exc
    return _object(value, str(path))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} must be an object.")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {field_name} field(s): {', '.join(unknown)}")


def _identifier(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if not _ID_RE.fullmatch(text):
        raise ConfigError(
            f"{field_name} must start with a letter or number and contain only letters, numbers, ., _, or -."
        )
    return text


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _scope(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name).casefold()
    if text not in SESSION_SCOPES:
        raise ConfigError(f"{field_name} must be one of: {', '.join(sorted(SESSION_SCOPES))}.")
    return text


def _peer_kind(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name).casefold()
    if text not in PEER_KINDS:
        raise ConfigError(f"{field_name} must be one of: {', '.join(sorted(PEER_KINDS))}.")
    return text


def _optional_identifier_list(value: Any, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ConfigError(f"{field_name} must be an array or null.")
    result: list[str] = []
    for index, item in enumerate(value):
        name = _identifier(item, f"{field_name}[{index}]")
        if name not in result:
            result.append(name)
    return tuple(result)


def _bounded_int(value: Any, field_name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field_name} must be an integer.")
    if not minimum <= value <= maximum:
        raise ConfigError(f"{field_name} must be between {minimum} and {maximum}.")
    return value
