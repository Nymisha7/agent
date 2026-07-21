from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import socket
import subprocess
import time
from typing import Any


TTL_MS = 5 * 60 * 1000
MAX_ENTRIES = 200
TRACK_KEYS = ("host", "ip", "version", "mode", "reason")
NODE_PRESENCE_RE = re.compile(
    r"Node:\s*([^ (]+)\s*\(([^)]+)\)\s*·\s*app\s*([^·]+?)\s*·\s*last input\s*([0-9]+)s ago\s*·\s*mode\s*([^·]+?)\s*·\s*reason\s*(.+)$",
    re.IGNORECASE,
)

_entries: dict[str, dict[str, Any]] = {}


def list_system_presence() -> list[dict[str, Any]]:
    _ensure_self_presence()
    now = _now_ms()
    for key, presence in tuple(_entries.items()):
        if now - int(presence.get("ts", 0)) > TTL_MS:
            del _entries[key]
    if len(_entries) > MAX_ENTRIES:
        sorted_entries = sorted(_entries.items(), key=lambda item: int(item[1].get("ts", 0)))
        for key, _presence in sorted_entries[:len(_entries) - MAX_ENTRIES]:
            del _entries[key]
    _touch_self_presence()
    return [dict(item) for item in sorted(_entries.values(), key=lambda item: int(item.get("ts", 0)), reverse=True)]


def reset_system_presence_for_tests() -> None:
    _entries.clear()


def update_system_presence(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_self_presence()
    parsed = _parse_presence(str(payload.get("text", "")).strip())
    key = (
        _normalize_presence_key(payload.get("deviceId"))
        or _normalize_presence_key(payload.get("instanceId"))
        or _normalize_presence_key(parsed.get("instanceId"))
        or _normalize_presence_key(parsed.get("host"))
        or _optional_text(parsed.get("ip"))
        or _optional_text(parsed.get("text"))[:64]
        or socket.gethostname().lower()
    )
    had_existing = key in _entries
    existing = _entries.get(key, {})
    merged = {
        **existing,
        **parsed,
        "host": _optional_text(payload.get("host")) or parsed.get("host") or existing.get("host"),
        "ip": _optional_text(payload.get("ip")) or parsed.get("ip") or existing.get("ip"),
        "version": _optional_text(payload.get("version")) or parsed.get("version") or existing.get("version"),
        "platform": _optional_text(payload.get("platform")) or existing.get("platform"),
        "deviceFamily": _optional_text(payload.get("deviceFamily")) or existing.get("deviceFamily"),
        "modelIdentifier": _optional_text(payload.get("modelIdentifier")) or existing.get("modelIdentifier"),
        "mode": _optional_text(payload.get("mode")) or parsed.get("mode") or existing.get("mode"),
        "lastInputSeconds": _optional_number(payload.get("lastInputSeconds"))
        if _optional_number(payload.get("lastInputSeconds")) is not None
        else parsed.get("lastInputSeconds", existing.get("lastInputSeconds")),
        "reason": _optional_text(payload.get("reason")) or parsed.get("reason") or existing.get("reason"),
        "deviceId": _optional_text(payload.get("deviceId")) or existing.get("deviceId"),
        "roles": _merge_string_list(existing.get("roles"), payload.get("roles")),
        "scopes": _merge_string_list(existing.get("scopes"), payload.get("scopes")),
        "instanceId": _optional_text(payload.get("instanceId")) or parsed.get("instanceId") or existing.get("instanceId"),
        "text": _optional_text(payload.get("text")) or parsed.get("text") or existing.get("text"),
        "ts": _now_ms(),
    }
    merged = {key: value for key, value in merged.items() if value is not None}
    _entries[key] = merged
    changes: dict[str, Any] = {}
    changed_keys: list[str] = []
    for item in TRACK_KEYS:
        previous = existing.get(item)
        next_value = merged.get(item)
        if previous != next_value:
            changes[item] = next_value
            changed_keys.append(item)
    return {
        "key": key,
        "previous": dict(existing) if had_existing else None,
        "next": dict(merged),
        "changes": changes,
        "changedKeys": changed_keys,
    }


def _ensure_self_presence() -> None:
    if not _entries:
        _init_self_presence()


def _touch_self_presence() -> None:
    host = socket.gethostname()
    key = host.lower()
    existing = _entries.get(key)
    if existing is None:
        _init_self_presence()
        return
    _entries[key] = {**existing, "ts": _now_ms()}


def _init_self_presence() -> None:
    host = socket.gethostname()
    ip = _resolve_primary_ipv4()
    version = _resolve_version()
    platform_label = _platform_label()
    device_family = _device_family()
    model_identifier = _model_identifier()
    text = f"Gateway: {host}{f' ({ip})' if ip else ''} · app {version} · mode gateway · reason self"
    presence = {
        "host": host,
        "ip": ip,
        "version": version,
        "platform": platform_label,
        "deviceFamily": device_family,
        "modelIdentifier": model_identifier,
        "mode": "gateway",
        "reason": "self",
        "text": text,
        "ts": _now_ms(),
    }
    _entries[host.lower()] = {key: value for key, value in presence.items() if value is not None}


def _resolve_primary_ipv4() -> str | None:
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        return None
    return ip if ip else None


def _resolve_version() -> str:
    explicit = os.environ.get("AGENT_VERSION")
    if explicit:
        return explicit
    try:
        return importlib.metadata.version("agent")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _platform_label() -> str:
    system = platform.system().lower() or os.name
    release = platform.release()
    if system == "darwin":
        return f"macos {platform.mac_ver()[0] or release}"
    if system == "windows":
        return f"windows {release}"
    return f"{system} {release}"


def _device_family() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "Mac"
    if system == "windows":
        return "Windows"
    if system == "linux":
        return "Linux"
    return system or os.name


def _model_identifier() -> str | None:
    if platform.system().lower() == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.model"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                timeout=2,
            )
        except OSError:
            return None
        output = result.stdout.strip() if isinstance(result.stdout, str) else ""
        return output or None
    return platform.machine() or None


def _parse_presence(text: str) -> dict[str, Any]:
    trimmed = text.strip()
    match = NODE_PRESENCE_RE.match(trimmed)
    if match is None:
        return {"text": trimmed, "ts": _now_ms()}
    host, ip, version, last_input, mode, reason = match.groups()
    return {
        "host": host.strip(),
        "ip": ip.strip(),
        "version": version.strip(),
        "lastInputSeconds": int(last_input),
        "mode": mode.strip(),
        "reason": reason.strip(),
        "text": trimmed,
        "ts": _now_ms(),
    }


def _normalize_presence_key(value: Any) -> str | None:
    text = _optional_text(value)
    return text.lower() if text else None


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _optional_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _merge_string_list(*values: Any) -> list[str] | None:
    out: dict[str, None] = {}
    for value in values:
        if not isinstance(value, list):
            continue
        for item in value:
            text = str(item).strip()
            if text:
                out[text] = None
    return list(out) if out else None


def _now_ms() -> int:
    return int(time.time() * 1000)
