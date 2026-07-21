from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .language_servers import language_server_context_text
from .project_identity import identity_text
from .session_store import SessionInfo, SessionStore


MAX_CONTEXT_CHARS = 4_000


@dataclass(frozen=True)
class StoredContext:
    text: str
    message_count: int
    event_count: int


def build_stored_context(
    *,
    store: SessionStore,
    session: SessionInfo,
) -> StoredContext:
    messages = store.list_messages(session.id, limit=None)
    events = store.list_events(session.id, limit=8)

    if not messages and not events and not session.last_prompt:
        return StoredContext(text="", message_count=0, event_count=0)

    sections: list[str] = [
        "Resumed session:",
        f"  id: {session.id}",
        f"  workspace: {session.workspace_root}",
        f"  model: {session.model or 'unknown'}",
    ]

    workspace_identity = identity_text(Path(session.workspace_root))
    if workspace_identity:
        sections.extend(["", workspace_identity])

    server_context = language_server_context_text()
    if server_context:
        sections.extend(["", server_context])

    if session.last_prompt:
        sections.append(f"  last prompt: {_truncate(session.last_prompt, 200)}")

    if events:
        sections.append("  recent events:")
        for item in reversed(events):
            tool = f" [{item.tool}]" if item.tool else ""
            sections.append(f"    - {item.event_type}{tool}: {_truncate(item.summary, 160)}")

    text = _truncate_block("\n".join(sections), MAX_CONTEXT_CHARS)
    return StoredContext(
        text=text,
        message_count=len(messages),
        event_count=len(events),
    )


def _truncate(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def _truncate_block(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:max(0, limit - 14)]}\n...[truncated]"
