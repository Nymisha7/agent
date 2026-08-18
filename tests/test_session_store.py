from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from agent.main import _selected_model, _workspace_root_key, create_new_session
from agent.session_store import CostUsage, SessionStore, TokenUsage
from agent.sqlx_session_store import _resolve_rust_binary


class SessionStoreTests(unittest.TestCase):
    def test_token_cost_breakdown_round_trips_and_accumulates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(
                workspace_root=root,
                provider="openai",
                model="gpt-5.4-mini",
            )
            first = CostUsage(
                input=0.0006,
                cached_input=0.000015,
                output=0.00045,
            )
            second = CostUsage(input=0.0001, output=0.0002)

            store.add_usage(
                session.id,
                tokens=TokenUsage(input=1_000, output=100, cache_read=200),
                cost_usd=first.total,
                costs=first,
            )
            store.add_usage(
                session.id,
                tokens=TokenUsage(input=100, output=20),
                cost_usd=second.total,
                costs=second,
            )

            restored = store.get_session(session.id)

        self.assertAlmostEqual(restored.cost_usd, first.total + second.total)
        self.assertAlmostEqual(restored.costs.input, 0.0007)
        self.assertAlmostEqual(restored.costs.cached_input, 0.000015)
        self.assertAlmostEqual(restored.costs.output, 0.00065)
        self.assertEqual(restored.tokens.input, 1_100)
        self.assertEqual(restored.tokens.output, 120)

    def test_existing_session_database_adds_cost_breakdown_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "sessions.sqlite3"
            with sqlite3.connect(database) as conn:
                conn.execute(
                    """create table sessions (
                        id text primary key, project_id text, workspace_id text,
                        title text not null, workspace_root text not null, cwd text,
                        created_at text not null, updated_at text not null,
                        provider text, model text, agent text, permission_json text,
                        cost_usd real not null default 0,
                        tokens_input integer not null default 0,
                        tokens_output integer not null default 0,
                        tokens_reasoning integer not null default 0,
                        tokens_cache_read integer not null default 0,
                        tokens_cache_write integer not null default 0,
                        summary text, active_root text, focus_path text,
                        last_prompt text, state_json text
                    )"""
                )
                conn.execute(
                    """insert into sessions (
                        id, title, workspace_root, created_at, updated_at, cost_usd
                    ) values ('legacy', 'Legacy', ?, 'now', 'now', 0.25)""",
                    (str(root),),
                )

            store = SessionStore(database)
            restored = store.get_session("legacy")
            with store._connect() as conn:
                columns = {
                    row[1] for row in conn.execute("pragma table_info(sessions)")
                }

        self.assertEqual(restored.cost_usd, 0.25)
        self.assertEqual(restored.costs, CostUsage())
        self.assertTrue({
            "cost_input_usd",
            "cost_cached_input_usd",
            "cost_cache_write_usd",
            "cost_output_usd",
        }.issubset(columns))

    def test_session_database_and_created_directory_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "private-data"
            database = data_dir / "sessions.sqlite3"

            store = SessionStore(database)
            store.create_session(workspace_root=Path(tmp))

            self.assertEqual(data_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(database.stat().st_mode & 0o777, 0o600)
            for sidecar in data_dir.glob("sessions.sqlite3-*"):
                self.assertEqual(sidecar.stat().st_mode & 0o777, 0o600)

    def test_rust_store_resolver_prefers_release_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            module_path = repo_root / "agent" / "sqlx_session_store.py"
            release = repo_root / "agent-rust" / "target" / "release" / "agent-rust"
            debug = repo_root / "agent-rust" / "target" / "debug" / "agent-rust"
            release.parent.mkdir(parents=True)
            debug.parent.mkdir(parents=True)
            release.write_text("release", encoding="utf-8")
            debug.write_text("debug", encoding="utf-8")

            with patch("agent.sqlx_session_store.__file__", str(module_path)), patch(
                "agent.sqlx_session_store.bundled_rust_binary", return_value=None
            ):
                resolved = _resolve_rust_binary(root / "data")

        self.assertEqual(resolved, release.resolve())

    def test_message_attachment_round_trips_through_session_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stored = root / "attachment.txt"
            stored.write_text("document body", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(workspace_root=root)

            message = store.add_message_with_attachments(
                session.id,
                "user",
                "summarize this",
                [{
                    "id": "attachment-1",
                    "filename": "report.txt",
                    "mime": "text/plain",
                    "size_bytes": stored.stat().st_size,
                    "sha256": "a" * 64,
                    "storage_path": str(stored),
                    "source": "user_file",
                }],
            )

            restored = store.list_messages(session.id, limit=None)[0]

        self.assertEqual(message.attachments[0].filename, "report.txt")
        self.assertEqual(restored.attachments[0].storage_path, str(stored))

    def test_session_store_write_lock_timeout_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3", write_lock_timeout_ms=1234)

            with store._connect() as conn:
                timeout = conn.execute("pragma busy_timeout").fetchone()[0]

            self.assertEqual(timeout, 1234)

    def test_add_message_waits_on_sqlite_write_lock_before_sequence_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3", write_lock_timeout_ms=25)
            session = store.create_session(workspace_root=root)

            blocker = store._connect()
            try:
                blocker.execute("begin immediate")
                with self.assertRaises(sqlite3.OperationalError):
                    store.add_message(session.id, "user", "blocked")
            finally:
                blocker.rollback()
                blocker.close()

            self.assertEqual(store.list_messages(session.id, limit=None), [])

    def test_add_event_waits_on_sqlite_write_lock_before_sequence_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3", write_lock_timeout_ms=25)
            session = store.create_session(workspace_root=root)

            blocker = store._connect()
            try:
                blocker.execute("begin immediate")
                with self.assertRaises(sqlite3.OperationalError):
                    store.add_event(
                        session.id,
                        event_type="turn_started",
                        summary="blocked",
                    )
            finally:
                blocker.rollback()
                blocker.close()

            self.assertEqual(store.list_events(session.id), [])

    def test_add_messages_appends_batch_with_contiguous_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(workspace_root=root)

            first = store.add_message(session.id, "system", "before")
            batch = store.add_messages(
                session.id,
                [
                    ("user", "question"),
                    ("assistant", "answer"),
                ],
            )

            self.assertEqual(first.seq, 1)
            self.assertEqual(
                [(message.seq, message.role, message.content) for message in batch],
                [(2, "user", "question"), (3, "assistant", "answer")],
            )
            self.assertEqual(batch[0].created_at, batch[1].created_at)
            self.assertEqual(
                [
                    (message.seq, message.role, message.content)
                    for message in store.list_messages(session.id, limit=None)
                ],
                [
                    (1, "system", "before"),
                    (2, "user", "question"),
                    (3, "assistant", "answer"),
                ],
            )

    def test_add_messages_can_touch_last_prompt_in_same_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(workspace_root=root)

            store.add_messages(
                session.id,
                [
                    ("user", "/help"),
                    ("assistant", "Commands"),
                ],
                last_prompt="/help",
            )

            updated = store.get_session(session.id)
            self.assertEqual(updated.last_prompt, "/help")
            self.assertEqual(updated.title, "/help")
            self.assertEqual(
                [
                    (message.role, message.content)
                    for message in store.list_messages(session.id, limit=None)
                ],
                [("user", "/help"), ("assistant", "Commands")],
            )

    def test_add_messages_accepts_current_route_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            session, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
            )

            store.add_messages(
                session.id,
                [("user", "question")],
                last_prompt="question",
                expected_route_key=route_key,
            )

            self.assertEqual(
                [(message.role, message.content) for message in store.list_messages(session.id, limit=None)],
                [("user", "question")],
            )
            self.assertEqual(store.get_session(session.id).last_prompt, "question")

    def test_add_messages_rejects_rebounded_route_before_writing_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            original, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
            )
            replacement = store.create_session(workspace_root=root, title="replacement")
            with store._connect() as conn:
                conn.execute(
                    "update session_routes set session_id = ? where route_key = ?",
                    (replacement.id, route_key),
                )

            with self.assertRaisesRegex(RuntimeError, "Session route rebound"):
                store.add_messages(
                    original.id,
                    [
                        ("user", "stale question"),
                        ("assistant", "stale answer"),
                    ],
                    last_prompt="stale question",
                    expected_route_key=route_key,
                )

            self.assertEqual(store.list_messages(original.id, limit=None), [])
            self.assertIsNone(store.get_session(original.id).last_prompt)

    def test_add_messages_rejects_invalid_role_before_writing_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(workspace_root=root)

            with self.assertRaisesRegex(ValueError, "Unsupported message role"):
                store.add_messages(
                    session.id,
                    [
                        ("user", "question"),
                        ("invalid", "bad"),
                    ],
                )

            self.assertEqual(store.list_messages(session.id, limit=None), [])

    def test_session_persists_provider_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")

            session = store.create_session(
                workspace_root=root,
                provider="ollama",
                model="llama3.1",
            )

            self.assertEqual(session.provider, "ollama")
            self.assertEqual(session.model, "llama3.1")
            self.assertEqual(store.get_session(session.id).provider, "ollama")

    def test_update_llm_config_persists_provider_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(workspace_root=root)

            store.update_llm_config(
                session.id,
                provider="deepseek",
                model="deepseek-chat",
            )

            updated = store.get_session(session.id)
            self.assertEqual(updated.provider, "deepseek")
            self.assertEqual(updated.model, "deepseek-chat")

    def test_patch_session_metadata_updates_supported_fields_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(
                workspace_root=root,
                provider="openai",
                model="gpt-5.5-mini",
            )

            updated = store.patch_session_metadata(
                session.id,
                title="RA project",
                provider="anthropic",
                model="claude-3-5-sonnet-latest",
                state_patch={
                    "reasoning_effort": "high",
                    "active_root": str(root),
                },
            )

            self.assertEqual(updated.title, "RA project")
            self.assertEqual(updated.provider, "anthropic")
            self.assertEqual(updated.model, "claude-3-5-sonnet-latest")
            self.assertEqual(updated.state, {
                "reasoning_effort": "high",
                "active_root": str(root),
            })
            self.assertEqual(updated.active_root, str(root))

    def test_reset_routed_session_rebinds_route_and_preserves_old_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            session, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
                provider="ollama",
                model="llama3.1",
                title="RA project",
            )
            store.add_message(session.id, "user", "old question")
            store.patch_session_metadata(
                session.id,
                state_patch={
                    "reasoning_effort": "medium",
                    "active_root": str(root / "old"),
                },
            )

            old_session, new_session = store.reset_routed_session(route_key)

            self.assertEqual(old_session.id, session.id)
            self.assertNotEqual(new_session.id, session.id)
            self.assertEqual(store.get_route(route_key).session_id, new_session.id)
            self.assertEqual(
                [(message.role, message.content) for message in store.list_messages(old_session.id, limit=None)],
                [("user", "old question")],
            )
            self.assertEqual(store.list_messages(new_session.id, limit=None), [])
            self.assertEqual(new_session.title, "RA project")
            self.assertEqual(new_session.provider, "ollama")
            self.assertEqual(new_session.model, "llama3.1")
            self.assertEqual(new_session.state, {"reasoning_effort": "medium"})
            self.assertIsNone(new_session.active_root)

    def test_delete_routed_session_cascades_transcript_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            session, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
            )
            store.add_messages(
                session.id,
                [
                    ("user", "question"),
                    ("assistant", "answer"),
                ],
            )
            store.add_event(
                session.id,
                event_type="tool_call",
                summary="inspected",
            )

            deleted = store.delete_routed_session(route_key)

            self.assertEqual(deleted["session_id"], session.id)
            self.assertEqual(deleted["messages_deleted"], 2)
            self.assertEqual(deleted["events_deleted"], 1)
            self.assertEqual(deleted["routes_deleted"], 1)
            with self.assertRaises(KeyError):
                store.get_route(route_key)
            with self.assertRaises(KeyError):
                store.get_session(session.id)
            self.assertEqual(store.list_messages(session.id, limit=None), [])
            self.assertEqual(store.list_events(session.id), [])

    def test_compact_routed_session_keeps_tail_and_archives_pruned_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            session, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
            )
            for index in range(5):
                role = "user" if index % 2 == 0 else "assistant"
                store.add_message(session.id, role, f"message {index}")

            compacted = store.compact_routed_session(route_key, max_messages=2)
            messages = store.list_messages(session.id, limit=None)
            events = store.list_events(session.id, limit=10)
            archive_event = next(event for event in events if event.event_type == "session_compacted")

            self.assertTrue(compacted["compacted"])
            self.assertEqual(compacted["lines_before"], 5)
            self.assertEqual(compacted["lines_after"], 2)
            self.assertEqual(compacted["kept"], 2)
            self.assertEqual(compacted["pruned"], 3)
            self.assertEqual(compacted["archived_event_id"], archive_event.id)
            self.assertEqual(
                [(message.seq, message.content) for message in messages],
                [(4, "message 3"), (5, "message 4")],
            )
            self.assertEqual(
                [(item["seq"], item["content"]) for item in archive_event.data["messages"]],
                [(1, "message 0"), (2, "message 1"), (3, "message 2")],
            )

            checkpoints = store.list_session_compaction_checkpoints(
                session.id,
                session_key=route_key,
            )
            self.assertEqual(store.count_session_compaction_checkpoints(session.id), 1)
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(checkpoints[0]["checkpointId"], f"sqlite:event:{archive_event.id}")
            self.assertEqual(checkpoints[0]["sessionKey"], route_key)
            self.assertEqual(checkpoints[0]["sessionId"], session.id)
            self.assertEqual(checkpoints[0]["reason"], "manual")
            self.assertEqual(checkpoints[0]["linesBefore"], 5)
            self.assertEqual(checkpoints[0]["linesAfter"], 2)
            self.assertEqual(checkpoints[0]["maxMessages"], 2)
            self.assertEqual(checkpoints[0]["firstKeptEntryId"], "4")
            self.assertEqual(checkpoints[0]["preCompaction"]["entryId"], "3")
            self.assertEqual(checkpoints[0]["postCompaction"]["entryId"], "4")
            checkpoint = store.get_session_compaction_checkpoint(
                session.id,
                session_key=route_key,
                checkpoint_id=checkpoints[0]["checkpointId"],
            )
            self.assertEqual(checkpoint, checkpoints[0])
            with self.assertRaises(KeyError):
                store.get_session_compaction_checkpoint(
                    session.id,
                    session_key=route_key,
                    checkpoint_id="sqlite:event:999999",
                )

    def test_compact_routed_session_noops_when_within_max_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            session, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
            )
            store.add_message(session.id, "user", "one")
            store.add_message(session.id, "assistant", "two")

            compacted = store.compact_routed_session(route_key, max_messages=5)

            self.assertFalse(compacted["compacted"])
            self.assertEqual(compacted["lines_before"], 2)
            self.assertEqual(compacted["lines_after"], 2)
            self.assertIsNone(compacted["archived_event_id"])
            self.assertEqual(
                [(message.seq, message.content) for message in store.list_messages(session.id, limit=None)],
                [(1, "one"), (2, "two")],
            )
            self.assertFalse(any(event.event_type == "session_compacted" for event in store.list_events(session.id)))

    def test_branch_routed_session_from_compaction_checkpoint_reconstructs_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            session, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
                provider="ollama",
                model="llama3.1",
            )
            store.patch_session_metadata(
                session.id,
                state_patch={"reasoning_effort": "medium"},
            )
            for index in range(5):
                role = "user" if index % 2 == 0 else "assistant"
                store.add_message(session.id, role, f"message {index}")
            compacted = store.compact_routed_session(route_key, max_messages=2)
            checkpoint_id = f"sqlite:event:{compacted['archived_event_id']}"
            store.add_message(session.id, "assistant", "after")

            branched = store.branch_routed_session_from_compaction_checkpoint(
                route_key,
                checkpoint_id=checkpoint_id,
            )
            branch_session = store.get_session(branched["session_id"])
            branch_messages = store.list_messages(branch_session.id, limit=None)

            self.assertEqual(branched["source_session_id"], session.id)
            self.assertNotEqual(branched["session_id"], session.id)
            self.assertEqual(store.get_route(branched["key"]).session_id, branched["session_id"])
            self.assertEqual(branch_session.provider, "ollama")
            self.assertEqual(branch_session.model, "llama3.1")
            self.assertEqual(branch_session.state["parent_session_key"], route_key)
            self.assertEqual(branch_session.state["source_session_id"], session.id)
            self.assertEqual(branch_session.state["checkpoint_id"], checkpoint_id)
            self.assertEqual(branch_session.state["reasoning_effort"], "medium")
            self.assertEqual(
                [(message.seq, message.role, message.content) for message in branch_messages],
                [
                    (1, "user", "message 0"),
                    (2, "assistant", "message 1"),
                    (3, "user", "message 2"),
                    (4, "assistant", "message 3"),
                    (5, "user", "message 4"),
                ],
            )

    def test_restore_routed_session_from_compaction_checkpoint_rebinds_same_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            route_key = "agent:main:tui:alice"
            session, _created = store.get_or_create_routed_session(
                route_key=route_key,
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
                provider="ollama",
                model="llama3.1",
            )
            store.patch_session_metadata(
                session.id,
                state_patch={"reasoning_effort": "medium"},
            )
            for index in range(5):
                role = "user" if index % 2 == 0 else "assistant"
                store.add_message(session.id, role, f"message {index}")
            compacted = store.compact_routed_session(route_key, max_messages=2)
            checkpoint_id = f"sqlite:event:{compacted['archived_event_id']}"
            store.add_message(session.id, "assistant", "after")

            restored = store.restore_routed_session_from_compaction_checkpoint(
                route_key,
                checkpoint_id=checkpoint_id,
            )
            restored_session = store.get_session(restored["session_id"])
            restored_messages = store.list_messages(restored_session.id, limit=None)
            checkpoint = store.get_session_compaction_checkpoint(
                restored_session.id,
                session_key=route_key,
                checkpoint_id=checkpoint_id,
            )

            self.assertEqual(restored["previous_session_id"], session.id)
            self.assertNotEqual(restored["session_id"], session.id)
            self.assertEqual(store.get_route(route_key).session_id, restored["session_id"])
            self.assertEqual(restored_session.provider, "ollama")
            self.assertEqual(restored_session.model, "llama3.1")
            self.assertEqual(restored_session.state["restored_from_session_id"], session.id)
            self.assertEqual(restored_session.state["restored_checkpoint_id"], checkpoint_id)
            self.assertEqual(restored_session.state["reasoning_effort"], "medium")
            self.assertEqual(checkpoint["checkpointId"], checkpoint_id)
            self.assertEqual(checkpoint["sessionId"], restored["session_id"])
            self.assertEqual(
                [(message.seq, message.role, message.content) for message in restored_messages],
                [
                    (1, "user", "message 0"),
                    (2, "assistant", "message 1"),
                    (3, "user", "message 2"),
                    (4, "assistant", "message 3"),
                    (5, "user", "message 4"),
                ],
            )

    def test_list_messages_limit_returns_newest_tail_in_chronological_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(workspace_root=root)
            for index in range(5):
                store.add_message(session.id, "user", f"message {index}")

            messages = store.list_messages(session.id, limit=3)

            self.assertEqual(
                [(message.seq, message.content) for message in messages],
                [(3, "message 2"), (4, "message 3"), (5, "message 4")],
            )

    def test_list_sessions_supports_agent_active_and_unbounded_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            old = store.create_session(workspace_root=root, title="old", agent_id="main")
            current = store.create_session(workspace_root=root, title="current", agent_id="main")
            other = store.create_session(workspace_root=root, title="other", agent_id="work")
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
            stale = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
            with store._connect() as conn:
                conn.execute(
                    "update sessions set updated_at = ? where id = ?",
                    (stale, old.id),
                )

            all_sessions = store.list_sessions(limit=None)
            active_main = store.list_sessions(
                limit=None,
                agent_id="main",
                updated_after=cutoff,
            )

            self.assertEqual({session.id for session in all_sessions}, {old.id, current.id, other.id})
            self.assertEqual([session.id for session in active_main], [current.id])
            self.assertEqual(store.count_sessions(), 3)
            self.assertEqual(store.count_sessions(agent_id="main"), 2)
            self.assertEqual(store.count_sessions(agent_id="main", updated_after=cutoff), 1)

    def test_default_falls_back_to_workspace_db_when_default_db_is_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg_home = Path(tmp) / "xdg"
            preferred = xdg_home / "agent" / "sessions.sqlite3"
            fallback = Path.cwd() / ".agent-session.sqlite3"

            def fake_init(store: SessionStore, db_path: Path) -> None:
                store.db_path = Path(db_path)
                if Path(db_path) == preferred:
                    raise sqlite3.OperationalError("attempt to write a readonly database")

            with (
                patch.dict("os.environ", {"XDG_DATA_HOME": str(xdg_home)}, clear=True),
                patch.object(SessionStore, "__init__", fake_init),
            ):
                store = SessionStore.default()

            self.assertEqual(store.db_path, fallback)

    def test_default_respects_explicit_session_db_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            explicit = Path(tmp) / "explicit.sqlite3"

            def fake_init(store: SessionStore, db_path: Path) -> None:
                store.db_path = Path(db_path)
                raise sqlite3.OperationalError("attempt to write a readonly database")

            with (
                patch.dict("os.environ", {"AGENT_SESSION_DB": str(explicit)}, clear=True),
                patch.object(SessionStore, "__init__", fake_init),
                self.assertRaises(sqlite3.OperationalError),
            ):
                SessionStore.default()

    def test_session_maintenance_prunes_stale_unrouted_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            active = store.create_session(workspace_root=root, title="active")
            stale = store.create_session(workspace_root=root, title="stale")

            old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
            with store._connect() as conn:
                conn.execute(
                    "update sessions set updated_at = ? where id in (?, ?)",
                    (old, active.id, stale.id),
                )

            report = store.apply_maintenance(
                prune_after_days=30,
                active_session_id=active.id,
                force=True,
            )

            self.assertEqual(report.pruned, 1)
            self.assertEqual(store.get_session(active.id).id, active.id)
            with self.assertRaises(KeyError):
                store.get_session(stale.id)

    def test_session_maintenance_caps_oldest_unrouted_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            sessions = [
                store.create_session(workspace_root=root, title=f"session {index}")
                for index in range(4)
            ]

            base = datetime.now(timezone.utc) - timedelta(days=1)
            with store._connect() as conn:
                for index, session in enumerate(sessions):
                    updated = (base + timedelta(minutes=index)).isoformat()
                    conn.execute(
                        "update sessions set updated_at = ? where id = ?",
                        (updated, session.id),
                    )

            report = store.apply_maintenance(
                max_entries=2,
                prune_after_days=365,
                force=True,
            )

            remaining = {session.id for session in store.list_sessions(limit=10)}
            self.assertEqual(report.capped, 2)
            self.assertEqual(remaining, {sessions[2].id, sessions[3].id})

    def test_session_maintenance_preserves_routed_sessions_when_capping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            routed, _created = store.get_or_create_routed_session(
                route_key="agent:main:tui:alice",
                workspace_root=root,
                agent_id="main",
                scope="per-sender",
                channel="tui",
                account_id="default",
                sender_id="alice",
                title="routed",
            )
            extra = [
                store.create_session(workspace_root=root, title=f"extra {index}")
                for index in range(2)
            ]

            old = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            with store._connect() as conn:
                conn.execute(
                    "update sessions set updated_at = ? where id = ?",
                    (old, routed.id),
                )

            report = store.apply_maintenance(
                max_entries=1,
                prune_after_days=365,
                force=True,
            )

            remaining = {session.id for session in store.list_sessions(limit=10)}
            self.assertEqual(report.capped, len(extra))
            self.assertEqual(remaining, {routed.id})

    def test_session_maintenance_warn_mode_does_not_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            session = store.create_session(workspace_root=root, title="stale")
            old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
            with store._connect() as conn:
                conn.execute(
                    "update sessions set updated_at = ? where id = ?",
                    (old, session.id),
                )

            report = store.apply_maintenance(
                prune_after_days=30,
                force=True,
                mode="warn",
            )

            self.assertEqual(report.mode, "warn")
            self.assertEqual(report.pruned, 1)
            self.assertEqual(store.get_session(session.id).id, session.id)

    def test_session_maintenance_waits_for_high_water_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            store.create_session(workspace_root=root)
            store.create_session(workspace_root=root)

            report = store.apply_maintenance(max_entries=500)

            self.assertFalse(report.applied)
            self.assertEqual(report.before_count, 2)
            self.assertEqual(report.after_count, 2)


class ModelSelectionTests(unittest.TestCase):
    def test_workspace_root_key_treats_wsl_drive_mount_as_same_windows_project(self) -> None:
        self.assertEqual(
            _workspace_root_key("/mnt/d/Codex/nym"),
            _workspace_root_key("D:/Codex/nym"),
        )

    def test_new_session_inherits_last_workspace_model_when_unspecified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            store.create_session(
                workspace_root=root,
                provider="ollama",
                model="llama3.1",
            )
            args = SimpleNamespace(
                root=str(root),
                channel=None,
                provider=None,
                model=None,
            )

            session = create_new_session(args, store)

        self.assertEqual(session.provider, "ollama")
        self.assertEqual(session.model, "llama3.1")

    def test_new_session_does_not_inherit_unconfigured_hosted_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            store.create_session(
                workspace_root=root,
                provider="glm",
                model="glm-5",
            )
            args = SimpleNamespace(
                root=str(root),
                config=None,
                channel=None,
                provider=None,
                model=None,
            )

            with patch.dict("os.environ", {"GLM_API_KEY": ""}, clear=False):
                session = create_new_session(args, store)

        self.assertIsNone(session.provider)
        self.assertIsNone(session.model)

    def test_new_session_ignores_last_model_from_other_agent_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".agent"
            config_dir.mkdir()
            (config_dir / "config.json").write_text(
                json.dumps({
                    "agents": {
                        "default": "main",
                        "list": [{"id": "main"}, {"id": "docs"}],
                    }
                }),
                encoding="utf-8",
            )
            store = SessionStore(root / "sessions.sqlite3")
            store.create_session(
                workspace_root=root,
                provider="anthropic",
                model="claude-sonnet-4.5",
                agent_id="docs",
            )
            args = SimpleNamespace(
                root=str(root),
                config=None,
                channel=None,
                provider=None,
                model=None,
            )

            session = create_new_session(args, store)

        self.assertEqual(session.agent, "main")
        self.assertIsNone(session.provider)
        self.assertIsNone(session.model)

    def test_new_session_uses_config_default_agent_when_remembering_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".agent"
            config_dir.mkdir()
            (config_dir / "config.json").write_text(
                json.dumps({
                    "agents": {
                        "default": "docs",
                        "list": [{"id": "main"}, {"id": "docs"}],
                    }
                }),
                encoding="utf-8",
            )
            store = SessionStore(root / "sessions.sqlite3")
            store.create_session(
                workspace_root=root,
                provider="anthropic",
                model="claude-sonnet-4.5",
                agent_id="docs",
            )
            args = SimpleNamespace(
                root=str(root),
                config=None,
                channel=None,
                provider=None,
                model=None,
            )

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
                session = create_new_session(args, store)

        self.assertEqual(session.agent, "docs")
        self.assertEqual(session.provider, "anthropic")
        self.assertEqual(session.model, "claude-sonnet-4.5")

    def test_new_session_does_not_inherit_model_when_provider_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            store.create_session(
                workspace_root=root,
                provider="ollama",
                model="llama3.1",
            )
            args = SimpleNamespace(
                root=str(root),
                channel=None,
                provider="openai",
                model=None,
            )

            session = create_new_session(args, store)

        self.assertEqual(session.provider, "openai")
        self.assertIsNone(session.model)

    def test_resume_reuses_session_model_without_provider_override(self) -> None:
        args = SimpleNamespace(model=None, provider=None)
        session = SimpleNamespace(provider="ollama", model="llama3.1")

        self.assertEqual(_selected_model(args, session), "llama3.1")

    def test_provider_override_uses_provider_default_when_model_unspecified(self) -> None:
        args = SimpleNamespace(model=None, provider="ollama")
        session = SimpleNamespace(provider="openai", model="gpt-4o")

        self.assertIsNone(_selected_model(args, session))


if __name__ == "__main__":
    unittest.main()
