from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent.system_presence as system_presence
from agent.channel_health_policy import (
    ChannelHealthPolicy,
    evaluate_channel_health,
    resolve_channel_restart_reason,
)
from agent.config import ConfigError, AgentConfig, parse_agent_config, load_agent_config
from agent.device_identity import (
    default_identity_path,
    load_or_create_device_identity,
    public_key_raw_base64url_from_pem,
)
from agent.gateway import (
    ChannelRegistry,
    GatewayError,
    InboundAddress,
    InboundMessage,
    LifecycleHooks,
    AgentGateway,
    canonical_route_key,
)
from agent.gateway_runtime import (
    ChannelLifecycleManager,
    GatewayMethod,
    GatewayMethodRegistry,
    GatewayRuntimeError,
    LazyService,
    create_gateway_runtime_state,
)
from agent.heartbeat_events import (
    emit_heartbeat_event,
    get_last_heartbeat_event,
    resolve_indicator_type,
    reset_heartbeat_events_for_test,
)
from agent.heartbeat_wake import (
    are_heartbeats_enabled,
    reset_heartbeat_wake_state_for_tests,
)
from agent.main import _gateway_control_snapshot, _handle_local_command
from agent.planner import AgentSession, _build_initial_messages, run_agent
from agent.plugin_sdk import (
    PluginRegistry,
    active_plugin_registry,
    release_pinned_plugin_registry,
)
from agent.session_store import SessionStore, TokenUsage
from agent.skills import discover_skill_catalog
from agent.system_events import (
    enqueue_system_event,
    peek_system_events,
    reset_system_events_for_test,
    resolve_main_system_event_session_key,
)
from agent.system_presence import list_system_presence, reset_system_presence_for_tests
from agent.tools import ToolContext, build_tool_registry


class ConfigTests(unittest.TestCase):
    def test_defaults_keep_one_main_profile_and_per_sender_scope(self) -> None:
        config = parse_agent_config({}, Path("/workspace"))

        self.assertEqual(config.default_agent_id, "main")
        self.assertEqual(config.session.default_scope, "per-sender")
        self.assertEqual(tuple(config.agents), ("main",))
        self.assertIsNone(config.skill_allowlist("main"))

    def test_workspace_config_overrides_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "repo"
            user_config = root / "xdg" / "agent" / "config.json"
            workspace_config = workspace / ".agent" / "config.json"
            user_config.parent.mkdir(parents=True)
            workspace_config.parent.mkdir(parents=True)
            user_config.write_text(json.dumps({
                "session": {"default_scope": "shared"},
                "skills": {"max_loaded": 7},
            }), encoding="utf-8")
            workspace_config.write_text(json.dumps({
                "session": {"default_scope": "global"},
            }), encoding="utf-8")

            config = load_agent_config(
                workspace,
                environ={"XDG_CONFIG_HOME": str(root / "xdg")},
                home=root / "home",
            )

        self.assertEqual(config.session.default_scope, "global")
        self.assertEqual(config.skills.max_loaded, 7)
        self.assertEqual(config.source_paths, (user_config.resolve(), workspace_config.resolve()))

    def test_agent_skill_allowlist_supports_inherit_replace_and_disable(self) -> None:
        config = parse_agent_config({
            "agents": {
                "default": "main",
                "defaults": {"skills": ["shared"]},
                "list": [
                    {"id": "main"},
                    {"id": "docs", "skills": ["docs-search"], "tools": ["grep", "read_path"]},
                    {"id": "locked", "skills": [], "tools": []},
                ],
            },
        }, Path("/workspace"))

        self.assertEqual(config.skill_allowlist("main"), ("shared",))
        self.assertEqual(config.skill_allowlist("docs"), ("docs-search",))
        self.assertEqual(config.skill_allowlist("locked"), ())
        self.assertEqual(config.tool_allowlist("docs"), ("grep", "read_path"))
        self.assertEqual(config.tool_allowlist("locked"), ())

    def test_config_rejects_unknown_fields_and_unknown_binding_agent(self) -> None:
        with self.assertRaisesRegex(ConfigError, "Unknown config field"):
            parse_agent_config({"plugins": {}}, Path("/workspace"))
        with self.assertRaisesRegex(ConfigError, "unknown agent"):
            parse_agent_config({
                "bindings": [{
                    "agent": "missing",
                    "match": {"channel": "tui"},
                }],
            }, Path("/workspace"))


class SkillsTests(unittest.TestCase):
    def test_bundled_code_review_skill_is_available_without_workspace_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = discover_skill_catalog(
                root,
                AgentConfig(),
                home=root / "home",
            )

        self.assertIn("code-review", catalog.names())
        self.assertEqual(catalog.get("code-review").source, "bundled")  # type: ignore[union-attr]

    def test_skill_precedence_and_loading_are_bounded_to_configured_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace_skill = root / "skills" / "review" / "SKILL.md"
            project_skill = root / ".agents" / "skills" / "review" / "SKILL.md"
            workspace_skill.parent.mkdir(parents=True)
            project_skill.parent.mkdir(parents=True)
            workspace_skill.write_text(
                "---\nname: review\ndescription: Workspace review\ntools: [\"grep\", \"read_path\"]\n---\nUse workspace rules.",
                encoding="utf-8",
            )
            project_skill.write_text(
                "---\nname: review\ndescription: Lower priority\n---\nUse project rules.",
                encoding="utf-8",
            )

            catalog = discover_skill_catalog(
                root,
                AgentConfig(),
                home=root / "home",
                bundled_root=root / "bundled",
            )

        loaded = catalog.load("review")
        self.assertTrue(loaded["ok"])
        self.assertEqual(loaded["source"], "workspace")
        self.assertEqual(loaded["instructions"], "Use workspace rules.")
        self.assertEqual(loaded["required_tools"], ["grep", "read_path"])
        self.assertTrue(any("shadowed" in item.reason for item in catalog.skipped))

    def test_skill_allowlist_and_binary_requirements_filter_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, metadata in (
                ("enabled", ""),
                ("disabled", ""),
                ("missing-bin", "requires_bins: [\"definitely-not-a-agent-test-binary\"]\n"),
            ):
                path = root / "skills" / name / "SKILL.md"
                path.parent.mkdir(parents=True)
                path.write_text(
                    f"---\nname: {name}\ndescription: {name}\n{metadata}---\nInstructions for {name}.",
                    encoding="utf-8",
                )
            config = parse_agent_config({
                "agents": {
                    "defaults": {"skills": ["enabled", "missing-bin"]},
                    "list": [{"id": "main"}],
                },
            }, root)

            catalog = discover_skill_catalog(
                root,
                config,
                home=root / "home",
                bundled_root=root / "bundled",
            )

        self.assertEqual(catalog.names(), ("enabled",))
        self.assertTrue(any(item.name == "disabled" and "not enabled" in item.reason for item in catalog.skipped))
        self.assertTrue(any(item.name == "missing-bin" and "missing required" in item.reason for item in catalog.skipped))

    def test_load_skill_is_registered_without_expanding_discovery_subagent_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_path = root / "skills" / "review" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(
                "---\nname: review\ndescription: Review code\n---\nInspect first.",
                encoding="utf-8",
            )
            catalog = discover_skill_catalog(
                root,
                AgentConfig(),
                home=root / "home",
                bundled_root=root / "bundled",
            )
            ctx = ToolContext(
                rust=SimpleNamespace(),
                workspace_root=root,
                search_roots=[],
                skill_catalog=catalog,
            )
            registry = build_tool_registry(ctx)

            result = registry.execute("load_skill", {"name": "review"}, ctx)
            restricted = registry.restricted({"glob", "grep", "read_path"})

        self.assertTrue(result["ok"])
        self.assertIn("cannot expand", result["safety"].casefold())
        self.assertNotIn("load_skill", {schema["name"] for schema in restricted.schemas()})

    def test_initial_prompt_exposes_skill_index_without_eagerly_injecting_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_path = root / "skills" / "review" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(
                "---\nname: review\ndescription: Review code safely\n---\nPRIVATE FULL INSTRUCTIONS",
                encoding="utf-8",
            )
            catalog = discover_skill_catalog(
                root,
                AgentConfig(),
                home=root / "home",
                bundled_root=root / "bundled",
            )

            messages = _build_initial_messages(
                workspace_root=str(root),
                context_text="",
                session=AgentSession(),
                user_prompt="review this",
                conversation_history=None,
                skill_index_text=catalog.prompt_index(),
            )

        self.assertIn("review: Review code safely", messages[0]["content"])
        self.assertNotIn("PRIVATE FULL INSTRUCTIONS", messages[0]["content"])

    def test_agent_tool_allowlist_physically_removes_tools_and_skill_index(self) -> None:
        class CapturingLLM:
            mode = "hosted"

            def __init__(self) -> None:
                self.tool_names: list[str] = []
                self.first_message = ""

            def respond(self, *, messages: list[dict[str, object]], tools: list[dict[str, object]], **_kwargs: object) -> dict[str, object]:
                self.tool_names = [str(schema.get("name")) for schema in tools]
                self.first_message = str(messages[0]["content"])
                return {"output": [], "output_text": "done"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_path = root / "skills" / "review" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(
                "---\nname: review\ndescription: Review code safely\n---\nInspect first.",
                encoding="utf-8",
            )
            catalog = discover_skill_catalog(
                root,
                AgentConfig(),
                home=root / "home",
                bundled_root=root / "bundled",
            )
            llm = CapturingLLM()

            answer = run_agent(
                llm=llm,  # type: ignore[arg-type]
                rust=SimpleNamespace(),
                workspace_root=str(root),
                user_prompt="review this",
                skill_catalog=catalog,
                tool_allowlist=("grep",),
            )

        self.assertEqual(answer, "done")
        self.assertEqual(llm.tool_names, ["grep"])
        self.assertNotIn("Available Agent skills", llm.first_message)


class GatewayRoutingTests(unittest.TestCase):
    def _config(self, scope: str = "per-sender") -> AgentConfig:
        return parse_agent_config({
            "agents": {
                "default": "main",
                "list": [{"id": "main"}, {"id": "peer-agent"}],
            },
            "session": {"default_scope": scope},
            "bindings": [
                {
                    "agent": "main",
                    "match": {"channel": "chat", "account_id": "*"},
                },
                {
                    "agent": "peer-agent",
                    "scope": "shared",
                    "match": {
                        "channel": "chat",
                        "account_id": "*",
                        "peer": {"kind": "group", "id": "engineering"},
                    },
                },
            ],
        }, Path("/workspace"))

    def test_binding_match_prefers_peer_over_channel_account(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions.sqlite3")
            gateway = AgentGateway(config=self._config(), store=store)
            decision = gateway.decide_route(InboundAddress(
                channel="chat",
                account_id="work",
                sender_id="alice",
                peer_kind="group",
                peer_id="engineering",
            ))

        self.assertEqual(decision.agent_id, "peer-agent")
        self.assertEqual(decision.scope, "shared")
        self.assertEqual(decision.matched_binding, 1)

    def test_per_sender_route_is_durable_and_isolates_other_sender(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            alice = InboundAddress(channel="tui", sender_id="alice")
            bob = InboundAddress(channel="tui", sender_id="bob")

            first = gateway.open_session(alice, workspace_root=root)
            resumed = gateway.open_session(alice, workspace_root=root)
            separate = gateway.open_session(bob, workspace_root=root)
            route = store.get_route(first.decision.route_key)
            events = store.list_events(first.session.id)

        self.assertTrue(first.created)
        self.assertFalse(resumed.created)
        self.assertEqual(first.session.id, resumed.session.id)
        self.assertNotEqual(first.session.id, separate.session.id)
        self.assertEqual(route.session_id, first.session.id)
        self.assertEqual(route.sender_id, "alice")
        self.assertEqual(events[0].event_type, "session_route_bound")

    def test_same_channel_identity_is_isolated_between_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            address = InboundAddress(channel="tui", sender_id="alice")

            first = gateway.open_session(address, workspace_root=first_root)
            second = gateway.open_session(address, workspace_root=second_root)

        self.assertNotEqual(first.decision.route_key, second.decision.route_key)
        self.assertNotEqual(first.session.id, second.session.id)

    def test_shared_and_global_keys_ignore_sender_as_intended(self) -> None:
        alice = InboundAddress(channel="chat", sender_id="alice", peer_kind="group", peer_id="g1")
        bob = InboundAddress(channel="chat", sender_id="bob", peer_kind="group", peer_id="g1")

        self.assertEqual(
            canonical_route_key(alice, agent_id="main", scope="shared"),
            canonical_route_key(bob, agent_id="main", scope="shared"),
        )
        self.assertEqual(
            canonical_route_key(alice, agent_id="main", scope="global"),
            canonical_route_key(InboundAddress(channel="other", sender_id="carol"), agent_id="main", scope="global"),
        )
        self.assertNotEqual(
            canonical_route_key(alice, agent_id="main", scope="per-sender"),
            canonical_route_key(bob, agent_id="main", scope="per-sender"),
        )

    def test_per_sender_route_requires_identity(self) -> None:
        with self.assertRaisesRegex(GatewayError, "requires sender_id"):
            canonical_route_key(
                InboundAddress(channel="chat"),
                agent_id="main",
                scope="per-sender",
            )

    def test_hooks_are_observers_and_channel_registry_normalizes_ingress(self) -> None:
        class TestAdapter:
            channel_id = "test"

            def normalize(self, payload: dict[str, object]) -> InboundMessage:
                return InboundMessage(
                    address=InboundAddress(channel="test", sender_id=str(payload["sender"])),
                    text=str(payload["text"]),
                )

        channels = ChannelRegistry()
        channels.register(TestAdapter())
        hooks = LifecycleHooks()
        observed: list[str] = []
        hooks.register("message_ingress", lambda payload: observed.append(str(payload["channel"])))
        hooks.register("message_ingress", lambda _payload: (_ for _ in ()).throw(RuntimeError("hook failed")))

        result = hooks.emit("message_ingress", {"channel": "test"})
        message = channels.normalize("test", {"sender": "alice", "text": "hello"})

        self.assertEqual(observed, ["test"])
        self.assertEqual(result.errors, ["hook failed"])
        self.assertEqual(message.text, "hello")

    def test_default_gateway_exposes_structured_control_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            snapshot = gateway.control_snapshot()

        self.assertEqual(snapshot["overview"]["state"], "ready")
        self.assertIn("tui", snapshot["overview"]["channels"])
        self.assertEqual(snapshot["routes"][0]["session_id"], routed.session.id)
        self.assertTrue(any(item["name"] == "health" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "status" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "diagnostics.stability" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "logs.tail" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "gateway.suspend.prepare" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "gateway.suspend.status" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "gateway.suspend.resume" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "usage.status" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "usage.cost" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "models.list" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "models.authStatus" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "commands.list" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "tools.catalog" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "tools.effective" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "agents.list" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "skills.status" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "tasks.list" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "tasks.get" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "tasks.cancel" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "config.get" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "config.schema" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "config.schema.lookup" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "chat.history" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "chat.metadata" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "chat.message.get" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "tts.status" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "cron.list" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "plugins.list" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "terminal.open" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "update.status" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "gateway.routes" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "channels.status" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "channels.start" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "channels.stop" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "channels.logout" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "gateway.identity.get" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "last-heartbeat" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "set-heartbeats" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "system-presence" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "system-event" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.list" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.preview" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.compaction.list" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.compaction.get" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.compaction.branch" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.compaction.restore" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.describe" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.resolve" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.create" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.send" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.steer" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.abort" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.patch" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.reset" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.delete" for item in snapshot["methods"]))
        self.assertTrue(any(item["name"] == "sessions.compact" for item in snapshot["methods"]))
        self.assertEqual(snapshot["channels"][0]["state"], "registered")

    def test_channels_status_returns_openclaw_runtime_payload_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            payload = gateway.call("channels.status")

        self.assertIsInstance(payload["ts"], int)
        self.assertEqual(payload["channelOrder"], ["tui"])
        self.assertEqual(payload["channelLabels"], {"tui": "TUI"})
        self.assertEqual(payload["channelDetailLabels"], {"tui": "Local terminal UI"})
        self.assertEqual(payload["channelSystemImages"], {})
        self.assertEqual(payload["channelMeta"], [{
            "id": "tui",
            "label": "TUI",
            "detailLabel": "Local terminal UI",
        }])
        self.assertEqual(payload["channelDefaultAccountId"], {"tui": "default"})
        self.assertEqual(payload["channels"]["tui"]["configured"], True)
        self.assertEqual(payload["channels"]["tui"]["running"], False)
        account = payload["channelAccounts"]["tui"][0]
        self.assertEqual(account["accountId"], "default")
        self.assertEqual(account["state"], "registered")
        self.assertEqual(account["configured"], True)
        self.assertEqual(account["enabled"], True)
        self.assertEqual(account["running"], False)
        self.assertEqual(account["connected"], False)
        self.assertEqual(account["healthState"], "not-running")

    def test_channels_status_filters_channel_and_validates_protocol_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            filtered = gateway.call("channels.status", {
                "channel": "TUI",
                "probe": True,
                "timeoutMs": 999_999,
            })

            with self.assertRaisesRegex(GatewayError, "unknown channel"):
                gateway.call("channels.status", {"channel": "missing"})
            with self.assertRaisesRegex(GatewayError, "probe must be a boolean"):
                gateway.call("channels.status", {"probe": "yes"})
            with self.assertRaisesRegex(GatewayError, "timeoutMs must be a finite number"):
                gateway.call("channels.status", {"timeoutMs": "slow"})

        self.assertEqual(filtered["channelOrder"], ["tui"])
        self.assertEqual(set(filtered["channelAccounts"]), {"tui"})

    def test_channels_status_reflects_lifecycle_runtime_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            generation = gateway.runtime.channels.begin_start("tui", "default")
            self.assertTrue(gateway.runtime.channels.mark_running("tui", "default", generation))
            running_payload = gateway.call("channels.status", {"channel": "tui"})

            failed_generation = gateway.runtime.channels.begin_start("tui", "default")
            self.assertTrue(gateway.runtime.channels.mark_failed("tui", "default", failed_generation, "transport closed"))
            failed_payload = gateway.call("channels.status", {"channel": "tui"})

        running = running_payload["channelAccounts"]["tui"][0]
        self.assertEqual(running["state"], "running")
        self.assertTrue(running["running"])
        self.assertTrue(running["connected"])
        self.assertIsInstance(running["lastStartAt"], int)
        self.assertIsInstance(running["lastTransportActivityAt"], int)
        self.assertNotIn("healthState", running)

        failed = failed_payload["channelAccounts"]["tui"][0]
        self.assertEqual(failed["state"], "backoff")
        self.assertFalse(failed["running"])
        self.assertEqual(failed["reconnectAttempts"], 1)
        self.assertEqual(failed["lastError"], "transport closed")
        self.assertEqual(failed["healthState"], "not-running")
        self.assertIn("retryAt", failed)

    def test_channels_stop_requires_write_scope_and_channel_param(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call("channels.stop", {"channel": "tui"})
            with self.assertRaisesRegex(GatewayError, "invalid channels.stop channel"):
                gateway.call(
                    "channels.stop",
                    {},
                    granted_scopes=frozenset({"gateway.write"}),
                )
            with self.assertRaisesRegex(GatewayError, "unknown channel"):
                gateway.call(
                    "channels.stop",
                    {"channel": "missing"},
                    granted_scopes=frozenset({"gateway.write"}),
                )
            with self.assertRaisesRegex(GatewayError, "invalid channels.stop accountId"):
                gateway.call(
                    "channels.stop",
                    {"channel": "tui", "accountId": 7},
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_channels_stop_marks_running_channel_stopped_without_logging_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            generation = gateway.runtime.channels.begin_start("tui", "default")
            self.assertTrue(gateway.runtime.channels.mark_running("tui", "default", generation))

            result = gateway.call(
                "channels.stop",
                {"channel": "TUI"},
                granted_scopes=frozenset({"gateway.write"}),
            )
            status = gateway.call("channels.status", {"channel": "tui"})

        self.assertEqual(result, {"channel": "tui", "accountId": "default", "stopped": True})
        account = status["channelAccounts"]["tui"][0]
        self.assertEqual(account["state"], "stopped")
        self.assertFalse(account["running"])
        self.assertFalse(account["connected"])
        self.assertTrue(account["abortRequested"])
        self.assertTrue(account["manuallyStopped"])
        self.assertEqual(account["healthState"], "not-running")

    def test_channels_start_requires_write_scope_and_channel_param(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call("channels.start", {"channel": "tui"})
            with self.assertRaisesRegex(GatewayError, "invalid channels.start channel"):
                gateway.call(
                    "channels.start",
                    {},
                    granted_scopes=frozenset({"gateway.write"}),
                )
            with self.assertRaisesRegex(GatewayError, "unknown channel"):
                gateway.call(
                    "channels.start",
                    {"channel": "missing"},
                    granted_scopes=frozenset({"gateway.write"}),
                )
            with self.assertRaisesRegex(GatewayError, "invalid channels.start accountId"):
                gateway.call(
                    "channels.start",
                    {"channel": "tui", "accountId": 7},
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_channels_start_marks_registered_channel_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call(
                "channels.start",
                {"channel": "TUI"},
                granted_scopes=frozenset({"gateway.write"}),
            )
            status = gateway.call("channels.status", {"channel": "tui"})

        self.assertEqual(result, {"channel": "tui", "accountId": "default", "started": True})
        account = status["channelAccounts"]["tui"][0]
        self.assertEqual(account["state"], "running")
        self.assertTrue(account["running"])
        self.assertTrue(account["connected"])
        self.assertTrue(account["taskActive"])
        self.assertFalse(account["startPending"])
        self.assertFalse(account["manuallyStopped"])
        self.assertIsInstance(account["lastStartAt"], int)
        self.assertIsInstance(account["lastTransportActivityAt"], int)
        self.assertNotIn("healthState", account)

    def test_channels_start_after_stop_clears_manual_stopped_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            gateway.call(
                "channels.stop",
                {"channel": "tui"},
                granted_scopes=frozenset({"gateway.write"}),
            )
            restarted = gateway.call(
                "channels.start",
                {"channel": "tui"},
                granted_scopes=frozenset({"gateway.write"}),
            )
            status = gateway.call("channels.status", {"channel": "tui"})

        self.assertTrue(restarted["started"])
        account = status["channelAccounts"]["tui"][0]
        self.assertEqual(account["state"], "running")
        self.assertFalse(account["abortRequested"])
        self.assertFalse(account["manuallyStopped"])

    def test_channels_logout_requires_write_scope_and_channel_param(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call("channels.logout", {"channel": "tui"})
            with self.assertRaisesRegex(GatewayError, "invalid channels.logout channel"):
                gateway.call(
                    "channels.logout",
                    {},
                    granted_scopes=frozenset({"gateway.write"}),
                )
            with self.assertRaisesRegex(GatewayError, "unknown channel"):
                gateway.call(
                    "channels.logout",
                    {"channel": "missing"},
                    granted_scopes=frozenset({"gateway.write"}),
                )
            with self.assertRaisesRegex(GatewayError, "invalid channels.logout accountId"):
                gateway.call(
                    "channels.logout",
                    {"channel": "tui", "accountId": 7},
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_channels_logout_reports_unsupported_without_stopping_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))
            gateway.call(
                "channels.start",
                {"channel": "tui"},
                granted_scopes=frozenset({"gateway.write"}),
            )

            with self.assertRaisesRegex(GatewayError, "channel tui does not support logout"):
                gateway.call(
                    "channels.logout",
                    {"channel": "TUI"},
                    granted_scopes=frozenset({"gateway.write"}),
                )

            status = gateway.call("channels.status", {"channel": "tui"})

        account = status["channelAccounts"]["tui"][0]
        self.assertEqual(account["state"], "running")
        self.assertTrue(account["running"])
        self.assertFalse(account["manuallyStopped"])

    def test_health_returns_openclaw_style_gateway_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.add_message(routed.session.id, "user", "hello")

            health = gateway.call("health")

        self.assertTrue(health["ok"])
        self.assertIsInstance(health["ts"], int)
        self.assertIsInstance(health["durationMs"], int)
        self.assertEqual(health["defaultAgentId"], "main")
        self.assertEqual(health["channelOrder"], ["tui"])
        self.assertEqual(health["channelLabels"], {"tui": "TUI"})
        self.assertEqual(health["heartbeatSeconds"], 0)
        self.assertEqual(health["plugins"]["loaded"], ["agent.builtin.channels"])
        self.assertEqual(health["plugins"]["errors"], [])
        self.assertEqual(health["sessions"]["path"], str(root / "sessions.sqlite3"))
        self.assertEqual(health["sessions"]["count"], 1)
        self.assertEqual(health["sessions"]["recent"][0]["key"], routed.decision.route_key)
        self.assertIsInstance(health["sessions"]["recent"][0]["updatedAt"], int)
        self.assertIsInstance(health["sessions"]["recent"][0]["age"], int)
        self.assertEqual(health["agents"][0]["agentId"], "main")
        self.assertTrue(health["agents"][0]["isDefault"])
        self.assertEqual(health["agents"][0]["sessions"]["count"], 1)
        self.assertEqual(health["channels"]["tui"]["accountId"], "default")
        self.assertEqual(set(health["channels"]["tui"]["accounts"]), {"default"})
        self.assertEqual(health["channels"]["tui"]["healthState"], "not-running")

    def test_health_accepts_probe_timeout_params_without_mutating_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            health = gateway.call("health", {"probe": True, "timeoutMs": 999_999})
            account = gateway.call("channels.status", {"channel": "tui"})["channelAccounts"]["tui"][0]

        self.assertTrue(health["ok"])
        self.assertEqual(account["state"], "registered")

    def test_health_validates_protocol_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "probe must be a boolean"):
                gateway.call("health", {"probe": "yes"})
            with self.assertRaisesRegex(GatewayError, "timeoutMs must be a finite number"):
                gateway.call("health", {"timeoutMs": "slow"})

    def test_health_reflects_running_channel_without_unhealthy_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))
            gateway.call(
                "channels.start",
                {"channel": "tui"},
                granted_scopes=frozenset({"gateway.write"}),
            )

            health = gateway.call("health")

        tui = health["channels"]["tui"]
        self.assertEqual(tui["state"], "running")
        self.assertTrue(tui["running"])
        self.assertNotIn("healthState", tui)

    def test_status_returns_openclaw_style_redacted_gateway_summary(self) -> None:
        reset_system_events_for_test()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = parse_agent_config({}, root)
                store = SessionStore(root / "sessions.sqlite3")
                gateway = AgentGateway(config=config, store=store)
                gateway.open_session(
                    InboundAddress(channel="tui", sender_id="alice"),
                    workspace_root=root,
                )
                enqueue_system_event(
                    "maintenance window",
                    session_key=resolve_main_system_event_session_key(config),
                )

                status = gateway.call("status")

            self.assertEqual(status["runtimeVersion"], "0.1.0")
            self.assertEqual(
                status["heartbeat"],
                {
                    "defaultAgentId": "main",
                    "agents": [
                        {
                            "agentId": "main",
                            "enabled": False,
                            "every": "off",
                            "everyMs": None,
                        }
                    ],
                },
            )
            self.assertEqual(status["channelSummary"], ["TUI: registered (default)"])
            self.assertEqual(status["queuedSystemEvents"], ["maintenance window"])
            self.assertEqual(
                status["tasks"],
                {
                    "total": 0,
                    "active": 0,
                    "terminal": 0,
                    "failures": 0,
                    "byStatus": {
                        "queued": 0,
                        "running": 0,
                        "succeeded": 0,
                        "failed": 0,
                        "timed_out": 0,
                        "cancelled": 0,
                        "lost": 0,
                    },
                    "byRuntime": {
                        "subagent": 0,
                        "acp": 0,
                        "cli": 0,
                        "cron": 0,
                    },
                },
            )
            self.assertEqual(status["taskAudit"]["total"], 0)
            self.assertEqual(status["taskAudit"]["errors"], 0)
            self.assertEqual(status["taskAudit"]["warnings"], 0)
            self.assertEqual(status["sessions"]["paths"], [])
            self.assertEqual(status["sessions"]["count"], 1)
            self.assertEqual(status["sessions"]["defaults"], {"model": None, "contextTokens": None})
            self.assertEqual(status["sessions"]["recent"], [])
            self.assertEqual(
                status["sessions"]["byAgent"],
                [
                    {
                        "agentId": "main",
                        "path": "[redacted]",
                        "count": 1,
                        "recent": [],
                    }
                ],
            )
        finally:
            reset_system_events_for_test()

    def test_status_can_omit_channel_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            status = gateway.call("status", {"includeChannelSummary": False})

        self.assertEqual(status["channelSummary"], [])

    def test_status_validates_protocol_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "includeChannelSummary must be a boolean"):
                gateway.call("status", {"includeChannelSummary": "no"})

    def test_status_reflects_running_channel_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))
            gateway.call(
                "channels.start",
                {"channel": "tui"},
                granted_scopes=frozenset({"gateway.write"}),
            )

            status = gateway.call("status")

        self.assertEqual(status["channelSummary"], ["TUI: running (default)"])

    def test_diagnostics_stability_returns_safe_runtime_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.add_message(routed.session.id, "user", "secret prompt body")

            result = gateway.call("diagnostics.stability", {"probe": False})

        self.assertTrue(result["ok"])
        self.assertEqual(result["runtime"]["state"], "ready")
        self.assertEqual(result["channels"]["order"], ["tui"])
        self.assertEqual(result["sessions"]["count"], 1)
        self.assertFalse(result["privacy"]["messageBodies"])
        self.assertNotIn("secret prompt body", json.dumps(result))

    def test_diagnostics_and_logs_validate_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "probe must be a boolean"):
                gateway.call("diagnostics.stability", {"probe": "yes"})
            with self.assertRaisesRegex(GatewayError, "limit must be an integer from 1 to 1000"):
                gateway.call("logs.tail", {"limit": 0})

    def test_logs_tail_reports_unavailable_without_faking_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call("logs.tail", {"limit": 10})

        self.assertEqual(
            result,
            {
                "available": False,
                "entries": [],
                "cursor": None,
                "reason": "Agent Gateway file logging is not configured.",
            },
        )

    def test_gateway_suspend_prepare_status_resume_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            prepared = gateway.call(
                "gateway.suspend.prepare",
                {"timeoutMs": 10_000},
                granted_scopes=frozenset({"gateway.admin"}),
            )
            status = gateway.call("gateway.suspend.status")
            resumed = gateway.call(
                "gateway.suspend.resume",
                {"leaseId": prepared["leaseId"]},
                granted_scopes=frozenset({"gateway.admin"}),
            )
            after = gateway.call("gateway.suspend.status")

        self.assertTrue(prepared["prepared"])
        self.assertTrue(status["active"])
        self.assertEqual(status["leaseId"], prepared["leaseId"])
        self.assertTrue(resumed["resumed"])
        self.assertFalse(after["active"])

    def test_gateway_suspend_validates_scope_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call("gateway.suspend.prepare")
            with self.assertRaisesRegex(GatewayError, "timeoutMs must be an integer"):
                gateway.call(
                    "gateway.suspend.prepare",
                    {"timeoutMs": 1},
                    granted_scopes=frozenset({"gateway.admin"}),
                )

    def test_config_get_schema_and_lookup_return_sanitized_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_agent_config(
                {
                    "agents": {
                        "default": "main",
                        "defaults": {"tools": ["read_path"]},
                        "list": [{"id": "main"}],
                    },
                    "session": {"default_scope": "shared"},
                },
                root,
            )
            gateway = AgentGateway(config=config, store=SessionStore(root / "sessions.sqlite3"))

            config_result = gateway.call("config.get")
            schema = gateway.call("config.schema", granted_scopes=frozenset({"gateway.admin"}))
            lookup = gateway.call("config.schema.lookup", {"path": "session.default_scope"})

        self.assertEqual(config_result["config"]["session"]["default_scope"], "shared")
        self.assertEqual(config_result["config"]["agents"]["defaults"]["tools"], ["read_path"])
        self.assertFalse(config_result["secretsIncluded"])
        self.assertEqual(len(config_result["configRevisionHash"]), 64)
        self.assertEqual(schema["schema"]["type"], "object")
        self.assertTrue(lookup["found"])
        self.assertEqual(lookup["schema"]["enum"], ["global", "per-sender", "shared"])

    def test_chat_history_metadata_and_message_get_are_session_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.update_llm_config(routed.session.id, provider="openai", model="gpt-5.5")
            first = store.add_message(routed.session.id, "user", "hello world")
            second = store.add_message(routed.session.id, "assistant", "long assistant reply")

            history = gateway.call(
                "chat.history",
                {"sessionKey": routed.decision.route_key, "maxChars": 5},
            )
            metadata = gateway.call("chat.metadata", {"sessionKey": routed.decision.route_key})
            message = gateway.call(
                "chat.message.get",
                {"sessionKey": routed.decision.route_key, "messageId": str(second.id)},
            )

        self.assertEqual(history["sessionKey"], routed.decision.route_key)
        self.assertEqual(
            [(item["seq"], item["role"], item["content"], item["truncated"]) for item in history["messages"]],
            [(first.seq, "user", "hello", True), (second.seq, "assistant", "long ", True)],
        )
        self.assertEqual(metadata["provider"], "openai")
        self.assertEqual(metadata["model"], "gpt-5.5")
        self.assertEqual(message["message"]["id"], str(second.id))
        self.assertEqual(message["message"]["content"], "long assistant reply")

    def test_chat_read_methods_validate_required_session_and_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "chat RPC requires sessionKey"):
                gateway.call("chat.history")
            with self.assertRaisesRegex(GatewayError, "maxChars must be a positive integer"):
                gateway.call("chat.history", {"sessionKey": "missing", "maxChars": 0})

    def test_usage_status_aggregates_default_agent_session_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.update_llm_config(routed.session.id, provider="openai", model="gpt-5.5")
            store.add_usage(
                routed.session.id,
                tokens=TokenUsage(input=10, output=20, reasoning=5, cache_read=3, cache_write=2),
                cost_usd=0.25,
            )

            result = gateway.call("usage.status")

        self.assertEqual(result["agentId"], "main")
        self.assertIsNone(result["agentScope"])
        self.assertFalse(result["quotaAvailable"])
        self.assertEqual(result["sessions"], 1)
        self.assertEqual(
            result["tokens"],
            {
                "input": 10,
                "output": 20,
                "reasoning": 5,
                "cacheRead": 3,
                "cacheWrite": 2,
                "total": 40,
            },
        )
        self.assertEqual(result["costUsd"], 0.25)
        self.assertEqual(result["byProvider"][0]["provider"], "openai")
        self.assertEqual(result["byProvider"][0]["tokens"]["total"], 40)

    def test_usage_cost_can_aggregate_all_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_agent_config(
                {
                    "agents": {
                        "default": "main",
                        "list": [{"id": "main"}, {"id": "docs"}],
                    }
                },
                root,
            )
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=config, store=store)
            main = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            docs = store.create_session(workspace_root=root, agent_id="docs")
            store.update_llm_config(main.session.id, provider="openai", model="gpt-5.5")
            store.update_llm_config(docs.id, provider="anthropic", model="claude-sonnet-4.5")
            store.add_usage(main.session.id, tokens=TokenUsage(input=1, output=2), cost_usd=0.10)
            store.add_usage(docs.id, tokens=TokenUsage(input=3, output=4), cost_usd=0.20)

            result = gateway.call("usage.cost", {"agentScope": "all"})

        self.assertIsNone(result["agentId"])
        self.assertEqual(result["agentScope"], "all")
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["tokens"]["total"], 10)
        self.assertAlmostEqual(result["costUsd"], 0.30)
        self.assertEqual({row["provider"] for row in result["byProvider"]}, {"anthropic", "openai"})

    def test_usage_methods_validate_agent_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "agentId must be a string"):
                gateway.call("usage.status", {"agentId": 1})
            with self.assertRaisesRegex(GatewayError, "unknown agent: docs"):
                gateway.call("usage.cost", {"agentId": "docs"})
            with self.assertRaisesRegex(GatewayError, "agentScope must be all"):
                gateway.call("usage.status", {"agentScope": "team"})

    def test_models_list_returns_openclaw_style_catalog_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with patch.dict(os.environ, {"OPENAI_API_KEY": "", "AGENT_OPENAI_COMPAT_BASE_URL": ""}, clear=False):
                with patch("agent.main._discover_provider_models", side_effect=AssertionError("live discovery")):
                    result = gateway.call("models.list", {"view": "configured"})

        self.assertEqual(result["view"], "configured")
        self.assertTrue(result["models"])
        by_id = {row["id"]: row for row in result["models"]}
        self.assertIn("openai/gpt-5.5", by_id)
        self.assertEqual(by_id["openai/gpt-5.5"]["provider"], "openai")
        self.assertEqual(by_id["openai/gpt-5.5"]["model"], "gpt-5.5")
        self.assertEqual(by_id["openai/gpt-5.5"]["status"], "auth_required")
        self.assertEqual(by_id["openai/gpt-5.5"]["auth"]["kind"], "api_key")
        self.assertEqual(by_id["openai/gpt-5.5"]["auth"]["env"], "OPENAI_API_KEY")
        self.assertNotIn("sk-", json.dumps(result))

        self.assertIn("ollama/qwen3", by_id)
        self.assertTrue(by_id["ollama/qwen3"]["local"])
        self.assertTrue(by_id["ollama/qwen3"]["openSource"])
        self.assertEqual(by_id["ollama/qwen3"]["auth"], {"required": False, "kind": "none"})
        self.assertEqual(by_id["ollama/qwen3"]["status"], "local_runtime_not_checked")
        self.assertEqual(by_id["ollama/qwen3"]["install"]["provider"], "ollama")
        self.assertEqual(by_id["ollama/qwen3"]["install"]["parameters"], "8B")

        providers = {row["id"]: row for row in result["providers"]}
        self.assertFalse(providers["openai"]["local"])
        self.assertTrue(providers["ollama"]["local"])

    def test_models_list_validates_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "view must be default, configured, provider-config, or all"):
                gateway.call("models.list", {"view": "everything"})
            with self.assertRaisesRegex(GatewayError, "view must be a string"):
                gateway.call("models.list", {"view": 1})

    def test_models_list_reports_api_key_configured_without_secret_material(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret"}, clear=False):
                result = gateway.call("models.list", {"view": "all"})

        by_id = {row["id"]: row for row in result["models"]}
        self.assertEqual(result["view"], "all")
        self.assertTrue(by_id["openai/gpt-5.5"]["auth"]["configured"])
        self.assertEqual(by_id["openai/gpt-5.5"]["status"], "available")
        self.assertNotIn("sk-test-secret", json.dumps(result))

    def test_models_auth_status_strips_credentials_and_reports_provider_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "anthropic-secret"}, clear=False):
                result = gateway.call("models.authStatus")

        self.assertFalse(result["cached"])
        self.assertEqual(result["cacheTtlSeconds"], 0)
        self.assertIsInstance(result["generatedAt"], int)
        providers = {row["provider"]: row for row in result["providers"]}
        self.assertEqual(providers["openai"]["status"], "missing")
        self.assertTrue(providers["openai"]["attention"])
        self.assertEqual(providers["openai"]["env"], "OPENAI_API_KEY")
        self.assertEqual(providers["anthropic"]["status"], "configured")
        self.assertFalse(providers["anthropic"]["attention"])
        self.assertEqual(providers["ollama"]["status"], "not_required")
        self.assertFalse(providers["ollama"]["required"])
        self.assertNotIn("anthropic-secret", json.dumps(result))

    def test_models_auth_status_filters_provider_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call("models.authStatus", {"provider": "claude"})

        self.assertEqual([row["provider"] for row in result["providers"]], ["anthropic"])

    def test_models_auth_status_validates_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "provider must be a string"):
                gateway.call("models.authStatus", {"provider": 2})
            with self.assertRaisesRegex(GatewayError, "Unsupported LLM provider"):
                gateway.call("models.authStatus", {"provider": "unknown-provider"})

    def test_commands_list_returns_openclaw_style_runtime_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call("commands.list", {"provider": "claude"})

        self.assertEqual(result["agentId"], "main")
        self.assertEqual(result["scope"], "both")
        self.assertEqual(result["provider"], "anthropic")
        commands = {row["name"]: row for row in result["commands"]}
        self.assertIn("model", commands)
        self.assertEqual(commands["model"]["source"], "core")
        self.assertEqual(commands["model"]["textAliases"], ["/model", "/models"])
        self.assertEqual(
            commands["model"]["args"],
            [
                {"name": "provider", "type": "string", "required": False},
                {"name": "model", "type": "string", "required": False},
            ],
        )
        self.assertEqual(commands["install"]["textAliases"], ["/install"])
        self.assertNotIn("tools", commands)
        self.assertNotIn("devices", commands)
        self.assertNotIn("capabilities", commands)
        self.assertEqual(commands["apikey"]["textAliases"], ["/apikey", "/key"])
        self.assertTrue(commands["apikey"]["args"][1]["secret"])
        self.assertEqual(commands["exit"]["textAliases"], ["/exit", "/quit", "/q"])

    def test_commands_list_can_omit_argument_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call("commands.list", {"scope": "text", "includeArgs": False})

        self.assertEqual(result["scope"], "text")
        self.assertTrue(result["commands"])
        self.assertTrue(all("args" not in row for row in result["commands"]))
        self.assertIn("/model", result["commands"][0]["textAliases"])

    def test_commands_list_validates_protocol_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_agent_config({}, root)
            gateway = AgentGateway(config=config, store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "scope must be text, native, or both"):
                gateway.call("commands.list", {"scope": "visual"})
            with self.assertRaisesRegex(GatewayError, "includeArgs must be a boolean"):
                gateway.call("commands.list", {"includeArgs": "no"})
            with self.assertRaisesRegex(GatewayError, "provider must be a string"):
                gateway.call("commands.list", {"provider": 1})
            with self.assertRaisesRegex(GatewayError, "unknown agent: other"):
                gateway.call("commands.list", {"agentId": "other"})

    def test_tools_catalog_returns_core_tool_inventory_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call("tools.catalog")

        self.assertEqual(result["agentId"], "main")
        tools = {row["name"]: row for row in result["tools"]}
        self.assertIn("read_path", tools)
        self.assertIn("write_file", tools)
        self.assertNotIn("discovery_subagent", tools)
        self.assertEqual(tools["read_path"]["source"], "core")
        self.assertFalse(tools["read_path"]["optional"])
        self.assertTrue(tools["read_path"]["enabled"])
        self.assertEqual(tools["read_path"]["group"], "workspace")
        self.assertEqual(tools["read_path"]["schema"]["name"], "read_path")
        self.assertIn("description", tools["read_path"]["schema"])
        groups = {row["id"]: row for row in result["groups"]}
        self.assertIn("workspace", groups)
        self.assertIn("host_inspection", groups)
        self.assertIn("desktop_control", groups)
        self.assertIn("read_path", groups["workspace"]["tools"])

    def test_tools_catalog_marks_agent_allowlist_without_hiding_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_agent_config(
                {
                    "agents": {
                        "default": "main",
                        "list": [
                            {
                                "id": "main",
                                "tools": ["read_path", "grep"],
                            }
                        ],
                    }
                },
                root,
            )
            gateway = AgentGateway(config=config, store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call("tools.catalog", {"agentId": "main"})

        tools = {row["name"]: row for row in result["tools"]}
        self.assertTrue(tools["read_path"]["enabled"])
        self.assertTrue(tools["grep"]["enabled"])
        self.assertFalse(tools["write_file"]["enabled"])

    def test_tools_catalog_validates_agent_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "agentId must be a string"):
                gateway.call("tools.catalog", {"agentId": 3})
            with self.assertRaisesRegex(GatewayError, "unknown agent: other"):
                gateway.call("tools.catalog", {"agentId": "other"})

    def test_tools_effective_derives_session_agent_and_applies_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_agent_config(
                {
                    "agents": {
                        "default": "main",
                        "list": [{"id": "main", "tools": ["read_path", "grep"]}],
                    }
                },
                root,
            )
            gateway = AgentGateway(config=config, store=SessionStore(root / "sessions.sqlite3"))
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            result = gateway.call("tools.effective", {"sessionKey": routed.decision.route_key})

        self.assertEqual(result["sessionKey"], routed.decision.route_key)
        self.assertEqual(result["sessionId"], routed.session.id)
        self.assertEqual(result["agentId"], "main")
        tools = {row["name"]: row for row in result["tools"]}
        self.assertEqual(set(tools), {"grep", "read_path"})
        self.assertTrue(tools["read_path"]["enabled"])
        self.assertEqual(result["notices"], [])

    def test_tools_effective_validates_session_key_and_agent_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_agent_config(
                {
                    "agents": {
                        "default": "main",
                        "list": [{"id": "main"}, {"id": "docs"}],
                    }
                },
                root,
            )
            gateway = AgentGateway(config=config, store=SessionStore(root / "sessions.sqlite3"))
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayError, "sessionKey must be a non-empty string"):
                gateway.call("tools.effective", {})
            with self.assertRaisesRegex(KeyError, "Session target belongs to agent main"):
                gateway.call(
                    "tools.effective",
                    {"sessionKey": routed.decision.route_key, "agentId": "docs"},
                )

    def test_agents_list_returns_configured_agent_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_agent_config(
                {
                    "agents": {
                        "default": "main",
                        "defaults": {
                            "skills": ["code-review"],
                            "tools": ["read_path", "grep"],
                        },
                        "list": [
                            {"id": "main"},
                            {"id": "docs", "skills": ["writing"], "tools": ["read_path"]},
                        ],
                    }
                },
                root,
            )
            gateway = AgentGateway(config=config, store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call("agents.list")

        self.assertEqual(result["defaultAgentId"], "main")
        agents = {row["id"]: row for row in result["agents"]}
        self.assertTrue(agents["main"]["isDefault"])
        self.assertFalse(agents["docs"]["isDefault"])
        self.assertEqual(agents["main"]["skills"], ["code-review"])
        self.assertEqual(agents["main"]["tools"], ["read_path", "grep"])
        self.assertEqual(agents["docs"]["skills"], ["writing"])
        self.assertEqual(agents["docs"]["tools"], ["read_path"])
        self.assertIsNone(agents["main"]["model"])
        self.assertEqual(agents["main"]["runtime"], {"orchestrator": "python", "executor": "rust"})

    def test_skills_status_returns_visible_skill_inventory_without_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "skills" / "alpha"
            skill_dir.mkdir(parents=True)
            skill_dir.joinpath("SKILL.md").write_text(
                "---\n"
                "name: alpha\n"
                "description: Alpha skill\n"
                "tools: [\"read_path\"]\n"
                "---\n"
                "Private instruction body that should not be returned by status.\n",
                encoding="utf-8",
            )
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))
            gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            result = gateway.call("skills.status")

        self.assertEqual(result["agentId"], "main")
        skills = {row["name"]: row for row in result["skills"]}
        self.assertIn("alpha", skills)
        self.assertEqual(skills["alpha"]["description"], "Alpha skill")
        self.assertEqual(skills["alpha"]["source"], "workspace")
        self.assertEqual(skills["alpha"]["requiredTools"], ["read_path"])
        self.assertTrue(skills["alpha"]["eligible"])
        self.assertNotIn("Private instruction body", json.dumps(result))
        self.assertIn("Skills provide instructions only", result["safety"])

    def test_skills_status_reports_missing_allowlisted_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = parse_agent_config(
                {
                    "agents": {
                        "default": "main",
                        "defaults": {"skills": ["missing-skill"]},
                        "list": [{"id": "main"}],
                    }
                },
                root,
            )
            gateway = AgentGateway(config=config, store=SessionStore(root / "sessions.sqlite3"))
            gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            result = gateway.call("skills.status")

        self.assertEqual(result["available"], 0)
        skipped = {row["name"]: row for row in result["skipped"]}
        self.assertEqual(
            skipped["missing-skill"]["reason"],
            "enabled skill was not found in any configured skill root",
        )

    def test_skills_status_validates_agent_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "agentId must be a string"):
                gateway.call("skills.status", {"agentId": 3})
            with self.assertRaisesRegex(GatewayError, "unknown agent: other"):
                gateway.call("skills.status", {"agentId": "other"})

    def test_tasks_list_returns_empty_task_ledger_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call("tasks.list", {"status": ["queued", "running"], "limit": 50})

        self.assertEqual(result, {"tasks": []})

    def test_tasks_get_reports_missing_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "task not found: task-1"):
                gateway.call("tasks.get", {"taskId": "task-1"})

    def test_tasks_cancel_returns_not_found_without_faking_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call(
                "tasks.cancel",
                {"taskId": "task-1", "reason": "user requested"},
                granted_scopes=frozenset({"gateway.write"}),
            )

        self.assertEqual(
            result,
            {
                "found": False,
                "cancelled": False,
                "taskId": "task-1",
                "reason": "user requested",
            },
        )

    def test_task_methods_validate_protocol_params_and_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayError, "unknown status stale"):
                gateway.call("tasks.list", {"status": "stale"})
            with self.assertRaisesRegex(GatewayError, "limit must be an integer from 1 to 500"):
                gateway.call("tasks.list", {"limit": 0})
            with self.assertRaisesRegex(GatewayError, "taskId must be a non-empty string"):
                gateway.call("tasks.get", {"taskId": ""})
            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call("tasks.cancel", {"taskId": "task-1"})

    def test_unsupported_openclaw_methods_return_explicit_unsupported_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            read_result = gateway.call("tts.status")
            write_result = gateway.call(
                "chat.send",
                granted_scopes=frozenset({"gateway.write"}),
            )
            admin_result = gateway.call(
                "terminal.open",
                granted_scopes=frozenset({"gateway.admin"}),
            )

        self.assertEqual(read_result["status"], "unsupported")
        self.assertEqual(read_result["method"], "tts.status")
        self.assertFalse(read_result["ok"])
        self.assertEqual(write_result["status"], "unsupported")
        self.assertEqual(write_result["method"], "chat.send")
        self.assertEqual(admin_result["status"], "unsupported")
        self.assertEqual(admin_result["method"], "terminal.open")

    def test_unsupported_write_methods_still_enforce_gateway_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call("chat.send")
            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call("terminal.open")

    def test_mcp_methods_remain_unimplemented_by_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            with self.assertRaisesRegex(GatewayRuntimeError, "Unknown gateway method"):
                gateway.call("mcp.app.listTools")

    def test_device_identity_is_persisted_with_private_file_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_home = Path(tmp) / "xdg"
            with patch.dict(os.environ, {"XDG_DATA_HOME": str(data_home)}):
                first = load_or_create_device_identity()
                second = load_or_create_device_identity()
                path = default_identity_path()
                raw_public_key = public_key_raw_base64url_from_pem(first.public_key_pem)

            stored = json.loads(path.read_text(encoding="utf-8"))
            file_mode = oct(path.stat().st_mode & 0o777)

        self.assertEqual(second.device_id, first.device_id)
        self.assertEqual(stored["version"], 1)
        self.assertEqual(stored["deviceId"], first.device_id)
        self.assertIn("BEGIN PRIVATE KEY", stored["privateKeyPem"])
        self.assertEqual(len(first.device_id), 64)
        self.assertEqual(len(raw_public_key), 43)
        self.assertNotIn("=", raw_public_key)
        self.assertEqual(file_mode, "0o600")

    def test_gateway_identity_get_returns_stable_public_identity_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"XDG_DATA_HOME": str(root / "xdg")}):
                gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

                first = gateway.call("gateway.identity.get")
                second = gateway.call("gateway.identity.get")

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"deviceId", "publicKey"})
        self.assertEqual(len(first["deviceId"]), 64)
        self.assertEqual(len(first["publicKey"]), 43)
        self.assertNotIn("private", json.dumps(first).casefold())

    def test_last_heartbeat_returns_none_before_event(self) -> None:
        reset_heartbeat_events_for_test()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

            result = gateway.call("last-heartbeat")

        self.assertIsNone(result)

    def test_last_heartbeat_returns_latest_emitted_event(self) -> None:
        reset_heartbeat_events_for_test()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

                first = emit_heartbeat_event({
                    "status": "sent",
                    "channel": "tui",
                    "to": "alice",
                    "preview": "ping",
                    "durationMs": 12,
                    "indicatorType": resolve_indicator_type("sent"),
                })
                self.assertEqual(gateway.call("last-heartbeat"), first)

                second = emit_heartbeat_event({
                    "status": "ok-empty",
                    "channel": "tui",
                    "indicatorType": resolve_indicator_type("ok-empty"),
                })
                result = gateway.call("last-heartbeat")

            self.assertEqual(result, second)
            self.assertEqual(result["status"], "ok-empty")
            self.assertEqual(result["indicatorType"], "ok")
            self.assertIsInstance(result["ts"], int)
        finally:
            reset_heartbeat_events_for_test()

    def test_last_heartbeat_rejects_unknown_status_at_emit_boundary(self) -> None:
        reset_heartbeat_events_for_test()
        with self.assertRaisesRegex(ValueError, "Heartbeat status"):
            emit_heartbeat_event({"status": "unknown"})
        self.assertIsNone(get_last_heartbeat_event())

    def test_heartbeat_indicator_resolution_matches_openclaw_status_mapping(self) -> None:
        self.assertEqual(resolve_indicator_type("ok-empty"), "ok")
        self.assertEqual(resolve_indicator_type("ok-token"), "ok")
        self.assertEqual(resolve_indicator_type("sent"), "alert")
        self.assertEqual(resolve_indicator_type("failed"), "error")
        self.assertIsNone(resolve_indicator_type("skipped"))
        with self.assertRaisesRegex(ValueError, "Unsupported heartbeat status"):
            resolve_indicator_type("unknown")

    def test_set_heartbeats_requires_admin_scope_and_boolean_enabled(self) -> None:
        reset_heartbeat_wake_state_for_tests()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

                with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                    gateway.call("set-heartbeats", {"enabled": False})
                self.assertTrue(are_heartbeats_enabled())

                with self.assertRaisesRegex(GatewayError, "enabled \\(boolean\\) required"):
                    gateway.call(
                        "set-heartbeats",
                        {"enabled": "false"},
                        granted_scopes=frozenset({"gateway.admin"}),
                    )

            self.assertTrue(are_heartbeats_enabled())
        finally:
            reset_heartbeat_wake_state_for_tests()

    def test_set_heartbeats_toggles_global_heartbeat_processing_state(self) -> None:
        reset_heartbeat_wake_state_for_tests()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

                disabled = gateway.call(
                    "set-heartbeats",
                    {"enabled": False},
                    granted_scopes=frozenset({"gateway.admin"}),
                )
                self.assertEqual(disabled, {"ok": True, "enabled": False})
                self.assertFalse(are_heartbeats_enabled())

                enabled = gateway.call(
                    "set-heartbeats",
                    {"enabled": True},
                    granted_scopes=frozenset({"gateway.admin"}),
                )
                self.assertEqual(enabled, {"ok": True, "enabled": True})
                self.assertTrue(are_heartbeats_enabled())
        finally:
            reset_heartbeat_wake_state_for_tests()

    def test_system_presence_returns_self_gateway_entry(self) -> None:
        reset_system_presence_for_tests()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

                presence = gateway.call("system-presence")

            self.assertGreaterEqual(len(presence), 1)
            self.assertEqual(presence[0]["mode"], "gateway")
            self.assertEqual(presence[0]["reason"], "self")
            self.assertIn("Gateway:", presence[0]["text"])
            self.assertIsInstance(presence[0]["ts"], int)
            self.assertIn("host", presence[0])
            self.assertIn("platform", presence[0])
            self.assertIn("deviceFamily", presence[0])
        finally:
            reset_system_presence_for_tests()

    def test_system_presence_matches_direct_presence_store_listing(self) -> None:
        reset_system_presence_for_tests()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))
                gateway_presence = gateway.call("system-presence")
                direct_presence = list_system_presence()

            self.assertEqual(gateway_presence[0]["host"], direct_presence[0]["host"])
            self.assertEqual(gateway_presence[0]["mode"], direct_presence[0]["mode"])
            self.assertEqual(gateway_presence[0]["reason"], direct_presence[0]["reason"])
        finally:
            reset_system_presence_for_tests()

    def test_system_presence_prunes_expired_entries_when_listed(self) -> None:
        reset_system_presence_for_tests()
        try:
            system_presence._entries["old-node"] = {
                "text": "Node: old · mode stale",
                "ts": 0,
            }

            presence = list_system_presence()

            self.assertFalse(any(item["text"] == "Node: old · mode stale" for item in presence))
            self.assertTrue(any(item.get("mode") == "gateway" for item in presence))
        finally:
            reset_system_presence_for_tests()

    def test_system_event_requires_admin_scope_and_text(self) -> None:
        reset_system_events_for_test()
        reset_system_presence_for_tests()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                gateway = AgentGateway(config=parse_agent_config({}, root), store=SessionStore(root / "sessions.sqlite3"))

                with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                    gateway.call("system-event", {"text": "maintenance"})
                with self.assertRaisesRegex(GatewayError, "text required"):
                    gateway.call(
                        "system-event",
                        {"text": "   "},
                        granted_scopes=frozenset({"gateway.admin"}),
                    )
        finally:
            reset_system_events_for_test()
            reset_system_presence_for_tests()

    def test_system_event_updates_presence_and_enqueues_direct_event_for_main_session(self) -> None:
        reset_system_events_for_test()
        reset_system_presence_for_tests()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                hooks = LifecycleHooks()
                observed: list[dict[str, object]] = []
                hooks.register("presence_snapshot", lambda payload: observed.append(dict(payload)))
                config = parse_agent_config({}, root)
                gateway = AgentGateway(
                    config=config,
                    store=SessionStore(root / "sessions.sqlite3"),
                    hooks=hooks,
                )

                result = gateway.call(
                    "system-event",
                    {
                        "text": "maintenance window",
                        "deviceId": "node-1",
                        "host": "worker-a",
                        "roles": ["runner"],
                        "scopes": ["local"],
                        "lastInputSeconds": "ignored",
                    },
                    granted_scopes=frozenset({"gateway.admin"}),
                )
                session_key = resolve_main_system_event_session_key(config)
                presence = gateway.call("system-presence")

            self.assertEqual(result, {"ok": True})
            self.assertEqual(peek_system_events(session_key), ["maintenance window"])
            self.assertTrue(any(item.get("host") == "worker-a" for item in presence))
            self.assertTrue(observed)
            self.assertIn("presence", observed[0])
        finally:
            reset_system_events_for_test()
            reset_system_presence_for_tests()

    def test_system_event_node_line_enqueues_only_changed_presence_delta(self) -> None:
        reset_system_events_for_test()
        reset_system_presence_for_tests()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = parse_agent_config({}, root)
                gateway = AgentGateway(config=config, store=SessionStore(root / "sessions.sqlite3"))
                session_key = resolve_main_system_event_session_key(config)

                gateway.call(
                    "system-event",
                    {
                        "text": "Node: worker (10.0.0.2) · app 1.0 · last input 3s ago · mode idle · reason startup",
                        "deviceId": "node-1",
                    },
                    granted_scopes=frozenset({"gateway.admin"}),
                )
                gateway.call(
                    "system-event",
                    {
                        "text": "Node: worker (10.0.0.2) · app 1.0 · last input 4s ago · mode idle · reason heartbeat",
                        "deviceId": "node-1",
                    },
                    granted_scopes=frozenset({"gateway.admin"}),
                )
                gateway.call(
                    "system-event",
                    {
                        "text": "Node: worker (10.0.0.2) · app 1.1 · last input 5s ago · mode busy · reason user",
                        "deviceId": "node-1",
                    },
                    granted_scopes=frozenset({"gateway.admin"}),
                )

            self.assertEqual(
                peek_system_events(session_key),
                [
                    "Node: worker (10.0.0.2) · app 1.0 · mode idle · reason startup",
                    "app 1.1 · mode busy · reason user",
                ],
            )
        finally:
            reset_system_events_for_test()
            reset_system_presence_for_tests()

    def test_sessions_list_returns_openclaw_bounded_index_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
                provider="anthropic",
                model="claude-3-5-sonnet-latest",
            )
            store.add_message(routed.session.id, "user", "hello")

            result = gateway.call("sessions.list", {})

        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["totalCount"], 1)
        self.assertEqual(result["limitApplied"], 100)
        self.assertFalse(result["hasMore"])
        self.assertEqual(result["stores"][0]["path"], str(store.db_path))
        self.assertEqual(len(result["sessions"]), 1)
        row = result["sessions"][0]
        self.assertEqual(row["key"], routed.decision.route_key)
        self.assertEqual(row["sessionId"], routed.session.id)
        self.assertEqual(row["agentId"], "main")
        self.assertEqual(row["model"], "anthropic/claude-3-5-sonnet-latest")
        self.assertEqual(row["provider"], "anthropic")
        self.assertEqual(row["modelName"], "claude-3-5-sonnet-latest")
        self.assertIsNone(row["agentRuntime"])
        self.assertEqual(row["routes"][0]["route_key"], routed.decision.route_key)
        json.dumps(result)

    def test_sessions_list_supports_limit_all_agent_and_active_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            old = store.create_session(workspace_root=root, title="old", agent_id="main")
            current = store.create_session(workspace_root=root, title="current", agent_id="main")
            other = store.create_session(workspace_root=root, title="other", agent_id="work")
            stale = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
            with store._connect() as conn:
                conn.execute(
                    "update sessions set updated_at = ? where id = ?",
                    (stale, old.id),
                )

            all_rows = gateway.call("sessions.list", {"limit": "all"})
            active_rows = gateway.call("sessions.list", {
                "agentId": "main",
                "activeMinutes": 30,
                "limit": "all",
            })
            first_row = gateway.call("sessions.list", {"limit": 1})

        self.assertIsNone(all_rows["limitApplied"])
        self.assertEqual(all_rows["totalCount"], 3)
        self.assertEqual(all_rows["count"], 3)
        self.assertFalse(all_rows["hasMore"])
        self.assertEqual(
            {row["sessionId"] for row in active_rows["sessions"]},
            {current.id},
        )
        self.assertEqual(active_rows["totalCount"], 1)
        self.assertEqual(first_row["limitApplied"], 1)
        self.assertEqual(first_row["count"], 1)
        self.assertEqual(first_row["totalCount"], 3)
        self.assertTrue(first_row["hasMore"])
        self.assertIn(first_row["sessions"][0]["sessionId"], {current.id, other.id})

    def test_sessions_list_rejects_invalid_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )

            with self.assertRaisesRegex(GatewayError, "limit"):
                gateway.call("sessions.list", {"limit": 0})

    def test_sessions_preview_returns_bounded_tail_for_route_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            for index in range(5):
                role = "user" if index % 2 == 0 else "assistant"
                store.add_message(routed.session.id, role, f"message {index}")

            preview = gateway.call(
                "sessions.preview",
                {"session_key": routed.decision.route_key, "limit": 3},
            )

        self.assertEqual(preview["session_id"], routed.session.id)
        self.assertEqual(preview["route_key"], routed.decision.route_key)
        self.assertEqual(preview["limit"], 3)
        self.assertEqual(
            [(message["seq"], message["role"], message["content"]) for message in preview["messages"]],
            [
                (3, "user", "message 2"),
                (4, "assistant", "message 3"),
                (5, "user", "message 4"),
            ],
        )

    def test_sessions_preview_resolves_direct_session_id_and_caps_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            session = store.create_session(workspace_root=root)
            store.add_message(session.id, "user", "hello")

            preview = gateway.call(
                "sessions.preview",
                {"session_id": session.id, "limit": 500},
            )

        self.assertEqual(preview["session_id"], session.id)
        self.assertIsNone(preview["route_key"])
        self.assertEqual(preview["limit"], 200)
        self.assertEqual(preview["messages"][0]["content"], "hello")

    def test_sessions_describe_returns_one_row_for_exact_route_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
                provider="ollama",
                model="qwen3",
            )
            store.add_messages(
                routed.session.id,
                [("user", "hello")],
                last_prompt="hello",
            )

            described = gateway.call(
                "sessions.describe",
                {"session_key": routed.decision.route_key},
            )

        self.assertEqual(described["session_id"], routed.session.id)
        self.assertEqual(described["session_key"], routed.decision.route_key)
        self.assertEqual(described["route_key"], routed.decision.route_key)
        self.assertEqual(described["agent_id"], "main")
        self.assertEqual(described["provider"], "ollama")
        self.assertEqual(described["model"], "qwen3")
        self.assertEqual(described["last_prompt"], "hello")
        self.assertEqual(described["tokens"]["input"], 0)
        self.assertEqual(described["routes"][0]["route_key"], routed.decision.route_key)

    def test_sessions_describe_supports_direct_session_id_without_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            session = store.create_session(
                workspace_root=root,
                provider="openai",
                model="gpt-5.5",
            )

            described = gateway.call(
                "sessions.describe",
                {"session_id": session.id},
            )

        self.assertEqual(described["session_id"], session.id)
        self.assertIsNone(described["session_key"])
        self.assertEqual(described["routes"], [])
        self.assertEqual(described["provider"], "openai")
        self.assertEqual(described["model"], "gpt-5.5")

    def test_sessions_describe_rejects_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )

            with self.assertRaisesRegex(GatewayError, "sessions.describe requires"):
                gateway.call("sessions.describe", {})

    def test_sessions_resolve_canonicalizes_route_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            resolved = gateway.call(
                "sessions.resolve",
                {"key": routed.decision.route_key},
            )

        self.assertTrue(resolved["found"])
        self.assertEqual(resolved["source"], "route_key")
        self.assertEqual(resolved["session_id"], routed.session.id)
        self.assertEqual(resolved["session_key"], routed.decision.route_key)
        self.assertEqual(resolved["key"], routed.decision.route_key)
        self.assertEqual(resolved["route"]["route_key"], routed.decision.route_key)

    def test_sessions_resolve_canonicalizes_session_id_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            session = store.create_session(workspace_root=root, title="direct")

            resolved = gateway.call(
                "sessions.resolve",
                {"sessionId": session.id[:8]},
            )

        self.assertTrue(resolved["found"])
        self.assertEqual(resolved["source"], "session_id")
        self.assertEqual(resolved["session_id"], session.id)
        self.assertIsNone(resolved["session_key"])
        self.assertIsNone(resolved["route"])

    def test_sessions_resolve_supports_exact_title_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            session = store.create_session(workspace_root=root, title="Release Plan")

            resolved = gateway.call(
                "sessions.resolve",
                {"label": "Release Plan", "agentId": "agent"},
            )

        self.assertTrue(resolved["found"])
        self.assertEqual(resolved["source"], "label")
        self.assertEqual(resolved["session_id"], session.id)
        self.assertEqual(resolved["agent_id"], "agent")
        self.assertEqual(resolved["title"], "Release Plan")

    def test_sessions_resolve_can_return_unknown_target_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )

            resolved = gateway.call(
                "sessions.resolve",
                {"key": "missing-session", "includeUnknown": True},
            )

        self.assertFalse(resolved["found"])
        self.assertEqual(resolved["source"], "unknown")
        self.assertEqual(resolved["key"], "missing-session")
        self.assertIsNone(resolved["session_id"])

    def test_sessions_create_requires_write_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.create",
                    {"label": "Write gated"},
                    granted_scopes=frozenset({"gateway.read"}),
                )

    def test_sessions_create_creates_direct_agent_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)

            created = gateway.call(
                "sessions.create",
                {
                    "label": "Scratch",
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "workspaceRoot": str(root),
                },
                granted_scopes=frozenset({"gateway.write"}),
            )

        self.assertTrue(created["created"])
        self.assertFalse(created["runStarted"])
        self.assertEqual(created["title"], "Scratch")
        self.assertEqual(created["agent_id"], "main")
        self.assertEqual(created["provider"], "openai")
        self.assertEqual(created["model"], "gpt-5.5")
        self.assertIsNone(created["session_key"])
        self.assertEqual(created["routes"], [])

    def test_sessions_create_creates_routed_session_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)

            created = gateway.call(
                "sessions.create",
                {
                    "key": "agent:main:manual:release",
                    "label": "Release",
                    "workspaceRoot": str(root),
                },
                granted_scopes=frozenset({"gateway.write"}),
            )
            route_session_id = store.get_route("agent:main:manual:release").session_id

        self.assertEqual(created["session_key"], "agent:main:manual:release")
        self.assertEqual(created["route_key"], "agent:main:manual:release")
        self.assertEqual(created["routes"][0]["channel"], "gateway")
        self.assertEqual(route_session_id, created["session_id"])

    def test_sessions_create_rejects_duplicate_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            params = {
                "key": "agent:main:manual:release",
                "workspaceRoot": str(root),
            }
            gateway.call("sessions.create", params, granted_scopes=frozenset({"gateway.write"}))

            with self.assertRaisesRegex(GatewayError, "already exists"):
                gateway.call("sessions.create", params, granted_scopes=frozenset({"gateway.write"}))

    def test_sessions_create_reports_nested_initial_send_as_not_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )

            created = gateway.call(
                "sessions.create",
                {
                    "label": "Draft",
                    "message": "start here",
                    "workspaceRoot": str(root),
                },
                granted_scopes=frozenset({"gateway.write"}),
            )

        self.assertTrue(created["created"])
        self.assertFalse(created["runStarted"])
        self.assertIn("sessions.send", created["runError"])

    def test_sessions_send_requires_write_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.send",
                    {
                        "key": routed.decision.route_key,
                        "message": "hello",
                    },
                    granted_scopes=frozenset({"gateway.read"}),
                )

    def test_sessions_send_admits_user_message_to_existing_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            observed: list[dict[str, object]] = []
            hooks = LifecycleHooks()
            hooks.register("session_transcript_updated", lambda payload: observed.append(dict(payload)))
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store, hooks=hooks)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            sent = gateway.call(
                "sessions.send",
                {
                    "key": routed.decision.route_key,
                    "message": "hello from dashboard",
                    "idempotencyKey": "idem-send-1",
                },
                granted_scopes=frozenset({"gateway.write"}),
            )
            messages = store.list_messages(routed.session.id, limit=None)
            updated = store.get_session(routed.session.id)

        self.assertTrue(sent["ok"])
        self.assertEqual(sent["runId"], "idem-send-1")
        self.assertFalse(sent["runStarted"])
        self.assertEqual(sent["messageSeq"], 1)
        self.assertIn("persisted", sent["runError"])
        self.assertEqual(
            [(message.seq, message.role, message.content) for message in messages],
            [(1, "user", "hello from dashboard")],
        )
        self.assertEqual(updated.last_prompt, "hello from dashboard")
        self.assertEqual(observed[-1]["message"], {"role": "user", "content": "hello from dashboard"})
        self.assertEqual(
            observed[-1]["target"],
            {
                "agent_id": "main",
                "session_id": routed.session.id,
                "session_key": routed.decision.route_key,
            },
        )

    def test_sessions_send_idempotency_replays_without_duplicate_transcript_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            params = {
                "key": routed.decision.route_key,
                "message": "send once",
                "idempotencyKey": "idem-send-once",
            }

            first = gateway.call(
                "sessions.send",
                params,
                granted_scopes=frozenset({"gateway.write"}),
            )
            second = gateway.call(
                "sessions.send",
                params,
                granted_scopes=frozenset({"gateway.write"}),
            )
            messages = store.list_messages(routed.session.id, limit=None)

        self.assertEqual(first, second)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "send once")

    def test_sessions_send_rejects_attachments_until_runtime_support_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayError, "attachments"):
                gateway.call(
                    "sessions.send",
                    {
                        "key": routed.decision.route_key,
                        "message": "has file",
                        "attachments": [{"name": "x.txt"}],
                    },
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_sessions_steer_requires_write_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.steer",
                    {
                        "key": routed.decision.route_key,
                        "message": "interrupt with this",
                    },
                    granted_scopes=frozenset({"gateway.read"}),
                )

    def test_sessions_steer_reports_idle_without_mutating_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            result = gateway.call(
                "sessions.steer",
                {
                    "key": routed.decision.route_key,
                    "message": "interrupt with this",
                },
                granted_scopes=frozenset({"gateway.write"}),
            )
            messages = store.list_messages(routed.session.id, limit=None)

        self.assertTrue(result["ok"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "idle")
        self.assertEqual(result["reason"], "no-active-run")
        self.assertFalse(result["runStarted"])
        self.assertEqual(messages, [])

    def test_sessions_steer_reports_active_run_unsupported_without_fake_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with gateway.session_lease(routed.session.id):
                result = gateway.call(
                    "sessions.steer",
                    {
                        "key": routed.decision.route_key,
                        "message": "interrupt with this",
                    },
                    granted_scopes=frozenset({"gateway.write"}),
                )
            messages = store.list_messages(routed.session.id, limit=None)

        self.assertTrue(result["ok"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["reason"], "active-run-steering-not-wired")
        self.assertEqual(messages, [])

    def test_sessions_steer_rejects_attachments_until_runtime_support_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayError, "attachments"):
                gateway.call(
                    "sessions.steer",
                    {
                        "key": routed.decision.route_key,
                        "message": "has file",
                        "attachments": [{"name": "x.txt"}],
                    },
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_sessions_abort_requires_write_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.abort",
                    {"key": routed.decision.route_key},
                    granted_scopes=frozenset({"gateway.read"}),
                )

    def test_sessions_abort_reports_idle_without_mutating_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            result = gateway.call(
                "sessions.abort",
                {"key": routed.decision.route_key},
                granted_scopes=frozenset({"gateway.write"}),
            )
            messages = store.list_messages(routed.session.id, limit=None)

        self.assertTrue(result["ok"])
        self.assertFalse(result["aborted"])
        self.assertEqual(result["status"], "idle")
        self.assertEqual(result["reason"], "no-active-run")
        self.assertEqual(result["sessionId"], routed.session.id)
        self.assertEqual(messages, [])

    def test_sessions_abort_reports_active_run_unsupported_without_unlocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with gateway.session_lease(routed.session.id):
                result = gateway.call(
                    "sessions.abort",
                    {
                        "key": routed.decision.route_key,
                        "runId": "run-active",
                    },
                    granted_scopes=frozenset({"gateway.write"}),
                )
                still_active = gateway._session_has_active_lease(routed.session.id)
            messages = store.list_messages(routed.session.id, limit=None)

        self.assertTrue(result["ok"])
        self.assertFalse(result["aborted"])
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["reason"], "active-run-abort-not-wired")
        self.assertEqual(result["runId"], "run-active")
        self.assertTrue(still_active)
        self.assertEqual(messages, [])

    def test_sessions_abort_run_id_only_reports_unwired_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )

            result = gateway.call(
                "sessions.abort",
                {"runId": "run-unknown"},
                granted_scopes=frozenset({"gateway.write"}),
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["aborted"])
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["reason"], "run-id-resolution-not-wired")
        self.assertEqual(result["runId"], "run-unknown")
        self.assertIsNone(result["sessionId"])

    def test_sessions_patch_requires_admin_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.patch",
                    {
                        "key": routed.decision.route_key,
                        "label": "Renamed",
                    },
                    granted_scopes=frozenset({"gateway.read"}),
                )
            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.patch",
                    {
                        "key": routed.decision.route_key,
                        "label": "Renamed",
                    },
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_sessions_patch_updates_label_model_and_reasoning_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
                provider="openai",
                model="gpt-5.5-mini",
            )

            result = gateway.call(
                "sessions.patch",
                {
                    "key": routed.decision.route_key,
                    "label": "RA project",
                    "providerOverride": "anthropic",
                    "modelOverride": "claude-3-5-sonnet-latest",
                    "thinkingLevel": "high",
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )
            updated = store.get_session(routed.session.id)
            messages = store.list_messages(routed.session.id, limit=None)

        self.assertTrue(result["ok"])
        self.assertTrue(result["patched"])
        self.assertEqual(result["title"], "RA project")
        self.assertEqual(result["provider"], "anthropic")
        self.assertEqual(result["model"], "claude-3-5-sonnet-latest")
        self.assertEqual(result["resolvedModel"], {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-latest",
        })
        self.assertIsNone(result["agentRuntime"])
        self.assertEqual(updated.title, "RA project")
        self.assertEqual(updated.provider, "anthropic")
        self.assertEqual(updated.model, "claude-3-5-sonnet-latest")
        self.assertEqual(updated.state, {"reasoning_effort": "high"})
        self.assertEqual(messages, [])

    def test_sessions_patch_rejects_unsupported_openclaw_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayError, "sendPolicy"):
                gateway.call(
                    "sessions.patch",
                    {
                        "key": routed.decision.route_key,
                        "sendPolicy": "manual",
                    },
                    granted_scopes=frozenset({"gateway.admin"}),
                )

            with self.assertRaisesRegex(GatewayError, "reasoningLevel"):
                gateway.call(
                    "sessions.patch",
                    {
                        "key": routed.decision.route_key,
                        "reasoningLevel": "off",
                    },
                    granted_scopes=frozenset({"gateway.admin"}),
                )

            self.assertEqual(store.get_session(routed.session.id).title, "New session")

    def test_sessions_patch_rejects_invalid_thinking_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayError, "thinking level"):
                gateway.call(
                    "sessions.patch",
                    {
                        "key": routed.decision.route_key,
                        "thinkingLevel": "xhigh",
                    },
                    granted_scopes=frozenset({"gateway.admin"}),
                )

            self.assertIsNone(store.get_session(routed.session.id).state)

    def test_sessions_reset_requires_admin_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.reset",
                    {"key": routed.decision.route_key},
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_sessions_reset_rolls_route_to_fresh_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
                provider="anthropic",
                model="claude-3-5-sonnet-latest",
            )
            store.add_messages(
                routed.session.id,
                [
                    ("user", "old question"),
                    ("assistant", "old answer"),
                ],
                last_prompt="old question",
            )
            store.patch_session_metadata(
                routed.session.id,
                title="RA project",
                state_patch={
                    "reasoning_effort": "high",
                    "active_root": str(root / "old-project"),
                    "tool_loop_history": [{"tool": "inspect", "signature": "old"}],
                },
            )

            result = gateway.call(
                "sessions.reset",
                {
                    "key": routed.decision.route_key,
                    "reason": "reset",
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )
            route = store.get_route(routed.decision.route_key)
            old_messages = store.list_messages(routed.session.id, limit=None)
            new_messages = store.list_messages(result["sessionId"], limit=None)
            updated = store.get_session(result["sessionId"])

        self.assertTrue(result["ok"])
        self.assertTrue(result["reset"])
        self.assertEqual(result["key"], routed.decision.route_key)
        self.assertEqual(result["previousSessionId"], routed.session.id)
        self.assertNotEqual(result["sessionId"], routed.session.id)
        self.assertEqual(route.session_id, result["sessionId"])
        self.assertEqual(
            [(message.role, message.content) for message in old_messages],
            [("user", "old question"), ("assistant", "old answer")],
        )
        self.assertEqual(new_messages, [])
        self.assertEqual(updated.title, "RA project")
        self.assertEqual(updated.provider, "anthropic")
        self.assertEqual(updated.model, "claude-3-5-sonnet-latest")
        self.assertEqual(updated.state, {"reasoning_effort": "high"})
        self.assertIsNone(updated.active_root)
        self.assertIsNone(updated.last_prompt)
        self.assertEqual(result["entry"]["session_id"], result["sessionId"])
        json.dumps(result)

    def test_sessions_reset_rejects_session_id_without_route_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            session = store.create_session(workspace_root=root)

            with self.assertRaisesRegex(GatewayError, "requires key"):
                gateway.call(
                    "sessions.reset",
                    {"sessionId": session.id},
                    granted_scopes=frozenset({"gateway.admin"}),
                )

    def test_sessions_delete_requires_admin_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.delete",
                    {
                        "key": routed.decision.route_key,
                        "deleteTranscript": True,
                    },
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_sessions_delete_removes_route_and_transcript_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.add_messages(
                routed.session.id,
                [
                    ("user", "old question"),
                    ("assistant", "old answer"),
                ],
            )
            store.add_event(
                routed.session.id,
                event_type="tool_call",
                summary="inspected project",
            )

            result = gateway.call(
                "sessions.delete",
                {
                    "key": routed.decision.route_key,
                    "deleteTranscript": True,
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )

            self.assertTrue(result["ok"])
            self.assertTrue(result["deleted"])
            self.assertEqual(result["key"], routed.decision.route_key)
            self.assertEqual(result["sessionId"], routed.session.id)
            self.assertEqual(result["messagesDeleted"], 2)
            self.assertEqual(result["eventsDeleted"], 2)
            self.assertEqual(result["routesDeleted"], 1)
            with self.assertRaises(KeyError):
                store.get_route(routed.decision.route_key)
            with self.assertRaises(KeyError):
                store.get_session(routed.session.id)
            self.assertEqual(store.list_messages(routed.session.id, limit=None), [])
            self.assertEqual(store.list_events(routed.session.id), [])
            json.dumps(result)

    def test_sessions_delete_rejects_keep_transcript_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.add_message(routed.session.id, "user", "keep me")

            with self.assertRaisesRegex(GatewayError, "cannot keep transcripts"):
                gateway.call(
                    "sessions.delete",
                    {
                        "key": routed.decision.route_key,
                        "deleteTranscript": False,
                    },
                    granted_scopes=frozenset({"gateway.admin"}),
                )

            self.assertEqual(store.get_route(routed.decision.route_key).session_id, routed.session.id)
            self.assertEqual(
                [(message.role, message.content) for message in store.list_messages(routed.session.id, limit=None)],
                [("user", "keep me")],
            )

    def test_sessions_delete_rejects_active_session_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.add_message(routed.session.id, "user", "active")

            with gateway.session_lease(routed.session.id):
                with self.assertRaisesRegex(GatewayError, "active session"):
                    gateway.call(
                        "sessions.delete",
                        {"key": routed.decision.route_key},
                        granted_scopes=frozenset({"gateway.admin"}),
                    )

            self.assertEqual(store.get_route(routed.decision.route_key).session_id, routed.session.id)
            self.assertEqual(
                [(message.role, message.content) for message in store.list_messages(routed.session.id, limit=None)],
                [("user", "active")],
            )

    def test_sessions_compact_requires_admin_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.compact",
                    {
                        "key": routed.decision.route_key,
                        "maxLines": 2,
                    },
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_sessions_compact_max_lines_prunes_active_transcript_and_archives_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.add_messages(
                routed.session.id,
                [
                    ("user", "message 0"),
                    ("assistant", "message 1"),
                    ("user", "message 2"),
                    ("assistant", "message 3"),
                    ("user", "message 4"),
                ],
            )

            result = gateway.call(
                "sessions.compact",
                {
                    "key": routed.decision.route_key,
                    "maxLines": 2,
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )
            messages = store.list_messages(routed.session.id, limit=None)
            events = store.list_events(routed.session.id, limit=10)
            compact_events = [event for event in events if event.event_type == "session_compacted"]

        self.assertTrue(result["ok"])
        self.assertTrue(result["compacted"])
        self.assertEqual(result["key"], routed.decision.route_key)
        self.assertEqual(result["mode"], "maxLines")
        self.assertEqual(result["kept"], 2)
        self.assertEqual(result["pruned"], 3)
        self.assertEqual(result["result"], {"linesBefore": 5, "linesAfter": 2})
        self.assertEqual(
            [(message.seq, message.role, message.content) for message in messages],
            [(4, "assistant", "message 3"), (5, "user", "message 4")],
        )
        self.assertEqual(len(compact_events), 1)
        self.assertEqual(result["archived"], f"sqlite:event:{compact_events[0].id}")
        self.assertEqual(result["archiveEventId"], compact_events[0].id)
        self.assertEqual(
            [(item["seq"], item["content"]) for item in compact_events[0].data["messages"]],
            [(1, "message 0"), (2, "message 1"), (3, "message 2")],
        )
        json.dumps(result)

    def test_sessions_compact_rejects_llm_mode_until_provider_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.add_message(routed.session.id, "user", "do not change")

            with self.assertRaisesRegex(GatewayError, "LLM summarization is not implemented"):
                gateway.call(
                    "sessions.compact",
                    {"key": routed.decision.route_key},
                    granted_scopes=frozenset({"gateway.admin"}),
                )

            self.assertEqual(
                [(message.role, message.content) for message in store.list_messages(routed.session.id, limit=None)],
                [("user", "do not change")],
            )

    def test_sessions_compact_rejects_active_session_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            store.add_messages(
                routed.session.id,
                [
                    ("user", "message 0"),
                    ("assistant", "message 1"),
                    ("user", "message 2"),
                ],
            )

            with gateway.session_lease(routed.session.id):
                with self.assertRaisesRegex(GatewayError, "active session"):
                    gateway.call(
                        "sessions.compact",
                        {
                            "key": routed.decision.route_key,
                            "maxLines": 1,
                        },
                        granted_scopes=frozenset({"gateway.admin"}),
                    )

            self.assertEqual(
                [(message.seq, message.content) for message in store.list_messages(routed.session.id, limit=None)],
                [(1, "message 0"), (2, "message 1"), (3, "message 2")],
            )

    def test_sessions_compaction_list_returns_checkpoint_archive_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            for index in range(5):
                role = "user" if index % 2 == 0 else "assistant"
                store.add_message(routed.session.id, role, f"message {index}")
            compact_result = gateway.call(
                "sessions.compact",
                {
                    "key": routed.decision.route_key,
                    "maxLines": 2,
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )

            result = gateway.call(
                "sessions.compaction.list",
                {"key": routed.decision.route_key},
            )
            listed = gateway.call("sessions.list", {"limit": "all"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["key"], routed.decision.route_key)
        self.assertEqual(result["sessionId"], routed.session.id)
        self.assertEqual(len(result["checkpoints"]), 1)
        checkpoint = result["checkpoints"][0]
        self.assertEqual(checkpoint["checkpointId"], compact_result["archived"])
        self.assertEqual(checkpoint["sessionKey"], routed.decision.route_key)
        self.assertEqual(checkpoint["sessionId"], routed.session.id)
        self.assertEqual(checkpoint["reason"], "manual")
        self.assertEqual(checkpoint["linesBefore"], 5)
        self.assertEqual(checkpoint["linesAfter"], 2)
        self.assertEqual(checkpoint["maxMessages"], 2)
        self.assertEqual(checkpoint["pruned"], 3)
        self.assertEqual(checkpoint["firstKeptEntryId"], "4")
        self.assertEqual(checkpoint["preCompaction"]["entryId"], "3")
        self.assertEqual(checkpoint["postCompaction"]["entryId"], "4")
        row = listed["sessions"][0]
        self.assertEqual(row["compactionCheckpointCount"], 1)
        self.assertEqual(row["latestCompactionCheckpoint"]["checkpointId"], checkpoint["checkpointId"])
        self.assertEqual(row["latestCompactionCheckpoint"]["reason"], "manual")
        json.dumps(result)

    def test_sessions_compaction_list_requires_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )

            with self.assertRaisesRegex(GatewayError, "sessions.compaction.list requires"):
                gateway.call("sessions.compaction.list", {})

    def test_sessions_compaction_get_returns_one_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            for index in range(4):
                store.add_message(routed.session.id, "user", f"message {index}")
            compact_result = gateway.call(
                "sessions.compact",
                {
                    "key": routed.decision.route_key,
                    "maxLines": 1,
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )

            result = gateway.call(
                "sessions.compaction.get",
                {
                    "key": routed.decision.route_key,
                    "checkpointId": compact_result["archived"],
                },
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["key"], routed.decision.route_key)
        self.assertEqual(result["sessionId"], routed.session.id)
        self.assertEqual(result["checkpoint"]["checkpointId"], compact_result["archived"])
        self.assertEqual(result["checkpoint"]["linesBefore"], 4)
        self.assertEqual(result["checkpoint"]["linesAfter"], 1)
        self.assertEqual(result["checkpoint"]["firstKeptEntryId"], "4")
        json.dumps(result)

    def test_sessions_compaction_get_requires_checkpoint_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayError, "checkpointId required"):
                gateway.call(
                    "sessions.compaction.get",
                    {"key": routed.decision.route_key},
                )

    def test_sessions_compaction_get_rejects_checkpoint_from_other_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            first = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            second = gateway.open_session(
                InboundAddress(channel="tui", sender_id="bob"),
                workspace_root=root,
            )
            for index in range(3):
                store.add_message(first.session.id, "user", f"first {index}")
            compact_result = gateway.call(
                "sessions.compact",
                {
                    "key": first.decision.route_key,
                    "maxLines": 1,
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )

            with self.assertRaisesRegex(KeyError, "Compaction checkpoint not found"):
                gateway.call(
                    "sessions.compaction.get",
                    {
                        "key": second.decision.route_key,
                        "checkpointId": compact_result["archived"],
                    },
                )

    def test_sessions_compaction_branch_requires_write_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.compaction.branch",
                    {
                        "key": routed.decision.route_key,
                        "checkpointId": "sqlite:event:1",
                    },
                    granted_scopes=frozenset({"gateway.read"}),
                )

    def test_sessions_compaction_branch_creates_child_from_pre_compaction_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
                provider="anthropic",
                model="claude-3-5-sonnet-latest",
            )
            for index in range(5):
                role = "user" if index % 2 == 0 else "assistant"
                store.add_message(routed.session.id, role, f"message {index}")
            store.patch_session_metadata(
                routed.session.id,
                state_patch={"reasoning_effort": "high"},
            )
            compact_result = gateway.call(
                "sessions.compact",
                {
                    "key": routed.decision.route_key,
                    "maxLines": 2,
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )
            store.add_message(routed.session.id, "assistant", "after compaction")

            result = gateway.call(
                "sessions.compaction.branch",
                {
                    "key": routed.decision.route_key,
                    "checkpointId": compact_result["archived"],
                },
                granted_scopes=frozenset({"gateway.write"}),
            )
            branched = store.get_session(result["sessionId"])
            branch_messages = store.list_messages(result["sessionId"], limit=None)
            source_messages = store.list_messages(routed.session.id, limit=None)
            branch_route = store.get_route(result["key"])

        self.assertTrue(result["ok"])
        self.assertEqual(result["sourceKey"], routed.decision.route_key)
        self.assertNotEqual(result["key"], routed.decision.route_key)
        self.assertEqual(result["entry"]["parentSessionKey"], routed.decision.route_key)
        self.assertEqual(result["checkpoint"]["checkpointId"], compact_result["archived"])
        self.assertEqual(branch_route.session_id, result["sessionId"])
        self.assertEqual(branched.provider, "anthropic")
        self.assertEqual(branched.model, "claude-3-5-sonnet-latest")
        self.assertEqual(branched.state["parent_session_key"], routed.decision.route_key)
        self.assertEqual(branched.state["source_session_id"], routed.session.id)
        self.assertEqual(branched.state["checkpoint_id"], compact_result["archived"])
        self.assertEqual(branched.state["reasoning_effort"], "high")
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
        self.assertEqual(
            [(message.seq, message.content) for message in source_messages],
            [(4, "message 3"), (5, "message 4"), (6, "after compaction")],
        )
        json.dumps(result)

    def test_sessions_compaction_branch_requires_checkpoint_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayError, "checkpointId required"):
                gateway.call(
                    "sessions.compaction.branch",
                    {"key": routed.decision.route_key},
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_sessions_compaction_branch_requires_route_key_not_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            session = store.create_session(workspace_root=root)

            with self.assertRaisesRegex(GatewayError, "requires key"):
                gateway.call(
                    "sessions.compaction.branch",
                    {
                        "sessionId": session.id,
                        "checkpointId": "sqlite:event:1",
                    },
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_sessions_compaction_restore_requires_admin_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
                gateway.call(
                    "sessions.compaction.restore",
                    {
                        "key": routed.decision.route_key,
                        "checkpointId": "sqlite:event:1",
                    },
                    granted_scopes=frozenset({"gateway.write"}),
                )

    def test_sessions_compaction_restore_rebinds_key_to_checkpoint_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
                provider="anthropic",
                model="claude-3-5-sonnet-latest",
            )
            for index in range(5):
                role = "user" if index % 2 == 0 else "assistant"
                store.add_message(routed.session.id, role, f"message {index}")
            store.patch_session_metadata(
                routed.session.id,
                state_patch={"reasoning_effort": "high"},
            )
            compact_result = gateway.call(
                "sessions.compact",
                {
                    "key": routed.decision.route_key,
                    "maxLines": 2,
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )
            store.add_message(routed.session.id, "assistant", "after compaction")

            result = gateway.call(
                "sessions.compaction.restore",
                {
                    "key": routed.decision.route_key,
                    "checkpointId": compact_result["archived"],
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )
            restored = store.get_session(result["sessionId"])
            restored_messages = store.list_messages(result["sessionId"], limit=None)
            route = store.get_route(routed.decision.route_key)
            checkpoints = gateway.call(
                "sessions.compaction.list",
                {"key": routed.decision.route_key},
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["key"], routed.decision.route_key)
        self.assertEqual(result["previousSessionId"], routed.session.id)
        self.assertNotEqual(result["sessionId"], routed.session.id)
        self.assertEqual(result["checkpoint"]["checkpointId"], compact_result["archived"])
        self.assertEqual(result["checkpoint"]["sessionId"], result["sessionId"])
        self.assertEqual(result["entry"]["sessionId"], result["sessionId"])
        self.assertEqual(route.session_id, result["sessionId"])
        self.assertEqual(restored.provider, "anthropic")
        self.assertEqual(restored.model, "claude-3-5-sonnet-latest")
        self.assertEqual(restored.state["restored_from_session_id"], routed.session.id)
        self.assertEqual(restored.state["restored_checkpoint_id"], compact_result["archived"])
        self.assertEqual(restored.state["reasoning_effort"], "high")
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
        self.assertEqual(checkpoints["checkpoints"][0]["checkpointId"], compact_result["archived"])
        self.assertEqual(checkpoints["checkpoints"][0]["sessionId"], result["sessionId"])
        json.dumps(result)

    def test_sessions_compaction_restore_rejects_active_session_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )
            for index in range(3):
                store.add_message(routed.session.id, "user", f"message {index}")
            compact_result = gateway.call(
                "sessions.compact",
                {
                    "key": routed.decision.route_key,
                    "maxLines": 1,
                },
                granted_scopes=frozenset({"gateway.admin"}),
            )

            with gateway.session_lease(routed.session.id):
                with self.assertRaisesRegex(GatewayError, "active session"):
                    gateway.call(
                        "sessions.compaction.restore",
                        {
                            "key": routed.decision.route_key,
                            "checkpointId": compact_result["archived"],
                        },
                        granted_scopes=frozenset({"gateway.admin"}),
                    )

            self.assertEqual(store.get_route(routed.decision.route_key).session_id, routed.session.id)
            self.assertEqual(
                [(message.seq, message.content) for message in store.list_messages(routed.session.id, limit=None)],
                [(3, "message 2")],
            )

    def test_sessions_compaction_restore_requires_checkpoint_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            gateway = AgentGateway(config=parse_agent_config({}, root), store=store)
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with self.assertRaisesRegex(GatewayError, "checkpointId required"):
                gateway.call(
                    "sessions.compaction.restore",
                    {"key": routed.decision.route_key},
                    granted_scopes=frozenset({"gateway.admin"}),
                )


class GatewayRuntimeTests(unittest.TestCase):
    def test_channel_health_policy_matches_openclaw_restart_decisions(self) -> None:
        policy = ChannelHealthPolicy(
            channel_id="chat",
            now=2_000_000,
            stale_event_threshold_ms=30 * 60_000,
            channel_connect_grace_ms=120_000,
        )

        self.assertEqual(evaluate_channel_health({"enabled": False}, policy).reason, "unmanaged")
        terminal = evaluate_channel_health({
            "running": False,
            "terminalDisconnect": True,
        }, policy)
        self.assertFalse(terminal.healthy)
        self.assertEqual(terminal.reason, "terminal-disconnect")
        self.assertEqual(evaluate_channel_health({"running": False}, policy).reason, "not-running")
        self.assertEqual(
            evaluate_channel_health({
                "running": True,
                "lastStartAt": policy.now - 10_000,
                "connected": False,
            }, policy).reason,
            "startup-connect-grace",
        )
        self.assertEqual(
            evaluate_channel_health({
                "running": True,
                "busy": True,
                "lastStartAt": policy.now - 200_000,
                "lastRunActivityAt": policy.now - 60_000,
            }, policy).reason,
            "busy",
        )
        self.assertEqual(
            evaluate_channel_health({
                "running": True,
                "activeRuns": 1,
                "lastStartAt": policy.now - 2_000_000,
                "lastRunActivityAt": policy.now - 26 * 60_000,
            }, policy).reason,
            "stuck",
        )
        self.assertEqual(
            evaluate_channel_health({
                "running": True,
                "lastStartAt": policy.now - 200_000,
                "connected": False,
            }, policy).reason,
            "disconnected",
        )
        self.assertEqual(
            evaluate_channel_health({
                "running": True,
                "connected": True,
                "lastStartAt": policy.now - 2_000_000,
                "lastTransportActivityAt": policy.now - 31 * 60_000,
            }, policy).reason,
            "stale-socket",
        )

    def test_channel_health_restart_reason_is_coarse_like_openclaw(self) -> None:
        policy = ChannelHealthPolicy(channel_id="chat", now=2_000_000)
        not_running = evaluate_channel_health({
            "running": False,
            "reconnectAttempts": 10,
        }, policy)
        disconnected = evaluate_channel_health({
            "running": True,
            "lastStartAt": policy.now - 200_000,
            "connected": False,
        }, policy)
        stale = evaluate_channel_health({
            "running": True,
            "connected": True,
            "lastStartAt": policy.now - 2_000_000,
            "lastTransportActivityAt": policy.now - 31 * 60_000,
        }, policy)
        stuck = evaluate_channel_health({
            "running": True,
            "busy": True,
            "lastRunActivityAt": policy.now - 26 * 60_000,
        }, policy)

        self.assertEqual(resolve_channel_restart_reason({"reconnectAttempts": 10}, not_running), "gave-up")
        self.assertEqual(resolve_channel_restart_reason({}, disconnected), "disconnected")
        self.assertEqual(resolve_channel_restart_reason({}, stale), "stale-socket")
        self.assertEqual(resolve_channel_restart_reason({}, stuck), "stuck")

    def test_lazy_service_deduplicates_concurrent_loaders(self) -> None:
        started = threading.Event()
        release = threading.Event()
        calls: list[int] = []

        def factory() -> object:
            calls.append(1)
            started.set()
            release.wait(timeout=2)
            return object()

        service = LazyService(factory)
        values: list[object] = []
        threads = [threading.Thread(target=lambda: values.append(service.get())) for _ in range(2)]
        for thread in threads:
            thread.start()
        self.assertTrue(started.wait(timeout=1))
        time.sleep(0.02)
        release.set()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(values), 2)
        self.assertIs(values[0], values[1])
        self.assertIs(service.peek(), values[0])
        self.assertEqual(service.status()["state"], "ready")

    def test_method_registry_enforces_uniqueness_readiness_and_scope(self) -> None:
        registry = GatewayMethodRegistry()
        method = GatewayMethod(name=" gateway.test ", handler=lambda params: params["value"])
        registry.register(method)

        with self.assertRaisesRegex(GatewayRuntimeError, "Duplicate"):
            registry.register(GatewayMethod(name="gateway.test", handler=lambda _params: None))
        with self.assertRaisesRegex(GatewayRuntimeError, "before ready"):
            registry.dispatch("gateway.test", {"value": 1}, ready=False)
        with self.assertRaisesRegex(GatewayRuntimeError, "requires scope"):
            registry.dispatch(
                "gateway.test",
                {"value": 1},
                ready=True,
                granted_scopes=frozenset(),
            )
        self.assertEqual(registry.dispatch("gateway.test", {"value": 7}, ready=True), 7)

    def test_method_registry_exposes_openclaw_style_policy_metadata(self) -> None:
        registry = GatewayMethodRegistry()
        registry.register(GatewayMethod(
            name="gateway.visible",
            handler=lambda _params: "visible",
            required_scopes=frozenset({"gateway.read"}),
            requires_ready=False,
        ))
        registry.register(GatewayMethod(
            name="gateway.hiddenWrite",
            handler=lambda _params: "hidden",
            required_scopes=frozenset({"gateway.write"}),
            control_write=True,
            advertise=False,
        ))

        self.assertEqual(
            registry.get_handler("gateway.visible")({}),  # type: ignore[misc]
            "visible",
        )
        self.assertEqual(registry.list_methods(), ("gateway.visible", "gateway.hiddenWrite"))
        self.assertEqual(registry.list_advertised_methods(), ("gateway.visible",))
        self.assertEqual(registry.get_scopes("gateway.hiddenWrite"), frozenset({"gateway.write"}))
        self.assertFalse(registry.is_startup_unavailable("gateway.visible"))
        self.assertTrue(registry.is_startup_unavailable("gateway.hiddenWrite"))
        self.assertTrue(registry.is_control_plane_write("gateway.hiddenWrite"))
        self.assertEqual(
            [method.name for method in registry.descriptors()],
            ["gateway.visible", "gateway.hiddenWrite"],
        )

    def test_extension_method_cannot_claim_control_plane_write(self) -> None:
        registry = GatewayMethodRegistry()
        with self.assertRaisesRegex(GatewayRuntimeError, "cannot claim"):
            registry.register(GatewayMethod(
                name="extension.write",
                handler=lambda _params: None,
                control_write=True,
                owner="extension",
            ))

    def test_runtime_state_factory_pins_and_releases_plugin_registry(self) -> None:
        release_pinned_plugin_registry()
        registry = PluginRegistry()
        runtime = create_gateway_runtime_state(plugin_registry=registry)

        try:
            self.assertIs(active_plugin_registry(), registry)
            self.assertIs(runtime.plugin_registry, registry)
            self.assertEqual(runtime.snapshot()["plugins"], registry.describe())
        finally:
            runtime.release_plugin_registry()

        self.assertIsNone(active_plugin_registry())

    def test_gateway_close_releases_pinned_plugin_registry(self) -> None:
        release_pinned_plugin_registry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )

            self.assertIsNotNone(active_plugin_registry())
            gateway.close()

        self.assertIsNone(active_plugin_registry())

    def test_gateway_owns_channel_health_monitor_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )

            self.assertFalse(gateway.health_monitor._stop_event.is_set())
            gateway.close()

        self.assertTrue(gateway.health_monitor._stop_event.is_set())

    def test_channel_lifecycle_rejects_stale_generation_and_breaks_crash_loop(self) -> None:
        manager = ChannelLifecycleManager(
            base_backoff_seconds=5,
            crash_loop_limit=2,
            random_value=lambda: 0.5,
        )
        manager.register("chat")
        first = manager.begin_start("chat")
        manager.stop("chat")
        self.assertFalse(manager.heartbeat("chat", "default", first))

        second = manager.begin_start("chat")
        self.assertTrue(manager.mark_failed("chat", "default", second, "first failure"))
        self.assertEqual(manager.snapshots()[0]["state"], "backoff")
        third = manager.begin_start("chat")
        self.assertTrue(manager.mark_failed("chat", "default", third, "second failure"))
        self.assertEqual(manager.snapshots()[0]["state"], "crash_loop")
        with self.assertRaisesRegex(GatewayRuntimeError, "crash-loop"):
            manager.begin_start("chat")

    def test_channel_lifecycle_store_tracks_start_task_abort_and_runtime(self) -> None:
        manager = ChannelLifecycleManager()
        manager.register("chat", "alerts")

        first = manager.begin_start("chat", "alerts")
        duplicate = manager.begin_start("chat", "alerts")
        starting = manager.lifecycle_store_snapshot()

        self.assertEqual(duplicate, first)
        self.assertEqual(starting["starting"], [{"channel": "chat", "account_id": "alerts"}])
        self.assertEqual(starting["aborts"], [{"channel": "chat", "account_id": "alerts"}])
        self.assertEqual(starting["tasks"], [])
        self.assertEqual(starting["runtimes"], [{"channel": "chat", "account_id": "alerts"}])

        self.assertTrue(manager.mark_running("chat", "alerts", first))
        running = manager.lifecycle_store_snapshot()
        runtime = manager.snapshots()[0]

        self.assertEqual(running["starting"], [])
        self.assertEqual(running["tasks"], [{"channel": "chat", "account_id": "alerts"}])
        self.assertTrue(runtime["task_active"])
        self.assertFalse(runtime["start_pending"])

        manager.stop("chat", "alerts")
        stopped = manager.lifecycle_store_snapshot()
        runtime = manager.snapshots()[0]

        self.assertEqual(stopped["starting"], [])
        self.assertEqual(stopped["tasks"], [])
        self.assertEqual(stopped["aborts"], [])
        self.assertEqual(stopped["runtimes"], [{"channel": "chat", "account_id": "alerts"}])
        self.assertTrue(runtime["abort_requested"])

    def test_gateway_rejects_a_second_parent_turn_for_the_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )
            routed = gateway.open_session(
                InboundAddress(channel="tui", sender_id="alice"),
                workspace_root=root,
            )

            with gateway.session_lease(routed.session.id):
                with self.assertRaisesRegex(GatewayError, "already active"):
                    with gateway.session_lease(routed.session.id):
                        self.fail("a second parent turn must not acquire the session lease")

    def test_routed_turn_runs_synchronously_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )
            message = InboundMessage(
                address=InboundAddress(channel="tui", sender_id="alice"),
                text="hello",
            )
            routed = gateway.open_session(message.address, workspace_root=root)

            result = gateway.run_routed_turn(
                message,
                routed,
                lambda session, inbound: f"{session.id}:{inbound.text}",
            )

        self.assertEqual(result, f"{routed.session.id}:hello")

    def test_routed_turn_rejects_async_background_handler(self) -> None:
        async def background_handler(_session: object, _message: object) -> str:
            return "later"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gateway = AgentGateway(
                config=parse_agent_config({}, root),
                store=SessionStore(root / "sessions.sqlite3"),
            )
            message = InboundMessage(
                address=InboundAddress(channel="tui", sender_id="alice"),
                text="hello",
            )
            routed = gateway.open_session(message.address, workspace_root=root)

            with self.assertRaisesRegex(GatewayError, "synchronously"):
                gateway.run_routed_turn(message, routed, background_handler)

    def test_gateway_and_skill_status_are_available_as_slash_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            config = parse_agent_config({}, root)
            gateway = AgentGateway(config=config, store=store)
            skills = discover_skill_catalog(root, config, home=root / "home")
            ctx = SimpleNamespace(
                gateway=gateway,
                skills=skills,
                agent_id="main",
                tool_allowlist=None,
                route_key=None,
            )

            gateway_text = _handle_local_command(ctx, "/gateway")
            skills_text = _handle_local_command(ctx, "/skills")

        self.assertIn("Agent control plane", gateway_text)
        self.assertIn("one active parent turn", gateway_text)
        self.assertIn("code-review", skills_text)

    def test_gateway_control_snapshot_adds_active_tui_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            config = parse_agent_config({}, root)
            gateway = AgentGateway(config=config, store=store)
            session = store.create_session(workspace_root=root)
            ctx = SimpleNamespace(
                gateway=gateway,
                session_id=session.id,
                agent_id="main",
                route_key=None,
                workspace_root=root,
                tool_allowlist=("grep", "read_path"),
            )

            snapshot = _gateway_control_snapshot(ctx)

        self.assertEqual(snapshot["overview"]["active_session"], session.id)
        self.assertEqual(snapshot["overview"]["workspace_root"], str(root))
        self.assertEqual(snapshot["overview"]["tool_policy"], "grep, read_path")


if __name__ == "__main__":
    unittest.main()
