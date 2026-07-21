from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest


class GatewayEntryPointTests(unittest.TestCase):
    def test_gateway_public_entrypoint_defers_implementation_import(self) -> None:
        code = "import sys; import agent.gateway; print('agent.gateway_impl' in sys.modules)"

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.stdout.strip(), "False")

    def test_start_gateway_loads_implementation_and_returns_gateway(self) -> None:
        code = textwrap.dedent(
            """
            import sys
            import tempfile
            from pathlib import Path

            from agent.config import parse_agent_config
            from agent.session_store import SessionStore
            import agent.gateway as gateway

            print("before", "agent.gateway_impl" in sys.modules)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                instance = gateway.start_gateway(
                    config=parse_agent_config({}, root),
                    store=SessionStore(root / "sessions.sqlite3"),
                )
                print("after", "agent.gateway_impl" in sys.modules, type(instance).__name__)
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.stdout.splitlines(), ["before False", "after True AgentGateway"])
