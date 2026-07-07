from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from nym_agent.main import _selected_model
from nym_agent.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
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


class ModelSelectionTests(unittest.TestCase):
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
