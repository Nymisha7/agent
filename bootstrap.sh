#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
if [ "${NYM_SKIP_LOCAL_VOICE:-0}" != "1" ]; then
  scripts/install-local-voice.sh --python "$ROOT/.venv/bin/python"
fi
cargo build --release --manifest-path agent-rust/Cargo.toml

printf 'Ready. Run: %s/.venv/bin/nym --tui\n' "$ROOT"
