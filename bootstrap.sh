#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
cargo build --release --manifest-path agent-rust/Cargo.toml

printf 'Ready. Run: %s/run-agent.sh --tui\n' "$ROOT"
