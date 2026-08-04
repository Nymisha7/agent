#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_VENV="$(mktemp -d)"
cleanup() {
  rm -rf "$BUILD_VENV"
}
trap cleanup EXIT

python3 -m venv "$BUILD_VENV"
"$BUILD_VENV/bin/python" -m pip install --upgrade pip build
"$BUILD_VENV/bin/python" -m build --wheel

echo "Wheel written to dist/"
