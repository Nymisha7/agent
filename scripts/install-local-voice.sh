#!/usr/bin/env bash
set -euo pipefail

VOICE_PACKAGE="${NYM_LOCAL_VOICE_PACKAGE:-speech-to-speech[websocket]}"

usage() {
  cat <<'EOF'
Install Nym's local Hugging Face voice backend.

Usage:
  scripts/install-local-voice.sh --python PATH
  scripts/install-local-voice.sh --pipx-package NAME

Environment:
  NYM_LOCAL_VOICE_PACKAGE  Package spec to install. Default: speech-to-speech[websocket]

The backend uses open-source Hugging Face models locally. Model weights are cached
by Hugging Face tooling on first use; no API key or token is required by this script.
EOF
}

if [ "$#" -ne 2 ]; then
  usage >&2
  exit 2
fi

case "$1" in
  --python)
    PYTHON="$2"
    if [ ! -x "$PYTHON" ]; then
      echo "Python interpreter not found or not executable: $PYTHON" >&2
      exit 1
    fi
    "$PYTHON" -m pip install "$VOICE_PACKAGE"
    ;;
  --pipx-package)
    PACKAGE="$2"
    python3 -m pipx inject --include-apps "$PACKAGE" "$VOICE_PACKAGE"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
