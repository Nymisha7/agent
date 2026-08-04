#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${NYM_REPO_URL:-https://github.com/Nymisha7/agent.git}"
INSTALL_ROOT="${NYM_INSTALL_ROOT:-$HOME/.local/share/nym}"
INSTALL_MODE="${NYM_INSTALL_MODE:-pipx}"

usage() {
  cat <<'EOF'
Install nym inside WSL.

Usage:
  scripts/install-wsl.sh [--repo URL] [--path DIR] [--venv]

Options:
  --repo URL   Git repository to clone and install from.
  --path DIR   Install from an existing checkout instead of cloning.
  --venv       Install into .venv inside the checkout instead of pipx.

Environment:
  NYM_REPO_URL       Default repository URL.
  NYM_INSTALL_ROOT   Clone destination. Default: ~/.local/share/nym
  NYM_INSTALL_MODE   pipx or venv. Default: pipx
  NYM_PREBUILT_WHEEL Set to 0 to compile the Rust backend from source.
EOF
}

SOURCE_PATH=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo)
      REPO_URL="${2:?--repo requires a URL}"
      shift 2
      ;;
    --path)
      SOURCE_PATH="${2:?--path requires a directory}"
      shift 2
      ;;
    --venv)
      INSTALL_MODE="venv"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

normalize_repo_url() {
  case "$REPO_URL" in
    git@github.com:*)
      REPO_URL="https://github.com/${REPO_URL#git@github.com:}"
      ;;
    ssh://git@github.com/*)
      REPO_URL="https://github.com/${REPO_URL#ssh://git@github.com/}"
      ;;
  esac
}

install_apt_deps() {
  if ! need_cmd apt-get; then
    return
  fi
  local missing=()
  for cmd in python3 git rg curl; do
    if ! need_cmd "$cmd"; then
      missing+=("$cmd")
    fi
  done
  if [ "$INSTALL_MODE" = "pipx" ] && ! need_cmd pipx; then
    missing+=("pipx")
  fi
  if ! python3 -m venv --help >/dev/null 2>&1; then
    missing+=("python3-venv")
  fi
  if [ "${#missing[@]}" -eq 0 ]; then
    return
  fi
  if ! need_cmd sudo; then
    echo "Missing dependencies: ${missing[*]}" >&2
    echo "Install them with apt, then rerun this script." >&2
    exit 1
  fi
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip git curl ripgrep pipx
}

ensure_rust() {
  # Ubuntu's packaged Cargo is often older than the lockfile format used by this
  # project. Keep the Rust toolchain in the user's home directory via rustup.
  if ! need_cmd rustup; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  fi
  "$HOME/.cargo/bin/rustup" toolchain install stable --profile minimal
  "$HOME/.cargo/bin/rustup" default stable
  export PATH="$HOME/.cargo/bin:$PATH"
}

prebuilt_wheel() {
  if [ "${NYM_PREBUILT_WHEEL:-1}" = "0" ] || [ "$(uname -m)" != "x86_64" ]; then
    return 1
  fi
  local wheel
  wheel=$(find packages -maxdepth 1 -type f -name 'agent-*-py3-none-linux_x86_64.whl' -print -quit 2>/dev/null || true)
  [ -n "$wheel" ] || return 1
  printf '%s\n' "$wheel"
}

checkout_repo() {
  if [ -n "$SOURCE_PATH" ]; then
    cd "$SOURCE_PATH"
    return
  fi
  if [ -d "$INSTALL_ROOT/.git" ]; then
    git -C "$INSTALL_ROOT" pull --ff-only
  else
    mkdir -p "$(dirname "$INSTALL_ROOT")"
    git clone "$REPO_URL" "$INSTALL_ROOT"
  fi
  cd "$INSTALL_ROOT"
}

install_with_pipx() {
  if ! need_cmd pipx; then
    python3 -m pip install --user --upgrade pipx
    python3 -m pipx ensurepath || true
  fi
  local package_source="."
  if wheel=$(prebuilt_wheel); then
    package_source="$wheel"
    echo "Installing bundled WSL wheel; Rust build is not needed."
  else
    ensure_rust
  fi
  python3 -m pipx install --force "$package_source"
  echo "Installed. If 'nym' is not on PATH yet, restart the shell or run:"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
}

install_with_venv() {
  local package_source="."
  if wheel=$(prebuilt_wheel); then
    package_source="$wheel"
    echo "Installing bundled WSL wheel; Rust build is not needed."
  else
    ensure_rust
  fi
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install "$package_source"
  mkdir -p "$HOME/.local/bin"
  ln -sfn "$(pwd)/.venv/bin/nym" "$HOME/.local/bin/nym"
  echo "Installed nym at $HOME/.local/bin/nym"
}

install_apt_deps
normalize_repo_url
checkout_repo

case "$INSTALL_MODE" in
  pipx)
    install_with_pipx
    ;;
  venv)
    install_with_venv
    ;;
  *)
    echo "Unsupported NYM_INSTALL_MODE: $INSTALL_MODE" >&2
    exit 2
    ;;
esac

echo "Run: nym --tui"
