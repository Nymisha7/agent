

# Agent

Agent is a terminal coding agent with a Python agent core and a Rust/Ratatui interface.

## Run

Portable local run:

```bash
./bootstrap.sh
./run-agent.sh --tui
```

`run-agent.sh` resolves the project directory relative to itself, so the checkout
can be moved to another drive/path and still run after `./bootstrap.sh`.

Editable install:

```bash
python -m pip install -e .
cd agent-rust && cargo build && cd ..
agent --tui
```

File discovery and content search require `rg` (ripgrep) on `PATH`, matching
Goose's Developer-extension search backend.

Release wheels bundle the gateway, agent core, plugin SDK, LLM layer, built-in
channel plugins, skills, and a compiled `agent-rust` backend in `agent/bin/`.
Building a wheel runs `cargo build --release` and copies the executable into the
package. Source-only development builds can set `AGENT_SKIP_RUST_BUILD=1`, but
distributed wheels should not use that flag.

Type `/` inside the TUI to open the command menu. The main commands are `/model`,
`/install`, `/reasoning`, `/tools`, `/skills`, `/gateway`, `/status`, `/setup`,
`/help`, and `/exit`.

## Desktop and system capabilities

Ask natural questions such as `What devices are connected here?`.
Agent reports live USB, Bluetooth, storage, network, input, audio, display, camera,
printer, and power records visible to its runtime. Every category has a separate
availability state, so an unavailable host service is not reported as an empty device
list. Under WSL, Agent uses a read-only PowerShell Plug-and-Play bridge when available and
labels Windows-host records separately from devices directly visible inside WSL. If the
bridge is blocked or unavailable, the result says that the inventory is WSL-only.
Windows device classes that Agent has never seen before are retained under `Other` with
their native class and status instead of being filtered out.

Agent also exposes typed desktop actions for application launch, opening paths and web
pages, audio and brightness changes, Bluetooth/network state, storage eject, and
process termination. These operations always require approval for the exact target and
value. An approval is consumed after one attempted action. Results include before/after
state and a verification status; an unverified launch or unchanged state is never
reported as confirmed. Agent does not expose arbitrary shell execution as a desktop tool.

## Open-source models (no login)

Agent can use open-source models through local runtimes. These providers never open
an account page or ask for an Agent login:

| Provider | Default endpoint | Example |
| --- | --- | --- |
| Ollama | `http://localhost:11434/v1` | `/model ollama qwen3` |
| LM Studio | `http://localhost:1234/v1` | `/model lmstudio <loaded-model>` |
| llama.cpp | `http://localhost:8080/v1` | `/model llamacpp <loaded-model>` |
| vLLM | `http://localhost:8000/v1` | `/model vllm <served-model>` |
| LocalAI | `http://localhost:8080/v1` | `/model localai <loaded-model>` |

The runtime owns model downloads and storage; Agent discovers models from its local
model-list endpoint. Override endpoints with `AGENT_OLLAMA_BASE_URL`,
`AGENT_LMSTUDIO_BASE_URL`, `AGENT_LLAMACPP_BASE_URL`, `AGENT_VLLM_BASE_URL`, or
`AGENT_LOCALAI_BASE_URL`.

Selecting a local model checks the runtime and installed-model list first. If an
Ollama model is missing, preview it from the TUI with `/install ollama <model>`.
The preview shows parameter count, expected download size, recommended memory,
context window, and quantization before anything is downloaded. Confirm with
`/install ollama <model> --yes`. Press `Esc` or `Ctrl+C` during a download to stop
the current task without closing Agent. Agent reports an offline runtime separately from a missing
model. LM Studio, llama.cpp, vLLM, and LocalAI use their own installed CLI for
downloads and local server startup; Agent verifies that the selected model appears in
the provider API before reporting it ready.
The `/install` picker contains open-source/open-weight models for Ollama, LM Studio,
llama.cpp, vLLM, and LocalAI. Each download is stored and served locally by the
selected provider runtime, not by Agent.

For models that expose configurable reasoning effort, use `/reasoning` and choose
`minimal`, `low`, `medium`, or `high`. Agent shows concise activity and tool results;
raw private chain-of-thought is not displayed.

Hosted providers remain available and request an API key or cloud credentials only
when their provider requires it.

## Subagent safety model

Agent supports only bounded discovery subagents. A discovery child is a fresh,
in-memory agent instance with no parent conversation/session state and only these
workspace tools: path resolution/status, listing/reading, tree inspection, glob,
and grep. Mutation, shell, desktop, device, secret-scan, approval, and nested-agent
tools are absent from its registry.

Subagents execute synchronously and sequentially. A child must finish and return its
evidence before the parent resumes; it cannot run in the background. The parent is
the only agent allowed to propose or perform edits. TUI turns also use a per-session
process lease plus a single Rust bridge slot, so a second active turn cannot replace
the current bridge.

## Local control plane, routing, and skills

Agent has a Python control-plane layer around the existing planner rather than a separate
Node.js gateway. It normalizes registered channel messages, applies deterministic
bindings, maps them to durable SQLite sessions, emits synchronous lifecycle hooks, and
then hands the turn to the existing agent runtime. Hooks are observers: a failing hook
is isolated and cannot change a routing decision or expand tool permissions. Channel
adapters are registered explicitly by trusted Python code; Agent does not import arbitrary
plugin code from a workspace.

The control plane also owns an inspectable runtime state, a scoped RPC method registry,
and generation-safe channel lifecycle records. Duplicate RPC names fail at registration;
startup-gated methods cannot run early; extension-owned methods cannot claim control-plane
write access. Channel state rejects stale generations, uses bounded exponential retry
delays, and suppresses restart after a crash loop. Heavy optional services can use Agent's
thread-safe lazy loader, which deduplicates concurrent loads and exposes unloaded/loading/
ready/failed state without forcing initialization.

Configuration is strict JSON. Agent loads `~/.config/agent/config.json` first and overlays
`<workspace>/.agent/config.json`. `AGENT_CONFIG` or `--config` selects one explicit file.
Agent profiles are routing and policy profiles, not concurrently editing workers. Their
optional `skills` and `tools` arrays replace inherited defaults; an empty array disables
that capability, and omitted fields inherit it.

```json
{
  "agents": {
    "default": "main",
    "defaults": {"skills": ["review"]},
    "list": [
      {"id": "main"},
      {"id": "docs", "skills": ["docs-search"], "tools": ["grep", "read_path"]},
      {"id": "locked", "skills": [], "tools": []}
    ]
  },
  "session": {"default_scope": "per-sender"},
  "bindings": [
    {
      "agent": "docs",
      "scope": "shared",
      "match": {
        "channel": "chat",
        "account_id": "*",
        "peer": {"kind": "channel", "id": "documentation"}
      }
    }
  ],
  "skills": {
    "extra_dirs": ["./company-skills"],
    "max_loaded": 32,
    "max_instruction_chars": 24000
  }
}
```

Bindings are matched in this order: exact peer, guild, team, exact account, wildcard
account, then the default agent. Session scopes are `per-sender`, `shared`, and `global`.
Routes are durable within an agent/workspace boundary, so the same canonical route
resumes the same SQLite session after a restart without leaking context into another
project. For a direct routed CLI/TUI session, for example:

```bash
agent --tui --channel chat --sender-id alice
```

Skills are `SKILL.md` instruction packages. Agent discovers them by name from these roots,
with the first occurrence winning:

1. `<workspace>/skills`
2. `<workspace>/.agents/skills`
3. `~/.agents/skills`
4. `~/.local/share/agent/skills`
5. bundled Agent skills
6. `skills.extra_dirs`

```markdown
---
name: review
description: Review a change using project conventions
tools: ["grep", "read_path", "language_server"]
requires_bins: ["git"]
---
Inspect the changed code and its callers before reporting findings.
```

The model sees only the bounded skill catalog initially and must call `load_skill` for
matching instructions. A skill cannot add a tool, escape the workspace, bypass an
approval, or override Agent's core safety prompt. `/skills` shows loaded and skipped
skills. In the Ratatui interface, `/gateway` opens a read-only control-plane overlay with
Overview, Routes, Bindings, Channels, Sessions, and RPC Methods tabs. The overlay is not
stored as a chat message. Use Tab or Left/Right to switch tabs, Up/Down or PgUp/PgDn to
scroll, R to refresh, and Esc to close. The line-oriented CLI keeps a compact text status.

The built-in local TUI adapter is registered automatically. External messaging adapters
must be supplied by trusted Python integration code and are shown as registered, running,
backing off, stopped, or crash-loop suppressed according to their real lifecycle state.
Agent does not report a cron scheduler or external channel as running unless a persistent
host has actually started it.

## Repository layout

This repository contains two parts:

- `agent/`: the Python agent
- `agent-rust/`: the Rust backend


Build the Rust backend from `agent-rust/` with:

```bash
cargo build
```

## Notes

The repository is organized as a combined workspace-style layout so both projects live in one place.
