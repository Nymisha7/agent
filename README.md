

# Agent

Agent is a terminal coding agent with a Python agent core and a Rust/Ratatui interface.

## Run

WSL install from Git:

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 python3-venv python3-pip ripgrep pipx
curl -fsSL https://raw.githubusercontent.com/Nymisha7/agent/main/scripts/install-wsl.sh | bash
nym --tui
```

The WSL installer builds the bundled Rust backend from source by default. Set
`NYM_PREBUILT_WHEEL=1` only when intentionally installing a matching release wheel.

If installing from a local checkout instead of GitHub:

```bash
./scripts/install-wsl.sh --path "$PWD"
nym --tui
```

The WSL installer uses `pipx` by default and builds the bundled Rust backend during
installation. To install into a checkout-local `.venv` instead, run:

```bash
./scripts/install-wsl.sh --path "$PWD" --venv
```

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
nym --tui
```

Build a wheel for distribution:

```bash
./scripts/build-wheel.sh
python -m pip install --force-reinstall dist/agent-*.whl
nym --tui
```

File discovery and content search require `rg` (ripgrep) on `PATH`, matching
Goose's Developer-extension search backend.

Release wheels bundle the gateway, agent core, plugin SDK, LLM layer, built-in
channel plugins, skills, and a compiled `agent-rust` backend in `agent/bin/`.
Building a wheel runs `cargo build --release` and copies the executable into the
package. Source-only development builds can set `AGENT_SKIP_RUST_BUILD=1`, but
distributed wheels should not use that flag.

The TUI keeps file sharing in the composer: click `+` or press Ctrl+O/F4 to add an
attachment, click the microphone or press F5 to dictate, and use Ctrl+V to paste text
or a clipboard image. Selected files appear beside the message and are sent with it.
Type `/` to open the secondary command menu
for `/model`, `/install`, `/reasoning`, `/skills`, `/gateway`, `/status`, `/setup`,
`/help`, and `/exit`.

## Voice

Voice setup is optional and never opens an API-key prompt. The microphone is active
only when Nym finds a recorder plus a transcription path; otherwise it stays muted and
explains the missing local capability when selected. On WSL, the installer provides
`ffmpeg` for microphone capture and `espeak-ng` for local speech playback.

Nym reuses an existing `OPENAI_API_KEY` for OpenAI-compatible transcription and speech.
`AGENT_VOICE_API_KEY` and `AGENT_VOICE_BASE_URL` can instead point voice at a separate
OpenAI-compatible service. Models and voice are configurable with
`AGENT_VOICE_STT_MODEL`, `AGENT_VOICE_TTS_MODEL`, and `AGENT_VOICE_TTS_VOICE`.

For an entirely local setup, provide commands without a key: `AGENT_VOICE_RECORDER`
must contain `{output}`, `AGENT_VOICE_STT_COMMAND` must contain `{input}`, and
`AGENT_VOICE_TTS_COMMAND` must contain `{input}` or `{text}` as a separate argument.
Nym invokes those commands directly rather than through a shell. Set
`AGENT_TTS_ENABLED=1` to speak completed replies in the background; it is off by
default. `AGENT_VOICE_RECORD_SECONDS` bounds a microphone turn (default 20 seconds),
and `AGENT_VOICE_ENABLED=0` disables the feature entirely.

Realtime speech backends can be used as an optional voice transport without making
them the agent brain. To use the public Hugging Face Space, set
`AGENT_VOICE_PROVIDER=huggingface`; Nym will use
`https://huggingface.co/spaces/smolagents/hf-realtime-voice` and will not require an
OpenAI key for transcription. You can also set `AGENT_VOICE_MODE=realtime` and
`AGENT_REALTIME_VOICE_URL` to a direct `wss://` endpoint, an HTTP server that returns
a `connect_url` from `/session` or `/api/session`, or a Hugging Face Space page URL such as
`https://huggingface.co/spaces/smolagents/hf-realtime-voice`. If the realtime session
endpoint requires authentication, set `AGENT_REALTIME_VOICE_API_KEY`; for Hugging Face
Spaces, existing `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, or `HUGGING_FACE_HUB_TOKEN`
values are also used as bearer tokens. Nym sends microphone audio through that
OpenAI-Realtime-style WebSocket path, receives the transcript, and submits the text
through the normal Agent runtime so desktop actions, session memory, and approval
policy stay exactly the same as typed commands. Advanced servers can override the
default `session.update` payload with `AGENT_REALTIME_VOICE_SESSION_UPDATE_JSON`.

## Desktop and system capabilities

### Strict hosted-model data boundaries

When using a hosted model, Nym keeps desktop observations purpose-limited: it shares
only app/process identifiers needed to act. Window titles, chat labels, application
paths, UI text, clipboard contents, device labels, and desktop downloads stay local.
Your typed prompt, messages in the active conversation, explicitly attached files, and
the file/tool content needed to complete a request may still be sent to the configured
model provider. Use a local model when no data may leave the PC.

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
process termination. Launching an application and closing an observed window run
immediately; all other desktop changes require approval for the exact target and value.
An approval is consumed after one attempted action. Results include before/after state
and a verification status; an unverified launch or unchanged state is never reported as
confirmed. Agent does not expose arbitrary shell execution as a desktop tool.

Intent selection belongs to the model and the full conversation, not English keyword
routers. Runtime code validates structural facts: exact tool schemas, workspace
boundaries, observed window/element identifiers, process IDs returned by observation,
approval records, and tool receipts. Desktop and workspace resolvers preserve the
model-supplied query instead of deleting words such as application or project. The Rust
TUI consumes typed command and configuration states from Python; human-readable status
text is display-only and is never parsed to decide whether a key, endpoint, install, or
other setup action is required.

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

Implemented hosted providers appear in the model picker and request an API key only
when required. Providers without a working transport are not advertised as selectable.

## Local data and retention

API keys entered through Agent are encrypted before being written to the private
configuration directory. Existing `credentials.json` files are migrated on first use.
For managed deployments, provide `AGENT_CREDENTIAL_ENCRYPTION_KEY` from the operating
system credential service or company secret manager so the encryption key is not kept
beside the encrypted data.

Sessions and attachment snapshots are private to the current OS user. Attachment
retention defaults to 30 days and a 1 GiB total cache. Configure these limits with
`AGENT_ATTACHMENT_RETENTION_DAYS` and `AGENT_ATTACHMENT_STORE_MAX_BYTES`; individual
file, text-input, and image-input limits use `AGENT_MAX_ATTACHMENT_BYTES`,
`AGENT_MAX_TEXT_ATTACHMENT_BYTES`, and `AGENT_MAX_IMAGE_ATTACHMENT_BYTES`.
`AGENT_PROTECTED_PATHS` can add platform- or company-specific filesystem roots that
must never be used as broad external tool targets.

## Parallel subagent safety model

Subagent delegation is a normal model tool decision inside the main agent loop. Agent
does not run a separate orchestration classifier before every message. The main model
receives `parallel_subagents` beside its other tools and either answers directly, uses
ordinary tools, or invokes one parallel batch when multiple independent, bounded
workstreams would materially improve the task. This follows the model-invoked task-tool
pattern used by OpenCode, Goose, Codex, and Claude Code. Singleton and sequential
subagent execution remain intentionally unsupported.

Every child is a fresh in-memory agent with its own model/tool loop and no parent
conversation/session state. It receives read/inspection tools and, when the parent
declares `owns` directories, `write_file` and `edit_file` limited to those exact
workspace-relative scopes. Ownership is normalized before spawning; workspace escape,
symlinks, protected/generated directories, whole-workspace ownership, and overlapping
child scopes are rejected. A child without `owns` remains read-only. Delete, arbitrary
shell, desktop control, messaging, secret-scan, approval, and nested-agent tools remain
absent. The coordinator records the plan, ownership, changed paths, and result of every
child in `.agent/parallel-work.md`; the parent handles cross-cutting integration,
destructive actions, approvals, and final verification.

Concurrency defaults to four workers and is configurable from 2–8 with
`AGENT_MAX_PARALLEL_SUBAGENTS`. Each worker receives eight model steps by default;
set `AGENT_SUBAGENT_MAX_STEPS` from 2–20 for broader or narrower independent work.
Set `AGENT_PARALLEL_WORK_FILE` to another workspace-relative path to move the shared
ledger. The TUI tracks each run and task by structured IDs and renders queued, running,
complete, and failed states instead of relying on summary text. TUI turns still use a
per-session process lease, so a second parent turn cannot replace the current bridge.

Primary implementation references:

- [OpenCode task tool](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/tool/task.ts)
- [Goose subagent execution](https://github.com/aaif-goose/goose/blob/main/crates/goose/src/agents/subagent_handler.rs)
- [Codex subagent documentation](https://developers.openai.com/codex/multi-agent)
- [Claude Code subagents](https://code.claude.com/docs/en/sub-agents)

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
that capability, and omitted fields inherit it. Language-server access is opt-in because
its current symbol-navigation API is most useful for focused code-intelligence profiles;
include `language_server` in a profile's `tools` array to expose it to that model.

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
