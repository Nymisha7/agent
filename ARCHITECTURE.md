# Nym Architecture Overview

Nym is a local agent runtime with a terminal user interface. In simple terms, it
is a conversation app that can also inspect files, edit code, run carefully
bounded tools, manage sessions, and optionally listen or speak. The user sees a
TUI, but most of the intelligence lives in the Agent runtime: a Python
orchestrator that prepares context, talks to the selected model, executes tools,
persists state, and streams updates back to the interface.

For non-technical readers: think of Nym as a small local team.

- The TUI is the front desk. It shows the conversation, accepts typing, files,
  paste, copy, and microphone actions.
- The Agent runtime is the coordinator. It decides what context is needed,
  asks the model for the next step, checks safety rules, and records what
  happened.
- The model is the reasoning partner. It reads the request and calls tools when
  it needs real information or needs to make a change.
- The tools are the hands. They read files, search code, edit files, inspect the
  desktop, or perform other host actions.
- The session store is the memory. It keeps messages, events, routes,
  attachments, token usage, and durable agent state.

For engineers: Nym is split into a Python control plane and a Rust execution/UI
plane. Python owns orchestration, session state, provider setup, model calls,
policy, skills, routing, and high-level tool dispatch. Rust owns the Ratatui
frontend and the fast native worker used for filesystem, search, desktop, and
host primitives. The boundary between them is an explicit JSON bridge.

## Top-Level Shape

```text
User
  |
  | keyboard / mouse / paste / mic / files
  v
Rust TUI (agent-rust tui)
  |
  | JSON bridge subprocess calls
  v
Python Agent runtime (agent.main)
  |
  +--> Session store (SQLite + attachment cache)
  +--> LLM client (OpenAI, compatible, local providers, etc.)
  +--> Planner/tool loop (agent.planner)
  +--> Tool registry and policy (agent.tools, agent.policy)
  +--> Rust worker (agent-rust serve)
  +--> Gateway/control plane (agent.gateway_impl)
  +--> Optional voice backend (agent.voice)
```

The important design choice is that the TUI does not directly call the model or
mutate the workspace. It asks the Python runtime to do that through bridge
commands. This keeps one source of truth for session state, approvals, model
configuration, credentials, agent naming, and tool policy.

## Main Components

### 1. CLI Entrypoint

The installed commands are `nym` and `agent`; both enter `agent.main:main`.
The CLI parses arguments, loads persisted credentials, opens the session store,
creates or resumes a session, builds an `AppContext`, and then either starts:

- a standard terminal REPL, or
- the Rust TUI when `--tui` is used.

The `AppContext` is the runtime bundle for a session. It contains the workspace
root, Rust binary path, selected model provider, language server manager,
session state, SQLite store, config, skills, gateway, agent profile, display
name, pending attachments, and debug settings.

### 2. Agent Runtime

The Agent runtime is the heart of Nym. It lives primarily in:

- `agent/main.py`
- `agent/planner.py`
- `agent/llm.py`
- `agent/tools.py`
- `agent/tool_registry.py`
- `agent/session_store.py`

Its responsibilities are:

- load configuration and user preferences;
- create an LLM client for the selected provider and model;
- gather stored context and recent conversation history;
- build the available tool list;
- apply safety and approval policy;
- execute the model/tool loop;
- persist messages, events, token usage, attachments, pending approvals, and
  agent state;
- stream intermediate events back to the UI.

This is the part meant by "Agent runtime": it is not just a model call. It is
the durable local execution environment that wraps the model with memory,
tools, policy, context, and UI bridge contracts.

### 3. Rust TUI

The Rust TUI lives in `agent-rust/src/main.rs` and is launched by Python with:

```text
agent-rust tui --python <python> --repo-root <repo> --session-id <id>
```

It handles terminal concerns:

- drawing the conversation, composer, status line, attachments, approvals,
  model palette, gateway overlay, and microphone button;
- keyboard shortcuts such as paste, copy, attach, send, cancel, and F5 voice;
- mouse handling for attachment clicks and transcript drag-copy;
- background subprocess management for bridge calls;
- stream rendering while a turn is active.

The Rust TUI has a local UI state, but the authoritative conversation state
comes from `BridgeSnapshot` responses produced by Python. For example, the
custom agent name flows from Python as `snapshot.agent_name`; Rust uses it for
the top badge and assistant message headers.

### 4. Python-to-Rust Bridge

There are two Rust/Python boundaries:

1. TUI bridge: Rust calls Python bridge commands.
2. Tool worker: Python calls Rust worker commands.

The TUI bridge is user-interface oriented. Rust launches Python with
`--tui-bridge` commands such as:

- `snapshot`: return the current session, messages, approvals, voice status,
  and agent name;
- `stream-submit`: submit a user prompt and stream frames back to the TUI;
- `complete`: return slash-command and palette completions;
- `approve` / `deny`: resolve a pending tool approval;
- `gateway`: return gateway/control-plane status;
- `voice-record`: record and transcribe a bounded microphone turn;
- `voice-speak`: speak completed output when TTS is enabled.

The tool worker bridge is execution oriented. Python keeps a Rust `serve`
worker process alive through `RustTools`. Requests are JSON envelopes sent over
stdin/stdout, not raw shell strings. This avoids putting sensitive payloads in
argv and gives the Python runtime structured success/error responses.

## End-to-End Prompt Pipeline

This is what happens when a user sends a message in the TUI.

```text
1. User types a prompt or uses mic/files/paste.
2. Rust TUI sends `stream-submit` to Python.
3. Python loads the session and builds AppContext.
4. Python emits a submitted frame so the TUI can render immediately.
5. Python checks whether the prompt is a local command.
6. If it is not local, Python calls handle_prompt -> run_agent.
7. run_agent builds model instructions, context, tools, policy, and messages.
8. The model either answers directly or requests tool calls.
9. Python validates and executes tool calls through the registry.
10. Rust worker performs fast native host/file operations when needed.
11. Python sanitizes tool observations and feeds them back to the model.
12. The loop repeats until the model produces a final answer.
13. Python saves messages/events/tokens/state in the session store.
14. Python emits a final bridge frame.
15. Rust updates the transcript, status, cost, approvals, and optional TTS.
```

The user experiences this as "I ask, Nym works, then answers." Internally it is
a controlled loop where every live fact comes from a tool, every risky action
passes policy, and every durable outcome is recorded.

## Agent Runtime Internals

### Context Construction

`build_context` creates the runtime view for one session. It resolves:

- workspace root and search roots;
- agent config from user and workspace config files;
- selected agent profile and tool allowlist;
- skill catalog;
- Rust binary path;
- persisted session state;
- stored context from previous messages/events;
- LLM provider/model;
- pending attachments;
- gateway instance;
- custom display name.

This keeps turn execution deterministic: the planner receives a complete
runtime object rather than repeatedly rediscovering configuration.

### Model Client

`LLMClient` normalizes the selected provider and model. It supports hosted,
OpenAI-compatible, and local-provider modes. It also records configuration
state, so the TUI can show setup notices instead of failing obscurely.

Examples of provider families:

- hosted OpenAI-compatible APIs such as OpenAI, Groq, OpenRouter, DeepSeek, GLM;
- local OpenAI-compatible servers such as Ollama, LM Studio, llama.cpp, vLLM,
  and LocalAI;
- placeholder/unavailable transports with explicit configuration errors.

The model is never given direct filesystem or desktop authority. It only sees
tool schemas and asks the runtime to execute them.

### Planner and Tool Loop

`run_agent` is the main reasoning loop. It builds instructions, appends
session context, passes available tool schemas to the model, receives responses,
and executes requested tools.

Key properties:

- The prompt tells the model to use tools for live workspace, filesystem,
  process, desktop, and device facts.
- Unknown tools are guarded so the model cannot loop forever on invalid names.
- Repeated identical tool calls are detected and blocked when they stop adding
  evidence.
- Tool results are sanitized before returning to the model.
- Mutations are verified after execution.
- Failures can be carried across turns as compact recovery receipts.
- Parallel subagents can be invoked as a normal tool for independent work, not
  as a hidden preflight phase.

The planner's job is not only "call the model"; it enforces a disciplined
interaction between model reasoning and real host state.

### Tool Registry

The tool registry is a typed catalog of capabilities. Each tool has:

- a schema shown to the model;
- a Python handler;
- a default-enabled flag;
- execution through a shared `ToolContext`.

The `ToolContext` contains the workspace root, approved external roots,
language server manager, Rust tools, optional skill catalog, and optional
parallel subagent runner.

Tools are intentionally grouped by capability: filesystem search/read/write,
language server operations, desktop observations/actions, devices, process
listing, attachments, skills, and parallel subagents.

### Policy and Approvals

Nym separates model intention from runtime permission.

For low-risk reads inside the workspace, the runtime can execute directly.
For sensitive actions, the runtime creates structured approval requests. The
TUI shows those approvals and sends `approve` or `deny` back through the bridge.

Examples of controlled areas:

- deletion;
- external reads/writes outside approved roots;
- many desktop mutations;
- system commands;
- actions requiring a concrete observed desktop target.

Approvals are stored in session state so the UI and runtime agree about what is
pending. Orphaned approvals are expired when there is no active turn.

## Data and Memory

Nym stores durable state in a SQLite session store. The store tracks:

- projects and workspaces;
- sessions;
- messages;
- attachments;
- events;
- channel routes;
- token usage and cost;
- session-level agent state.

Attachments are not copied into prompts blindly. They are imported into a
private attachment cache, tracked by metadata, bounded by size/quota, and
rehydrated only when needed. The cache has retention and quota maintenance.

Credentials are loaded into process environment for provider setup but are not
written into conversation history. Preferences such as the agent display name,
paste/copy shortcuts, and mouse capture live under the user's config directory.

## Voice Pipeline

Voice is optional and backend-only. The TUI shows a microphone affordance, but
the actual capability decision happens in Python.

```text
Mic click or F5
  |
  v
Rust TUI bridge command: voice-record
  |
  v
Python voice backend
  |
  +--> recorder: custom command, ffmpeg, or arecord
  +--> STT: custom command or OpenAI-compatible transcription
  |
  v
Transcribed text returned to composer
```

Text-to-speech follows the reverse direction:

```text
Final assistant answer
  |
  v
Rust optionally launches voice-speak in background
  |
  v
Python voice backend
  |
  +--> custom TTS command
  +--> local espeak/espeak-ng
  +--> OpenAI-compatible speech + playback command
```

Voice availability is detected without touching the microphone or network.
The UI does not ask users for voice API keys. The backend reuses existing
`OPENAI_API_KEY` or optional voice-specific environment variables, or it can run
fully local custom commands.

Realtime voice is modeled as another transport, not another agent. When
`AGENT_VOICE_MODE=realtime` is enabled, Nym can send recorded microphone audio
to an OpenAI-Realtime-style WebSocket server and use the returned transcript as
the user's command. The transcript still enters the normal Agent runtime, so
desktop actions, approvals, memory, events, and tool policy remain identical to
typed text. This is the right integration shape for open realtime speech
projects: reuse the voice protocol, but do not let an external speech app make
desktop decisions outside Nym.

## Gateway and Agent Profiles

The gateway is Nym's local control plane for routing messages from channels to
sessions and agent profiles. It is designed so future channel adapters can send
normalized inbound messages into the same Agent runtime.

Core ideas:

- an inbound address identifies channel, account, sender, peer, guild, or team;
- config bindings route addresses to agent profiles;
- each route maps to a durable session;
- agent profiles can restrict tools and skills;
- lifecycle hooks observe events without changing routing decisions;
- channel health and runtime method metadata are exposed to the TUI gateway
  overlay.

For non-technical users, this means Nym can eventually support more entry
points than the local terminal while keeping the same memory, policy, and tool
engine.

For engineers, it means routing, session identity, channel normalization, and
runtime status are modeled separately from the core planner.

## Installation and Packaging

The WSL installer:

1. installs system dependencies when apt is available;
2. checks out or updates the repo;
3. installs Rust via rustup when building from source;
4. installs the Python package through pipx or a local venv;
5. ensures `~/.local/bin` is on PATH;
6. prompts for the initial agent display name;
7. prints the `nym --tui` entry command.

The Python package exposes both `agent` and `nym` console scripts. Package data
includes prompts, skills, and the bundled Rust binary path when built.

## Runtime State Flow

The runtime can be understood as three synchronized states:

- UI state: current composer text, scroll position, visible palette,
  recording state, selected approval, and transient status.
- Session state: messages, events, approvals, attachments, token usage, routes,
  and durable planner memory.
- Host state: files, processes, desktop windows, devices, audio capabilities,
  and language servers observed through tools.

The TUI can redraw at any time from UI state plus the latest bridge snapshot.
The model can only reason over state that the runtime provides. Tools observe or
change host state and return structured observations. Session state records the
conversation and important runtime events.

## Safety Model

Nym's safety model relies on several layers:

- Workspace boundaries: filesystem tools are rooted and explicit about external
  access.
- Structured tools: the model requests named tools with typed arguments instead
  of arbitrary shell access.
- Runtime policy: Python decides whether a tool call is allowed, blocked,
  needs approval, or needs recovery.
- Approval bridge: the TUI renders approval requests and sends explicit
  decisions back to Python.
- Mutation verification: file or desktop changes are checked after execution
  where possible.
- Redaction and sanitization: observations are cleaned before reaching the
  model, and clipboard/UI metadata avoids leaking content unless requested.
- Session persistence: decisions, tool results, and errors are recorded as
  auditable events.

The model proposes. The runtime disposes.

## Extension Points

Nym is structured for extension without making the model omnipotent.

- Add a provider by extending `LLMClient`.
- Add a tool by registering a `ToolSpec` with a schema and handler.
- Add low-level host behavior in Rust and expose it through `RustTools`.
- Add a channel adapter through the gateway registry.
- Add task-specific instructions through skills.
- Add local or hosted voice providers through environment-configured commands
  or OpenAI-compatible endpoints.
- Add an agent profile in config to restrict tools or skills for a route.

Each extension should preserve the same contract: typed inputs, structured
outputs, explicit policy, and durable session evidence.

## Why This Architecture Works

The architecture is deliberately split:

- Rust gives the terminal UI and host tools speed, native behavior, and reliable
  event handling.
- Python gives the agent loop flexibility, provider integration, policy, and
  straightforward orchestration.
- SQLite gives durable memory without requiring a server.
- JSON bridge contracts keep the runtime boundary debuggable and testable.
- The gateway prepares Nym for multiple channels while preserving the same core
  Agent runtime.

For users, that means Nym feels like one assistant. For engineers, it is a set
of narrow contracts: UI bridge, tool worker, model client, planner loop,
session store, policy, and gateway routing.
