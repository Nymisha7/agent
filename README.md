

# Nym

Nym is a terminal coding agent with a Python agent core and a Rust/Ratatui interface.

## Run

Install the Python package, build the Rust frontend, then launch the TUI:

```bash
python -m pip install -e .
cd nym-rust && cargo build && cd ..
nym --tui
```

Type `/` inside the TUI to open the command menu. The main commands are `/model`,
`/install`, `/status`, `/setup`, `/help`, and `/exit`.

## Open-source models (no login)

Nym can use open-source models through local runtimes. These providers never open
an account page or ask for a Nym login:

| Provider | Default endpoint | Example |
| --- | --- | --- |
| Ollama | `http://localhost:11434/v1` | `/model ollama qwen2.5-coder` |
| LM Studio | `http://localhost:1234/v1` | `/model lmstudio <loaded-model>` |
| llama.cpp | `http://localhost:8080/v1` | `/model llamacpp <loaded-model>` |
| vLLM | `http://localhost:8000/v1` | `/model vllm <served-model>` |
| LocalAI | `http://localhost:8080/v1` | `/model localai <loaded-model>` |

The runtime owns model downloads and storage; Nym discovers models from its local
model-list endpoint. Override endpoints with `NYM_OLLAMA_BASE_URL`,
`NYM_LMSTUDIO_BASE_URL`, `NYM_LLAMACPP_BASE_URL`, `NYM_VLLM_BASE_URL`, or
`NYM_LOCALAI_BASE_URL`.

Selecting a local model checks the runtime and installed-model list first. If an
Ollama model is missing, install and select it from the TUI with
`/install ollama <model>`. Nym reports an offline runtime separately from a missing
model. LM Studio, llama.cpp, vLLM, and LocalAI show their runtime-specific manual
setup instructions because they do not share a reliable universal install command.

Hosted providers remain available and request an API key or cloud credentials only
when their provider requires it.

## Repository layout

This repository contains two parts:

- `nym/`: the Python agent
- `nym-rust/`: the Rust backend


Build the Rust backend from `nym-rust/` with:

```bash
cargo build
```

## Notes

The repository is organized as a combined workspace-style layout so both projects live in one place.
