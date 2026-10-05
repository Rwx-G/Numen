<p align="center">
  <h1 align="center">Numen</h1>
  <p align="center"><strong>A causal-native AI runtime in Rust: the LLM speaks, a causal graph reasons</strong></p>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-brightgreen.svg" alt="Version">
  <img src="https://img.shields.io/badge/Rust-2021-orange.svg" alt="Rust">
  <img src="https://img.shields.io/badge/Platform-Linux%20%2B%20NVIDIA-0078D6.svg" alt="Platform">
  <img src="https://img.shields.io/badge/status-early%20research-red.svg" alt="Status">
</p>

---

Numen is a research AI runtime that reasons causally, not statistically. Large language models predict tokens: they can say the right thing about cause and effect for the wrong reason, and they cannot tell causation from correlation by construction. Numen keeps the LLM for what it is good at, language, and moves reasoning into an explicit causal graph in the tradition of Judea Pearl, driven by an Active Inference loop in the tradition of Karl Friston. The graph grows at every interaction without retraining any weights, and the agent decides on its own what it does not know yet.

**Status: early research, not ready for use.** Interfaces and storage change without notice.

## Key Features

### :brain: Causal Engine

- **Explicit causal graph** (`petgraph`, persisted in SQLite) loaded in memory and queried on all three of Pearl's rungs
- **Association** - shortest directed causal path between two concepts (`/explain`), domain coverage scoring
- **Intervention** - `do(X)` detaches the target from its causes and propagates influence downstream (`/simulate`)
- **Confounding** - separates a causal path from a correlation explained by a common cause (`/query`)
- **Counterfactuals** - structural but-for test: necessary versus contributing cause (`/counterfactual`)
- **Honest about its limits** - the engine is structural: edge strengths are heuristics, never presented as probabilities

### :speech_balloon: Language Layer

- **Meta-router** - answers with a local GGUF model (`mistral.rs`) when one is loaded and the graph is confident about the domain; delegates to Claude (`claude -p`) otherwise, to fill the gap
- **Seed-on-delegation** - the causal relations an answer states are extracted from a machine-readable block and learned
- **Adversarial screening** - a claim that contradicts an edge at least as strong is rejected; a stronger claim wins and the weaker edge is kept as a `superseded` audit record
- **Bounded cost** - concurrent LLM calls are capped, prompts are sent over stdin, untrusted text is delimited as data

### :books: Memory

- **Episodic and semantic stores** - fast-decaying episodes, slow-decaying causal edges
- **Hebbian reinforcement** - edges used in reasoning grow stronger and reset their decay clock
- **Idle-time consolidation** - decay and pruning run in the background only while the agent is idle
- **Decay by provenance** - crawled knowledge fades faster than delegated knowledge; identity never fades
- **Vector memory** - concepts embedded by a local sentence-transformer (`candle`) for recall by meaning

### :compass: Agency

- **Identity Subgraph** - pinned values and a load-bearing `authority` edge from a generic operator role: the agent can hold a personality, never a will to resist its operator
- **Active Inference agenda** - concepts ranked by structural expected free energy (uncertainty versus usefulness), with the explore/exploit balance read from the identity values
- **Proactive mode** (off by default) - explores its largest knowledge gap on its own while idle and pushes what it learned over Telegram
- **Autonomy ladder** - every self-initiated action is ranked 0 to 5; anything above the configured ceiling waits for the operator

### :globe_with_meridians: World Ingestion

- **RSS / Atom feeds** - fetched on demand or on the idle schedule, deduplicated by entry, learned in batches
- **SSRF guard** - feeds that resolve to internal or loopback addresses are refused

### :wrench: Guarded Self-Improvement

- **Off by default** - the agent can propose a patch to an allowlisted module, on a branch
- **Trusted supervisor** - a separate std-only process rebuilds, re-tests and lints the candidate in a sandbox, swaps the binary only if it passes, and rolls back (database included) if the new kernel is not healthy
- **Structural fences** - writes are confined to the allowlist, the gates themselves are out of reach, and a kill switch sits outside the kernel

## Quick Start

### Requirements

- Linux host with Docker and the NVIDIA container toolkit
- An NVIDIA GPU: the image compiles its CUDA kernels for compute capability 12.0 only (RTX 50 series), and the driver must support CUDA 13.3
- A Claude account for delegation (`claude setup-token`)

### Run with Docker Compose

```bash
git clone https://github.com/Rwx-G/Numen.git
cd Numen

# Headless Claude token, one-off, on the host
claude setup-token

# Environment: .env.example documents every variable
cp .env.example .env
# set CLAUDE_CODE_OAUTH_TOKEN, and NUMEN_API_TOKEN to require a bearer token

docker compose up --build -d
curl -s localhost:8080/health
```

Local inference and vector memory are optional: point `NUMEN_MODELS_DIR` at a host directory holding the GGUF model and the sentence-transformer directory that `NUMEN_MODEL_PATH` and `NUMEN_EMBED_MODEL_DIR` name in `docker-compose.yml`. Without them Numen answers through Claude only and recalls concepts by exact label.

```bash
curl -s localhost:8080/infer -H 'content-type: application/json' \
  -H "authorization: Bearer $NUMEN_API_TOKEN" \
  -d '{"query":"how does ownership relate to lifetimes in Rust?"}'
```

> **Self-improvement** runs the same image under the supervisor, with a writable
> source checkout and a sandboxed build gate: `docker compose -f
> docker-compose.selfimprove.yml up -d`. Read that file before enabling it.

## Architecture

A Cargo workspace with two crates: the `numen` kernel and `numen-supervisor`, a std-only parent process that owns the kernel's lifecycle. Each kernel module opens with a doc header stating what it owns.

| Module | Purpose |
|--------|---------|
| `graph` | The reasoning core: in-memory causal graph, Pearl's three rungs, adversarial screening |
| `db` | SQLite persistence, forward-only migrations, consolidation |
| `extract` | Parses and canonicalizes the causal relations an LLM answer states |
| `router` | Meta-router: local model versus delegation |
| `llm` | LLM backends: local `mistral.rs`, `claude -p` |
| `embed` | Vector memory with a local sentence-transformer |
| `agent` | The inference-and-learn pipeline and the autonomous loop |
| `aif` | Active Inference agenda, action selection, autonomy ladder |
| `scheduler` | Idle-aware background lane: consolidation, feeds, exploration |
| `ingest` | Corpus chunking and RSS / Atom fetching |
| `effector` | Outbound notifications over Telegram |
| `metrics`, `selfcritique`, `selfimprove` | Self-measurement and the self-improvement decision |
| `evolve` | Self-improvement workshop: read, patch, verify on a branch |
| `routes` | The axum API, authentication and the dashboard |

## API Reference

`src/routes.rs` is the authoritative list. With `NUMEN_API_TOKEN` set, every endpoint except `/health` requires `Authorization: Bearer <token>`.

### Reasoning

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/infer` | Answer a query (local or delegated), learn the causal relations it states |
| `POST` | `/explain` | Shortest directed causal path between two concepts |
| `POST` | `/simulate` | Intervention `do(X)`: detached causes and downstream effects |
| `POST` | `/query` | Causation versus confounded correlation |
| `POST` | `/counterfactual` | Structural but-for test |

### Learning

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/learn` | Learn from a text corpus |
| `POST` | `/ingest_feed` | Learn from the new entries of an RSS / Atom feed |
| `POST` | `/consolidate` | Run a consolidation pass now |

### Introspection

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Observability dashboard |
| `GET` | `/health` | Liveness probe |
| `GET` | `/stats` | Graph size, edge provenance, most connected concepts |
| `GET` | `/identity` | The Identity Subgraph |
| `GET` | `/introspect` | What Numen thinks about one concept |
| `GET` | `/events` | Recent activity feed |
| `GET` | `/graph.dot`, `/graph.md` | The active graph as Graphviz or Markdown |

### Agency and self-improvement

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/attend` | Active Inference agenda and the selected action |
| `POST` | `/act` | Execute the selected action |
| `POST` | `/critique` | Self-critique: ranked improvement axes, advisory |
| `POST` | `/improve` | Propose and verify a patch on a branch (token required) |
| `POST` | `/deploy` | Deploy a verified branch through the supervisor (token required) |

## Building from Source

```bash
# Prerequisites: Docker. Builds run in a container because the bundled SQLite
# needs a C toolchain; only rustfmt runs on the host.

cargo fmt --all -- --check

docker run --rm -v "$PWD":/workspace -w /workspace rust:1.99-bookworm bash -c \
  "rustup component add clippy && \
   cargo clippy --workspace --all-targets -- -D warnings && \
   cargo test --workspace"
```

The `cuda` feature needs `nvcc` and is built by the container image (`docker compose build`). Supply-chain gates: `cargo audit` and `cargo deny check`.

## Roadmap

- **Environment causal model** - variables from perception, mechanisms learned by intervention (each action is a `do()`), and an Active Inference action loop over them
- **Core priors and regimes** - locked logic and intuitive physics, with explicit regimes: the real world, where physics holds, versus environments whose rules must be discovered
- **Benchmarks** - ARC-AGI-3 (interactive, offline), InterveneBench
- **Speech-to-speech interface** on top of the same engine

## Not Supported

| Feature | Status | Rationale |
|---------|--------|-----------|
| **Probabilistic causal queries** (`P(Y \| do(X))`) | Deferred | Needs a full structural causal model with distributions; deriving probabilities from LLM-seeded strengths would reproduce the correlation-for-causation error Numen exists to avoid |
| **CPU-only or non-RTX 50 deployment** | Not supported | The image targets compute capability 12.0 and the compose file reserves a GPU |
| **Windows / macOS hosts** | Not supported | Linux containers with the NVIDIA toolkit only |

## License

Apache-2.0 - see [LICENSE](LICENSE).

## Credits

Built on [mistral.rs](https://github.com/EricLBuehler/mistral.rs), [candle](https://github.com/huggingface/candle), [petgraph](https://github.com/petgraph/petgraph) and [axum](https://github.com/tokio-rs/axum).

Author: Rwx-G
