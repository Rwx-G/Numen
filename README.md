# Numen

> Causal-native AI runtime in Rust - Pearl causal graph · Active Inference · bio-inspired memory

Numen is a research AI runtime that reasons causally, not statistically. Current
LLMs predict tokens and cannot tell causation from correlation, so in Numen the
LLM is only the language interface: an explicit causal graph (Pearl) is the
reasoning engine, and an Active Inference loop (Friston) decides what to explore
next. The graph grows at every interaction, without retraining any weights.

**Status: early research. Not ready for use.** Interfaces and storage change
without notice.

## What works today

- **Causal graph** (`petgraph`, persisted in SQLite): explain a path, simulate an
  intervention `do(X)`, separate causation from confounding, structural but-for
  counterfactuals. The engine is structural: edge strengths are heuristics, not
  probabilities, and Numen does not pretend otherwise.
- **Learning from language**: answers are delegated to Claude (`claude -p`) or a
  local GGUF model (`mistral.rs`) when the graph is not confident enough; the
  causal relations they state are extracted, screened against high-confidence
  knowledge (contradictions are kept as audit records, not silently accepted),
  and seeded into the graph.
- **Bio-inspired memory**: fast-decaying episodes, slow-decaying semantic edges,
  Hebbian reinforcement on use, idle-time consolidation, decay by provenance.
- **Identity Subgraph**: pinned values and a load-bearing corrigibility edge to
  the operator, excluded from consolidation.
- **Active Inference loop**: concepts ranked by structural expected free energy
  (uncertainty vs usefulness); an optional proactive mode explores on its own.
- **World ingestion**: RSS/Atom feeds, deduplicated, learned in batches.
- **Guarded self-improvement** (off by default): the agent can propose a patch to
  an allowlisted module on a branch; a separate std-only supervisor rebuilds,
  re-tests and deploys it in a sandbox, with rollback.

## Roadmap

- Environment causal model learned by intervention (action = `do()`), with
  perception adapters and an Active Inference action loop.
- Locked core priors (logic, intuitive physics) with explicit regimes: real world
  versus environments whose rules must be discovered.
- Benchmarks: ARC-AGI-3 (interactive, offline), InterveneBench.
- Speech-to-speech interface on top of the same engine.

## Run

The service runs in Docker on an NVIDIA GPU host (the image builds the CUDA
kernels for the RTX 50 series); without a local model it runs Claude-only.

1. Generate a headless Claude token on the host (one-off): `claude setup-token`.
2. Create `.env` from the template and fill it in: `cp .env.example .env`.
   `.env.example` documents every variable.
3. Build and start: `docker compose up --build -d`.

```sh
curl -s localhost:8080/health
curl -s localhost:8080/infer -H 'content-type: application/json' \
  -d '{"query":"how does ownership relate to lifetimes in Rust?"}'
```

## API

`src/routes.rs` is the authoritative list. In short:

- **Reasoning**: `/infer`, `/explain`, `/simulate`, `/query`, `/counterfactual`.
- **Learning**: `/learn` (text corpus), `/ingest_feed` (RSS/Atom), `/consolidate`.
- **Introspection**: `/stats`, `/identity`, `/introspect`, `/graph.dot`,
  `/graph.md`, `/events`, and a dashboard at `/`.
- **Agency**: `/attend` (what Numen wants to explore), `/act`, `/critique`.
- **Self-improvement** (requires `NUMEN_API_TOKEN`): `/improve`, `/deploy`.

## Development

Builds and tests run in a container (`rust:1.99-bookworm`); the host only needs
Docker and `rustfmt`. Quality gates: `cargo fmt --check`, `cargo clippy
--workspace --all-targets -- -D warnings`, `cargo test --workspace`,
`cargo audit`, `cargo deny check`.

## License

Apache-2.0. See `LICENSE`.
