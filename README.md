# Numen

> Causal-native AI runtime in Rust - Pearl SCM · Active Inference · bio-inspired temporal stack

Numen is a research AI runtime that reasons causally, not statistically.
Unlike LLMs, it builds and queries an explicit causal model of the world (Pearl SCM), combines it with Active Inference (Friston/FEP), and routes computation through a bio-inspired temporal hierarchy - from reflexive cache to deliberate causal reasoning.

**Status: early research. Not ready for use.**

---

## What makes it different

- **Causal engine first**: explicit Pearl SCM (do-calculus, counterfactuals) - not correlation
- **Active Inference**: beliefs updated by minimizing variational free energy, not token prediction
- **Temporal hierarchy**: 4-layer stack (reflex → salience → contextual → deliberate) inspired by biological neural architecture
- **Async consolidation**: hippocampus-to-neocortex memory consolidation, solving catastrophic forgetting
- **Rust-native**: single binary, self-hosted, no cloud dependency

---

## Roadmap

- [ ] Phase 1 - SCM minimal (Rust, petgraph)
- [ ] Phase 2 - LLM layer (mistral.rs) + causal conditioning
- [ ] Phase 3 - Active Inference loop + temporal stack
- [ ] Phase 4 - Benchmark (InterveneBench, ARC-AGI-3)

---

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright (c) 2026 Rwx-G

Free to share and adapt with attribution. Commercial use requires explicit agreement.
Contact: romain@rwx-g.fr - https://creativecommons.org/licenses/by-nc/4.0
