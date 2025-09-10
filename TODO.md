# TODOs

## Sequential Tasks (next up)
1. [#30] Real dataset ingestion with Hugging Face dataset and Parquet cache.
2. [#32] Baselines: ALM/trust-constr optimizers and `opt run` CLI.
3. [#38] PCFM repair pack: aspect-ratio band, edge-iota proxy, clearance constraints.
4. [#50] VMEC resolution ladder and hot-restart toggles.
5. [#51] Unified metrics/constraints module as single source of truth.
6. [#28] ConStellaration evaluator wiring — core landed; track robustness/docs follow-ups here.

## Parallelizable Tasks
- [#53] Ablation harness for pipeline components.
- [#52] Multi-start NFP exploration with provenance tracking.
- [#47] Near-axis expansion seeding for QS/QI-friendly starts.
- [#46] Boozer/QS–QI proxy library with bounded residuals.
- [#45] Data-driven seeds prior (PCA + RF feasibility + GMM/flow models).
- [#44] P3 scalarization and Pareto sweep.
- [#43] Nevergrad NGOpt baseline (ALM parity with challenge).
- [#42] DESC integration: gradient trust-region baseline and resolution ladder.
- [#41] Integrate QS proxies into PCFM constraints and multi-fidelity gating.
- [#40] Constrained BoTorch qNEI baseline.

## Completed Recently
- [#33] Scoring parity tests for P1–P3 with golden fixtures — implemented and documented (PR #62).
- [#39] Multi-fidelity toggle with proxy→selection→real and provenance — implemented (PR #63) with single-worker phase fix (PR #65).
- [#54] Results DB + novelty checks — implemented (PR #69) and issue closed.
- Docs roadmap added (PR #68).
- Dataset/surrogate speedups (PR #70).
- Fix: boundary m=1 column mapping (PR #66).
