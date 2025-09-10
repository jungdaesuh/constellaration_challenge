# TODOs

## Sequential Tasks
1. [#28] Wire to ConStellaration evaluator and official scoring (drop-in physics).
2. [#30] Real dataset ingestion with Hugging Face dataset and Parquet cache.
3. [#32] Baselines: ALM/trust-constr optimizers and `opt run` CLI.
4. [#33] Scoring parity tests for P1–P3 with golden fixtures.
5. [#39] Multi-fidelity toggle for proxy vs high-fidelity scoring with provenance.
6. [#38] PCFM repair pack: aspect-ratio band, edge-iota proxy, clearance constraints.
7. [#50] VMEC resolution ladder and hot-restart toggles.
8. [#51] Unified metrics/constraints module as single source of truth.

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
