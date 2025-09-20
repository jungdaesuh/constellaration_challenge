# TODOs

## Sequential Tasks (next up)
1. [#41] Integrate QS proxies into PCFM constraints and multi-fidelity gating
   - PR #85 closed without merge; redo the wiring atop the new metrics facade and keep `--mf-proxy-metric` provenance consistent when re-landing.
2. [#28] ConStellaration evaluator wiring follow-ups
   - Core landed via PR #36; finish robustness/docs cleanup from #34 and reflect parity guidance before closing.
3. [#30] Real dataset ingestion (HF dataset) + Parquet cache
   - Add `--source hf`, parquet outputs, fixtures/examples, and document the ingestion path.
4. [#40] BoTorch qNEI baseline (optional extra)
   - Import-guarded baseline with feasibility-aware acquisition and CLI/tests/docs.

Tracking: [#27] remains open until the PR-01…06 checklist is fully cleared (pending #28/#30 and the revised #41 work).

## Parallelizable Tasks
- [#46] Boozer/QS–QI proxy library — heuristics landed in PR #84; polish docs/range guidance and close once #41 merges.
- [#45] Data-driven seeds prior (PCA + RF feasibility + GMM/flow models).
- [#44] P3 scalarization and Pareto sweep.
- [#43] Nevergrad NGOpt baseline (ALM parity with challenge).

  (De-dup with Sequential list as work starts.)

## Completed Recently
- [#51] Metrics/constraints single source of truth — added unified facade `src/constelx/physics/metrics.py`; `eval.forward`/`forward_many` now enrich results with geometry defaults and bounded Boozer proxies; docs updated (ARCHITECTURE.md); ruff+mypy+pytest green.
- [#32] Baselines and `opt run` CLI — trust-constr/ALM baselines shipped with CLI/docs/tests; issue closed 2025-09-15.
- [#50] VMEC resolution ladder and hot-restart toggles — CLI/env knobs wired through eval/agent/opt, cache separation and provenance recorded.
- [#38] PCFM repair pack — aspect-ratio band, edge-iota proxy, and clearance constraint landed with docs/examples.
- [#73] Agent surrogate screening hook — implemented with CLI/docs.
- [#74] Novelty gating and README/ROADMAP updates — agent skip logic and CLI flags landed (PR #79).
- [#78] CI parallelization + physics test job — core workflow split, optional physics job enabled, torch moved behind a surrogate extra.
- [#71] Evaluator cache housekeeping — `.cache/` ignored and TTL guidance documented (PR #71).
- Docs refresh: AGENTS/STRATEGY/ARCHITECTURE updates (PRs #82 & #81) keep guidance aligned with current modules and CLI.
- [#53] Ablation harness CLI (`constelx ablate run`) — toggle/spec planner merged (PR #76).
- [#52] Multi-start NFP exploration with provenance — round-robin `--nfp-list` support shipped (PR #75).
- [#47] Near-axis seeding for QS/QI-friendly starts — deterministic seed mode and CLI flags available (PR #77).
- [#42] DESC gradient trust-region baseline + resolution ladder — implemented (this PR).
- [#60] Evaluator robustness & provenance (timeouts/retries, score inf on failure, cache TTL, deterministic cache) — merged.
- [#61] Fourier helper, geometry guard thresholds, submit pack with top‑K — merged and documented.
- [#33] Scoring parity tests for P1–P3 (physics-gated) — implemented and documented (PR #62).
- [#39] Multi-fidelity gating with proxy→selection→real and provenance — implemented (PR #63) with single-worker phase fix (PR #65).
- [#54] Results DB + novelty checks — implemented (PR #69) and wired into the agent resume/novelty path.
- Docs roadmap added (PR #68).
- Dataset/surrogate speedups (PR #70).
- Fix: boundary m=1 column mapping (PR #66).
