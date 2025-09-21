# TODOs

## Sequential Tasks (next up)
1. [#28] ConStellaration evaluator wiring follow-ups
   - Core landed via PR #36; finish robustness/docs cleanup from #34 and reflect parity guidance before closing.
2. [#30] Real dataset ingestion (HF dataset) + Parquet cache
   - Add `--source hf`, parquet outputs, fixtures/examples, and document the ingestion path.
3. [#40] BoTorch qNEI baseline (optional extra)
   - Import-guarded baseline with feasibility-aware acquisition and CLI/tests/docs.

Tracking: [#27] remains open until the PR-01…06 checklist is fully cleared (pending #28/#30 and the qNEI baseline in #40).

## Parallelizable Tasks
- [#45] Data-driven seeds prior polish — rerun against HF ingestion once #30 lands, evaluate the flow-based variant, and fold the findings into docs before closing the issue.
- [#44] Pareto sweep QA — capture a physics-enabled Pareto example and thread artifacts/docs updates before shutting the issue.

  (De-dup with Sequential list as work starts.)

## Completed Recently
- [#46] Boozer/QS–QI evaluator bridge — dedicated evaluator wiring now threads proxy metrics through CLI physics toggles and documentation (PR #89).
- [#45] Data-driven seeds prior pipeline — training/sampling CLI added, agent seeding wired, and regression tests shipped (PR #90); polish tracked above before closing.
- [#44] P3 scalarization and Pareto sweep — scalarizers, CLI sweep tooling, and companion docs/tests landed (PR #93); capture physics-enabled artifacts next.
- [#41] QS proxies in PCFM correction and multi-fidelity gating — metrics-facade wiring logs phase/proxy provenance, docs refreshed, and agent smoke tests added (PR #88).
- [#51] Metrics/constraints single source of truth — added unified facade `src/constelx/physics/metrics.py`; `eval.forward`/`forward_many` now enrich results with geometry defaults and bounded Boozer proxies; docs updated (ARCHITECTURE.md); ruff+mypy+pytest green.
- [#32] Baselines and `opt run` CLI — trust-constr/ALM baselines shipped with CLI/docs/tests; issue closed 2025-09-15.
- [#50] VMEC resolution ladder and evaluator cache robustness — CLI/env knobs enable hot restarts while failed evaluations no longer populate the cache; VMEC metadata now threads through adapters.
- [#38] PCFM repair pack — aspect-ratio band, edge-iota proxy, and clearance constraint landed with docs/examples.
- [#73] Agent surrogate screening hook — implemented with CLI/docs.
- [#74] Novelty gating and README/ROADMAP updates — agent skip logic and CLI flags landed (PR #79).
- [#78] CI parallelization + physics test job — core workflow split, optional physics job enabled, torch moved behind a surrogate extra.
- [#71] Evaluator cache housekeeping — `.cache/` ignored and TTL guidance documented (PR #71).
- Docs refresh: AGENTS/STRATEGY/ARCHITECTURE updates (PRs #82 & #81) keep guidance aligned with current modules and CLI.
- [#53] Ablation harness CLI (`constelx ablate run`) — toggle/spec planner merged (PR #76).
- [#52] Multi-start NFP exploration with provenance — round-robin `--nfp-list` support shipped (PR #75).
- [#47] Near-axis seeding for QS/QI-friendly starts — deterministic seed mode and CLI flags available (PR #77).
- [#42] DESC gradient trust-region baseline + resolution ladder — implemented (PR #91) with resolution ladder hooks and agent wiring.
- [#60] Evaluator robustness & provenance (timeouts/retries, score inf on failure, cache TTL, deterministic cache) — merged.
- [#61] Fourier helper, geometry guard thresholds, submit pack with top‑K — merged and documented.
- [#33] Scoring parity tests for P1–P3 (physics-gated) — implemented and documented (PR #62).
- [#39] Multi-fidelity gating with proxy→selection→real and provenance — implemented (PR #63) with single-worker phase fix (PR #65).
- [#43] Nevergrad NGOpt baseline (ALM parity with challenge) — implemented (PR #92) with ALM parity tests and CLI docs.
- [#54] Results DB + novelty checks — implemented (PR #69) and wired into the agent resume/novelty path.
- Docs roadmap added (PR #68).
- Dataset/surrogate speedups (PR #70).
- Fix: boundary m=1 column mapping (PR #66).
