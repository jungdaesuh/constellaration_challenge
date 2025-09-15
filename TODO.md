# TODOs

## Sequential Tasks (next up)
1. [#32] Finalize baselines and `opt run` CLI
   - Trust-constr/ALM are implemented and tested; tighten docs/CLI help and ensure physics path smoke works.
2. [#40] BoTorch qNEI baseline (optional extra)
   - Import-guarded implementation; start with 2–6D helical coeffs and feasibility-aware acquisition.
3. [#38] PCFM repair pack
   - Add aspect-ratio band constraint; lightweight edge-iota proxy; simple clearance proxy. Update examples/docs.
4. [#74] Agent: wire ResultsDB novelty checks
   - Skip near-duplicate boundaries via `ResultsDB.is_novel` in the proposal stage; add a small toggle/threshold.
5. [#71] Docs/housekeeping for evaluator cache
   - Merge PR and document `CONSTELX_CACHE_TTL_SECONDS` + per-run cache dir guidance in README (cross-link CLI `--cache-dir`).
6. [#50] VMEC resolution ladder and hot-restart toggles
   - Parse/store provenance; ensure cache keys separate by resolution level.
7. [#51] Unified metrics/constraints module
   - Single source of truth for geometry/QS proxies to avoid drift across paths.
8. [#28] ConStellaration evaluator wiring
   - Core landed; track robustness/docs follow-ups here.

Note: [#30] HF dataset helpers and seeds are implemented; keep the issue open for expanded ingestion and examples.

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

  (De-dup with Sequential list as work starts.)

## Completed Recently
- [#73] Agent surrogate screening hook — implemented with CLI/docs.
- [#60] Evaluator robustness & provenance (timeouts/retries, score inf on failure, cache TTL, deterministic cache) — merged.
- [#61] Fourier helper, geometry guard thresholds, submit pack with top‑K — merged and documented.
- [#33] Scoring parity tests for P1–P3 (physics-gated) — implemented and documented (PR #62).
- [#39] Multi-fidelity gating with proxy→selection→real and provenance — implemented (PR #63) with single-worker phase fix (PR #65).
- [#54] Results DB + novelty checks — implemented (PR #69); awaiting agent wiring.
- Docs roadmap added (PR #68).
- Dataset/surrogate speedups (PR #70).
- Fix: boundary m=1 column mapping (PR #66).
