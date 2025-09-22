# TODOs

## Sequential Tasks (next up)

- [ ] Shepherd PR #100 to merge and close [#27]; confirm maintainers are satisfied once the checklist lands on main.
- [ ] [#45] Data-driven seeds prior polish — rerun the prior against the HF Parquet cache, compare GMM vs flow outputs, wire the HF seeds into `agent run --seed-mode prior`, and update README/AGENTS with the findings.
- [ ] [#44] Pareto sweep QA — produce a physics-enabled Pareto sweep (JSON + plot), archive it under `examples/` (or `runs/`) and document the workflow/results.
- [ ] [#43] Nevergrad NGOpt baseline — run parity vs the challenge ALM baseline using the official evaluator, capture the metrics, and document the comparison.
- [ ] [#42] DESC gradient trust-region baseline — execute a real DESC ladder with VMEC validation enabled, log the outcomes, and extend the docs with usage guidance and caveats.
- [ ] Retire "toy" fallback defaults — tighten docs/tests around the development-only sphere objective (`src/constelx/cli.py:508-520`, `README.md:79`, `docs/ROADMAP.md:13`, `src/constelx/surrogate/train.py:66`) so CI and primary docs emphasize physics-backed runs.
- [ ] Replace "stub" references — refresh guidance in `AGENTS.md:116`, `CONSTELX_ANALYSIS_REPORT.md:15`, `docs/GUIDELINE.md:532-543`, and `docs/STRATEGY.md:491,720` to describe production extension hooks instead of TODO stubs.

## Parallelizable Tasks
- [ ] Backlog grooming — review the open issues monthly to keep Sequential and Completed lists honest.

  (De-dup with Sequential list as work starts.)

## Dev-only and placeholder inventory (recorded)
- Synthetic data paths (dev/CI only)
  - `src/constelx/cli.py:63` `data fetch` defaults to `source="synthetic"`; pass `--source hf` for the real dataset.
  - `src/constelx/cli.py:307` `eval forward --example` uses a synthetic example boundary (see `README.md:24`).
  - `src/constelx/data/dataset.py:3,25,59` deterministic synthetic fallback via `_synthetic_examples()` used by tests and CI.
  - Docs callouts: `docs/ROADMAP.md:49` (synthetic as CI default), `docs/GUIDELINE.md:574` (synthetic-boundary smoke).

- Placeholder evaluator paths (development fallback; provenance recorded)
  - `src/constelx/eval/__init__.py:1102` placeholder evaluator helper and path; metrics include `source=placeholder` and `agg_score` distinct from `evaluator_score`.
  - `src/constelx/physics/proxima_eval.py:244` synthetic objectives for placeholder path; `_fallback_metrics` used when physics stack is missing.
  - `src/constelx/physics/constel_api.py:48-99` lightweight placeholder evaluator and norms for `placeholder_metric`.
  - `README.md:268` explicitly documents fallback behavior when physics extras are unavailable.
  - `src/constelx/eval/__init__.py` also sets `source=placeholder` across forward/forward_many fallbacks (e.g., lines 433, 453, 611, 628, 839–842, 895, 913, 920, 942, 1003, 1031–1042, 1083, 1091, 1099–1110).
  - `src/constelx/agents/simple_agent.py` records placeholder provenance in several branches (e.g., lines ~769, 856, 875, 936, 959, 1191, 1206).

- Diagnostic CMA‑ES baseline and related
  - `src/constelx/optim/evolution.py` “toy/placeholder” score path for the tiny CMA‑ES baseline (dev smoke).
  - `src/constelx/cli.py:508-520` `--toy` sphere objective switch; `README.md:79` notes it as development‑only.
  - `src/constelx/optim/baselines.py:97` ALM docstring references placeholder feasibility integration when available.
  - `src/constelx/cli.py:670-673` Pareto command docstring refers to the P3 placeholder.

- Provenance handling
  - `src/constelx/physics/metrics.py:69` treats `source in {"placeholder","synthetic"}` consistently for provenance.

- Physics placeholders (module‑level docstrings)
  - `src/constelx/physics/constraints.py:1` constraint placeholders note.
  - `src/constelx/physics/pbfm.py:1` PBFM placeholder tooling (training utilities), dev path only.

- Documentation references (non‑code)
  - `README.md:100,103,149` mention placeholder paths (P3 placeholder, fallback evaluator, and gating routing wording).
  - `AGENTS.md:154` documents `source=placeholder|real` in CSV columns.
  - `docs/ARCHITECTURE.md:18` notes `constel_api.py` as a lightweight placeholder evaluator.
  - `docs/STRATEGY.md:7` mentions placeholder in the metrics/proxy context.
  - `docs/ALPHA_FOLD_INSPIRED.md:204,223` discuss synthetic dataset coverage and caveats.
  - Tests: `tests/test_agent_mf_integration.py:29` uses the placeholder path (`use_physics=False`) in a smoke.

## Completed Recently
- [#40] BoTorch qNEI baseline — import-guarded feasibility-aware qNEI baseline with CLI/tests/docs.
- [#28] ConStellaration evaluator wiring follow-ups — README/AGENTS now document `--use-real`
  usage, parity workflow, and environment knobs (PR #98 follow-up).
- [#20] PCFM correction docs — README/AGENTS now ship norm JSON snippet, command example, and Gauss–Newton safety notes (PR #98).
- [#30] Real dataset ingestion (HF dataset) + Parquet cache — added `--source hf`, robust Parquet export with nested→flat conversion, examples, README/docs, and tests (PR #96).
- [#46] Boozer/QS–QI evaluator bridge — dedicated evaluator wiring now threads proxy metrics through CLI physics toggles and documentation (PR #89).
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
- [#60] Evaluator robustness & provenance (timeouts/retries, score inf on failure, cache TTL, deterministic cache) — merged.
- [#61] Fourier helper, geometry guard thresholds, submit pack with top‑K — merged and documented.
- [#33] Scoring parity tests for P1–P3 (physics-gated) — implemented and documented (PR #62).
- [#39] Multi-fidelity gating with proxy→selection→real and provenance — implemented (PR #63) with single-worker phase fix (PR #65).
- [#54] Results DB + novelty checks — implemented (PR #69) and wired into the agent resume/novelty path.
- Docs roadmap added (PR #68).
- Dataset/surrogate speedups (PR #70).
- Fix: boundary m=1 column mapping (PR #66).
