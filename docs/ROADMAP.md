# Implementation roadmap

This repository ships with stubs and guardrails so you can iterate quickly. A recommended path:

1. **Dataset access**: `constelx data fetch` to materialize a filtered subset (e.g., NFP=3) into Parquet.
2. **Forward model sanity checks**: `constelx eval forward --example` and `--random` to verify metrics computation.
3. **Baselines**:
   - Start with `constelx opt cmaes --toy` (sphere) and `constelx opt cmaes --nfp 3 --budget 50` (boundary mode).
   - Train a simple surrogate with `constelx surrogate train` (MLP baseline). A separate `surrogate eval` may be added later.
4. **Agent loop**: glue propose→simulate→select with `constelx agent run` (supports random and CMA‑ES, resume, correction hooks).
5. **Physics‑constrained generation (optional)**:
   - Use `constelx.physics.pcfm` to project generated fields onto hard constraints (norm/ratio/product examples provided).
   - Use `constelx.physics.pbfm` training utilities to add residuals without conflicting with distribution learning.
6. **Submission packaging**: (planned) `constelx submit pack` to export boundaries to the leaderboard format.

—

## Current status (what’s already in place)

- **CLI + agent**: `agent run` writes `config.yaml`, `proposals.jsonl`, `metrics.csv`, `best.json`, `README.md`; supports `--resume`, `--guard-simple`, `--guard-geo`, `--correction {eci_linear,pcfm}`.
- **Scoring/metrics**:
  - Aggregation uses evaluator `score` when present; otherwise computes an `agg_score` (CSV column distinct from `evaluator_score`).
  - `eval.forward`/`forward_many` annotate `elapsed_ms`, `feasible`, and `fail_reason` (defaults supplied on placeholder path).
  - Optional physics path via `--use-physics --problem {p1|p2|p3}` with graceful fallback to placeholders.
- **Caching/parallel**: disk cache backend available; controlled parallelism for real evaluator with thread caps.
- **Constraint hooks**: `eci_linear` (projection onto Ax=b) and `pcfm` (Gauss–Newton) with example JSON specs.
- **Surrogates**: simple MLP trainer available via `constelx surrogate train` (artifacts under `outputs/surrogates/mlp`).
- **Tests**: physics‑gated scoring parity test for P1 (skipped when physics deps missing); agent/data/eval unit and small integration tests.

—

## Near‑term milestones (recommended next steps)

- **Evaluator robustness (#34)**
  - Add per‑call timeouts, retries with exponential backoff, and deterministic failure recording (`feasible=False`, `fail_reason`, `agg_score=inf`).
  - Record provenance (`source=placeholder|real`) and `scoring_version` in CSV/JSON.
- **Physics parity & problems (#31, #33)**
  - Extend parity tests to P2/P3 with tiny fixtures; add Fourier indexing/negative‑n corrections and problem specs P1–P3.
- **Baselines (#32)**
  - Implement ALM + Nevergrad and trust‑constr baselines as an `opt run` CLI path with coefficient whitening and trust region.
- **Data ingestion (#30)**
  - Add HF dataset loader (real dataset) with Parquet cache; keep synthetic as CI default.
- **Multi‑fidelity VMEC**
  - Expose low‑fidelity search vs high‑fidelity scoring toggles; hot‑restart and convergence gating.
- **Surrogate in the loop**
  - Add `surrogate eval` and optional agent integration for model‑guided proposal scoring/screening.

—

## Usage pointers

- Placeholder smoke: `constelx agent run --nfp 3 --budget 6 --seed 0 --guard-simple`
- Physics micro: `CONSTELX_USE_REAL_EVAL=1 constelx agent run --nfp 3 --budget 10 --seed 0 --use-physics --problem p1`
- PCFM examples: see `examples/pcfm_*.json` and use `--correction pcfm --constraints-file <json>` with optional `--pcfm-gn-iters/--pcfm-damping/--pcfm-tol`.
