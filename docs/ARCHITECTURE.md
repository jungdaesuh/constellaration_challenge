# Architecture — constelx (repo‑aligned)

This document maps the conceptual strategy to the actual modules and CLI in this repository.

## Repo layout

- `src/constelx/cli.py` — Typer CLI entry; subcommands: `data`, `eval`, `opt`, `surrogate`, `agent`, `submit`, `ablate`.
- `src/constelx/agents/` — Agent loop and correction hooks
  - `simple_agent.py` — propose → simulate → select loop; artifacts under `runs/<ts>/`
  - `corrections/eci_linear.py` — linear (Ax=b) projection hook
  - `corrections/pcfm.py` — nonlinear (Gauss–Newton) projection hook with norm/ratio/product examples
- `src/constelx/eval/` — Forward evaluation and scoring
  - `__init__.py` — `forward`, `forward_many`, `score`, caching, MF proxy gating (`phase` field)
  - `boundary_param.py`, `boundary_fourier.py` — boundary sampling/validation utilities
  - `geometry.py` — quick geometry guards (`invalid_r0`, `helical_exceeds_ratio`)
- `src/constelx/physics/` — thin adapters and helpers
  - `constel_api.py` — lightweight placeholder evaluator
  - `proxima_eval.py` — adapter to `constellaration` evaluator; VMEC boundary validation; scoring passthrough
  - `metrics.py` — unified metrics/constraints facade; attaches Boozer proxies and geometry defaults
  - `pcfm.py` — damped Gauss–Newton projector used by correction hook
  - `pbfm.py` — conflict‑free gradient combination for training‑time residuals
- `src/constelx/optim/` — baselines
  - `cmaes.py` — small CMA‑ES wrapper for smoke tests
  - `baselines.py` — trust‑constr and ALM on 2D helical coefficients
- `src/constelx/surrogate/` — simple baseline model
  - `train.py` — MLP trainer; optional `--use-pbfm`
- `src/constelx/submit/pack.py` — packs a run into a submission zip
- `src/constelx/problems/` — minimal problem specs (P1–P3) used by CLI

## CLI → module mapping

- `constelx data fetch --nfp 3 --limit 128` → `data/dataset.py` or HF loader
- `constelx eval forward --example|--random|--near-axis` → `eval.forward`
- `constelx eval score --metrics-json|--metrics-file` → `eval.score`; `--problem p1|p2|p3` uses physics scorer when available
- `constelx opt cmaes|run` → `optim/cmaes.py`, `optim/baselines.py`
- `constelx surrogate train [--use-pbfm]` → `surrogate/train.py`
- `constelx agent run` → `agents/simple_agent.py` with options:
  - Correction hooks: `--correction eci_linear|pcfm --constraints-file <json> [--pcfm-gn-iters ...]`
  - Multi‑fidelity: `--mf-proxy [--mf-threshold|--mf-quantile] [--mf-max-high]`
  - Guards: `--guard-geom-validate [--guard-r0-min|--guard-r0-max|--guard-helical-ratio-max]`
  - Novelty: `--novelty-skip ...` (window + optional persisted DB). When a
    `--novelty-db` path is provided we now reuse the stored embeddings on later
    runs: duplicates are skipped without new evaluations, while fresh random
    samples continue until the requested budget is exhausted.
  - Surrogate screening: `--surrogate-screen --surrogate-model <pt> [--threshold|--quantile]`
  - NFP multi‑start: `--nfp-list "3,4,5"` (round‑robin)
- `constelx submit pack runs/<ts> --out submissions/<name>.zip [--top-k K]` → `submit/pack.py`

## Artifacts and schema

Agent writes `runs/<ts>/`:
- `config.yaml` — run config + env + git SHA + package versions
- `proposals.jsonl` — `{iteration,index,seed,nfp,boundary}`
- `metrics.csv` — columns include `nfp`, `evaluator_score`, `agg_score`, `elapsed_ms`, `feasible`, `fail_reason`, `source`; when MF/surrogate is enabled a `phase` column is present (`proxy|real|surrogate`)
- `best.json` — `agg_score` (and `score` alias), optional `evaluator_score`, `metrics`, `boundary`, `nfp`
- `README.md` — CLI used and environment info

Submission pack (`submit pack`) writes:
- `boundary.json`, `metadata.json`, and `best.json` (if present). With `--top-k>1`, also `boundaries.jsonl` (records include `{iteration,index,agg_score,evaluator_score,feasible,fail_reason,source,scoring_version,boundary}`).

## Current vs. planned

- Current: ECI/PCFM hooks, MF‑gating + cache, novelty, surrogate screening, MLP+PBFM trainer, trust‑constr/ALM/CMA‑ES baselines, submission pack.
- Planned: feasibility‑first TR‑BO (FuRBO/BoTorch), expanded surrogate families, and optional LLM‑assisted planner emitting ablation specs for `constelx ablate run`.
