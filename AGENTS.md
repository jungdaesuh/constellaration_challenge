# Repository Guidelines

These guidelines help contributors build, test, and extend the ConStelX starter cleanly and consistently.

## Project Structure & Module Organization

- Source: `constelx/` with submodules: `cli/`, `data/`, `eval/`, `optim/`, `models/` (aka `surrogate/`), `agents/`, `physics/`.
- Tests: `tests/` mirrors package layout (e.g., `tests/test_eval_forward.py`).
- Examples & runs: `examples/` for tiny inputs; `runs/<timestamp>/` for artifacts.

## Build, Test, and Development Commands

- Create env: `python3 -m venv .venv && source .venv/bin/activate`
- System deps (macOS): `brew install netcdf`
- Install (dev + BO): `pip install -e ".[dev,bo]"`
- Lint: `ruff check .` Format: `ruff format .`
- Type check: `mypy constelx` (keep clean; Python ≥ 3.10)
- Tests: `pytest -q` (fast unit + minimal integration)
- CLI help: `constelx --help`
  - Quick smoke: `constelx data fetch --nfp 3 --limit 8`
- E2E (small): `constelx agent run --nfp 3 --budget 5 --seed 0`
  - With PCFM correction:
   - Norm eq: `examples/pcfm_norm.json`
   - Ratio eq: `examples/pcfm_ratio.json`
   - Product eq: `examples/pcfm_product.json`
   - Command: `constelx agent run --nfp 3 --budget 4 --correction pcfm --constraints-file <json>`
   - Tuning: CLI flags or JSON top-level `{gn_iters,damping,tol}`

## Pre-commit Hooks

[![pre-commit enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

- Tools: `ruff` (lint + format), `mypy --strict`, whitespace fixers.
- One-time setup: `pip install -e ".[dev]" && pre-commit install`
- Run on all files: `pre-commit run -a`
- Update hook versions: `pre-commit autoupdate`
- Commit loop: hooks may modify files; re-run `git add -A` and commit again.
- Local equivalents: `ruff format . && ruff check . --fix && mypy --strict src/constelx && pytest -q`

## Coding Style & Naming Conventions

- Python: 4‑space indent, max line length 100 (ruff), f‑strings, dataclasses when helpful.
- Type hints required; `from __future__ import annotations` allowed.
- Naming: modules/functions/vars `lower_snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- Keep modules focused; avoid cross‑package imports that create cycles.

## Testing Guidelines

- Framework: `pytest`; name tests `test_*.py` with clear arrange‑act‑assert.
- Cover: new features need happy‑path + 1–2 edge cases; add a tiny integration test for new CLI.
- Use small fixtures under `tests/fixtures/`; prefer deterministic seeds.

## Commit & Pull Request Guidelines

- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- PRs: < ~400 LOC, green `ruff` + `mypy` + `pytest`, include docs and a minimal example.
- Description: what/why, linked issues, CLI example (if applicable), and notes on performance.

## Architecture & Ops Notes

- Minimum E2E path: `agents.simple_agent` → `eval.forward` → `eval.score` → `optim.cmaes` → artifacts in `runs/` (config, metrics, best).
- System requirements: NetCDF present; PyTorch may require manual install on macOS arm64.
- Reproducibility: log seeds, package versions, and git SHA; prefer small, resumable runs (`--resume`).

# constelx — Coding Agent Runbook

## Prime objective

Implement, test, and document an end‑to‑end optimizer for ConStellaration:
propose boundary -> evaluate via `constellaration` -> compute score -> select next proposals.
Target correctness, determinism, and incremental performance. Prefer small vertical slices that run in minutes.

## Non‑negotiables

- Python ≥ 3.10, type hints everywhere, pass `ruff` + `mypy`.
- Tests: `pytest -q` must pass locally, and in CI (GitHub Actions).
- No breaking changes to CLI without updating docs and tests.
- Reproducibility: log configs, seeds, package versions, git SHA.
- Keep each PR < ~400 LOC if possible; include docs and tests.

## Working agreements

- Commit style: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Add minimal integration test for every new CLI subcommand.
- For long‑running commands, add `--dry-run` and `--limit` flags.
- Always write a minimal example in the module docstring and cross‑link from README.

## Target architecture (already stubbed by the starter)

- `constelx.data`: dataset fetch/filter utilities, simple CSV index.
- `constelx.eval`: thin wrappers around `constellaration` evaluator; one function per metric + `score()`.
- `constelx.opt`: optimizers (`cmaes`, `bo_torch` stub).
- `constelx.models`: baseline MLP + scaffolding for FNO/transformers.
- `constelx.agents`: propose→simulate→select loop; checkpointing & resumability.
- `constelx.cli`: `constelx [data|eval|opt|surrogate|agent] ...`

## Minimum end‑to‑end (Sprint 0)

**Goal**: `constelx agent run --nfp 3 --budget 50` finishes and writes a results table, a JSONL of proposals, and a best‐of run summary.

Required pieces:

1. `constelx.eval.boundary_to_vmec(boundary: dict) -> VmecBoundary`
2. `constelx.eval.forward(boundary) -> dict[str, float]`
3. `constelx.eval.score(metrics: dict[str, float]) -> float` (aggregate as per evaluator defaults)
4. `constelx.opt.cmaes.optimize(f: Callable, x0: np.ndarray, bounds, budget) -> best_x, history`
5. `constelx.agents.simple_agent.run(config)`: wraps (1)-(4), logs to `runs/<timestamp>/...`

### Constraints and safety

- Parameterize boundaries in a small fixed basis (start with circular + low‑order Fourier modes).
- Clip & validate before evaluation (bounds, aspect ratio sanity, coil clearance if available).
- Timeouts and retries around evaluator calls; graceful skip on NaNs.

## Physics‑hard constraints

- Implement a pluggable “correction hook” in `agents` with two strategies:
  - `eci_linear`: projection for linear constraints (IC/BC/value/region).
  - `pcfm`: step‑wise Gauss–Newton projection on a constraint residual `h(u)=0` with Jacobian via autograd (zero‑shot inference correction).
- Training‑time residual minimization hook `pbfm` for flow‑matching surrogates (conflict‑free gradient combination of FM loss and residual loss).

## Logging & artifacts (required)

- Write `runs/<ts>/config.yaml`, `runs/<ts>/proposals.jsonl`, `runs/<ts>/metrics.csv`, `runs/<ts>/best.json`.
- Include a `README.md` in each run folder with CLI used and env info.

Artifacts fields (clarity and provenance)
- CSV columns include: `evaluator_score` (from official evaluator when present), `agg_score` (our aggregated score), `elapsed_ms` (per‑eval time), `feasible` (bool), `fail_reason` (string), and `source` (`placeholder|real`).
- When multi‑fidelity gating is enabled, a `phase` column indicates `proxy` or `real` evaluation phase for each row. Proxy results are cached separately from real results.
- `best.json` stores `agg_score` (and a backward‑compatible `score` alias), optional `evaluator_score`, and a `metrics` object without a conflicting `score` key.

Submission packaging
- `constelx submit pack runs/<ts> --out submissions/<name>.zip [--top-k K]` packs a run.
  - Always writes:
    - `boundary.json` (best boundary by `agg_score`), `metadata.json`.
    - `best.json` is included when present.
  - `metadata.json` fields include: `problem`, `scoring_version`, `git_sha`, and `top_k`.
  - When `--top-k > 1`, also writes `boundaries.jsonl` (one JSON per line) with
    `{iteration,index,agg_score,evaluator_score,feasible,fail_reason,source,scoring_version,boundary}`.

Physics test opt‑in
- Physics‑gated tests are skipped by default to avoid heavy imports on misconfigured systems.
- Set `CONSTELX_RUN_PHYSICS_TESTS=1` to enable parity/physics tests locally or in CI (physics job).

## CLI behavior to implement

- `constelx data fetch --nfp 3 --limit 128`
- `constelx eval forward --boundary-file examples/boundary.json`
- `constelx eval score --metrics-file runs/<ts>/metrics.csv`
- `constelx opt cmaes --nfp 3 --budget 50 [--seed 0]`
- `constelx opt run --baseline trust-constr|alm|cmaes --nfp 3 --budget 50 [--seed 0] [--use-physics --problem p1]`
- `constelx agent run --nfp 3 --budget 50 [--seed 0] [--resume PATH]`

### New helper flags (already available)

- Agent geometry guard: `--guard-geom-validate` pre-screens invalid shapes and logs
  `fail_reason=invalid_geometry` without spending evaluator calls. Thresholds are configurable:
  - `--guard-r0-min` (default 0.05)
  - `--guard-r0-max` (default 5.0)
  - `--guard-helical-ratio-max` (default 0.5)

- Submission packaging: `constelx submit pack runs/<ts> --out submissions/<name>.zip [--top-k 5]`
  - Packs `boundary.json` (best) and `metadata.json`. With `--top-k > 1`, also writes
    `boundaries.jsonl` with the top‑K boundaries by aggregate score.

## Testing checklist

- Unit: boundary validation, scoring math, CMA‑ES step.
- Integration: one agent run with `--budget 5 --limit 8` finishes < 60s and writes artifacts.
- Golden files: store a tiny fixtures boundary & metrics JSON for regression tests.

## Performance rules

- Vectorize evaluator calls when possible; otherwise process pool with a small `--max-workers`.
- Cache per‑boundary derived geometry (hashable key) in a local sqlite or diskcache.

## Definition of done (per task)

- Working code + tests + docstring example + one paragraph in README.
- CI green.
- No perf regressions on the small e2e test.
