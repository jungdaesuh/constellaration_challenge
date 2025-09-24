You're the world's best expert in physics, math, AI/ML, computation, computer science, and statistics.

# Repository Guidelines

These guidelines help contributors build, test, and extend the ConStelX research stack cleanly and consistently.

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
- E2E (small, real stack): `constelx agent run --nfp 3 --budget 5 --seed 0 --use-physics --problem p1`
  (See `examples/real_quickstart/` for reference config, metrics, and best.json.)
  - With PCFM correction (Gauss–Newton projection with clamped steps + geometry revalidation):
    ```json
    [
      {
        "type": "norm_eq",
        "radius": 0.06,
        "terms": [
          {"field": "r_cos", "i": 1, "j": 5, "w": 1.0},
          {"field": "z_sin", "i": 1, "j": 5, "w": 1.0}
        ]
      }
    ]
    ```
    - Launch: `constelx agent run --nfp 3 --budget 4 --correction pcfm --constraints-file examples/pcfm_norm.json`
    - Ready-to-use specs live in `examples/pcfm_*.json` (norm, ratio, product, aspect-ratio band, edge iota ratio, QS residual band, clearance floor).
    - Tuning: CLI flags `--pcfm-gn-iters/--pcfm-damping/--pcfm-tol` or JSON top-level `{gn_iters,damping,tol}`.
    - See [README.md#pcfm-correction-gaussnewton-projection](README.md#pcfm-correction-gaussnewton-projection) for additional context and safety guidance.
- ConStellaration parity:
  - Forward metrics via official evaluator: `constelx eval forward --near-axis --use-physics --problem p1 --json` (set `CONSTELX_USE_REAL_EVAL=1` to make this the default).
  - Score aggregation check (dev fixture): `constelx eval score --metrics-json examples/dev/metrics_small.json --problem p1` (or
    `--metrics-file runs/<ts>/metrics.csv --problem p1`).
  - Run gated parity tests (requires `pip install -e ".[physics]"` and the evaluator deps):
    `CONSTELX_RUN_PHYSICS_TESTS=1 pytest -q -k scoring_parity`.

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
- Auto-close issues on merge: when a PR fully resolves an issue, include closing keywords in the PR description so GitHub closes them automatically on merge. Examples: `Closes #123`, `Fixes owner/repo#456`, `Resolves #789`. Keep these only when the PR truly completes the issue.
- Auto-merge: it’s fine to enable GitHub’s auto-merge once checks are green. For PRs that should close issues on merge, ensure closing keywords are present in the PR description before enabling auto-merge.

## Architecture & Ops Notes

Refer to `docs/ARCHITECTURE.md` for a detailed map from CLI commands to modules and
`docs/ROADMAP.md` for upcoming milestones; the highlights are below.

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

## Target architecture

- `constelx.data`: dataset fetch/filter utilities, simple CSV index.
- `constelx.eval`: thin wrappers around `constellaration` evaluator; one function per metric + `score()`.
- `constelx.opt`: optimizers (CMA-ES, Nevergrad NGOpt, BoTorch qNEI, DESC trust-region) with extension hooks.
- `constelx.models`: baseline MLP + scaffolding for FNO/transformers.
- `constelx.agents`: propose→simulate→select loop; checkpointing & resumability.
- `constelx.cli`: `constelx [data|eval|opt|surrogate|agent] ...`
- `constelx.physics`: evaluator shims, constraint tooling, and Boozer-space QS/QI proxies (`booz_proxy`).

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

- CSV columns include: `nfp`, `evaluator_score` (from official evaluator when present), `agg_score` (our aggregated score), `elapsed_ms` (per‑eval time), `feasible` (bool), `fail_reason` (string), and `source` (`placeholder|real`).
- When multi‑fidelity gating is enabled, proxy rows include `phase=proxy`, the gating `proxy_metric`, and a `proxy_score` column in addition to the usual provenance (real evaluations keep `phase=real`). Proxy results are cached separately from real results.
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
- `constelx data prior-train data/cache/subset.parquet --out models/seeds_prior_hf_gmm.joblib`
- `constelx data prior-sample models/seeds_prior_hf_gmm.joblib --count 16 --nfp 3`
- `constelx eval forward --boundary-file examples/dev/boundary.json`
- `constelx eval score --metrics-file runs/<ts>/metrics.csv`
- `constelx opt cmaes --nfp 3 --budget 50 [--seed 0]`
- `constelx opt run --baseline trust-constr|alm|cmaes --nfp 3 --budget 50 [--seed 0] [--use-physics --problem p1]`
- `constelx agent run --nfp 3 --budget 50 --use-physics --problem p1 [--seed 0] [--resume PATH]`
- Prior seeding: `constelx agent run --nfp 3 --budget 20 --seed-mode prior --seed-prior models/seeds_prior_hf_gmm.joblib`

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
- Integration: one agent run with `--budget 5 --use-physics --problem p1` finishes < 60s and writes artifacts (see `examples/real_quickstart/`).
- Golden files: store a tiny fixtures boundary & metrics JSON for regression tests.

## Performance rules

- Vectorize evaluator calls when possible to keep iteration cost down.

## Timeless principles

- KISS, DRY, YAGNI: Keep designs simple, avoid duplication, don’t build speculative features.
- Separation of Concerns & SRP: Isolate responsibilities so each module has one reason to change.
- Modularity & Abstraction: Encapsulate details behind stable interfaces to enable safe swaps and evolution.
- Readability first: Prefer clear naming and straightforward logic—code is read far more than written.

## Object-oriented foundations (SOLID & beyond)

- OCP: Extend behavior without modifying existing code.
- LSP: Subtypes must honor base-type expectations.
- ISP: Use small, focused interfaces; don’t force unused methods.
- DIP: Depend on abstractions, not concrete implementations (enables DI).
- Favor composition over inheritance: Compose behaviors to reduce brittleness.
- Use patterns judiciously: Apply proven patterns (e.g., MVC, Observer, Adapter) as shared vocabulary and solutions—not cargo cults.
