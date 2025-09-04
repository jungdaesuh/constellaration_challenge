# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install system dependencies (macOS with Homebrew)
brew install netcdf

# Install the package with development dependencies
pip install -e ".[dev,bo]"
```

### Development Tools

```bash
# Run linting and formatting
ruff check .
ruff format .

# Run tests
pytest

# Run the main CLI
constelx --help
```

### Core CLI Commands

```bash
# Data operations
constelx data fetch --nfp 3 --limit 32

# Physics evaluation
constelx eval forward --example

# Optimization baselines
constelx opt baseline --algo cma-es --steps 50

# Surrogate model training
constelx surrogate train

# Agent loop (stub)
constelx agent run --iterations 3 --population 8
```

## Architecture Overview

This is a Python package for ML + physics-based optimization of stellarator plasma boundaries using the ConStellaration dataset. The project is structured as a CLI-first application with modular components.

### Key Modules

- **`constelx.cli`**: Main CLI interface with typer, organized into subcommands (`data`, `eval`, `opt`, `surrogate`, `agent`)
- **`constelx.physics`**: Physics wrappers around the `constellaration` package
  - `constel_api.py`: Core evaluation functions using ConStellaration metrics
  - `pcfm.py`, `pbfm.py`: Physics-constrained generation modules (stubs)
  - `constraints.py`: Hard constraint tooling
- **`constelx.data`**: Dataset access via HuggingFace datasets (`proxima-fusion/constellaration`)
- **`constelx.optim`**: Optimization algorithms (CMA-ES baseline, BoTorch stubs)
- **`constelx.surrogate`**: Simple MLP baseline + placeholders for advanced models

### Data Flow

1. **Data**: Fetch filtered subsets from HuggingFace dataset → cache as Parquet
2. **Physics**: Evaluate boundaries via ConStellaration's VMEC++ interface
3. **Optimization**: Use CMA-ES or BoTorch to optimize scalar metrics
4. **Surrogate**: Train ML models to replace expensive forward simulations
5. **Agent**: Multi-step propose→simulate→select→refine loop

### Dependencies

- **Core**: `constellaration>=0.1.6` for physics evaluation and VMEC++ interfaces
- **ML**: `torch` (optional on macOS arm64), `datasets`, `pandas`, `numpy`, `scipy`
- **CLI**: `typer`, `rich` for user interface
- **Optimization**: `cma` for evolution strategies, `botorch`+`gpytorch` for Bayesian optimization
- **Dev**: `pytest`, `ruff` for testing and linting

### System Requirements

- Python 3.10+
- NetCDF library (`brew install netcdf` on macOS, `libnetcdf-dev` on Ubuntu)
- PyTorch installation may need manual setup on macOS arm64

### Development Notes

- The codebase uses modern Python features (`from __future__ import annotations`)
- Ruff configuration: line length 100, basic linting enabled
- This is a starter/skeleton repo with many TODO stubs for extension
- Physics constraints (PCFM/PBFM) are placeholder modules for future implementation

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

## CLI behavior to implement

- `constelx data fetch --nfp 3 --limit 128`
- `constelx eval forward --boundary-file examples/boundary.json`
- `constelx eval score --metrics-file runs/<ts>/metrics.csv`
- `constelx opt cmaes --nfp 3 --budget 50 [--seed 0]`
- `constelx agent run --nfp 3 --budget 50 [--seed 0] [--resume PATH]`

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
