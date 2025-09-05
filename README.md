# constelx — Starter repo for the ConStellaration challenge

**Goal:** provide a clean, extensible Python codebase for ML + physics-based optimization of stellarator plasma boundaries using the [ConStellaration dataset] and the [`constellaration`] evaluation tools.

> This is a *skeleton* repo: modules and CLIs are stubbed with TODOs and simple working examples so you can plug in your models and agents immediately.

## Quick start (macOS)

```bash
# (1) create and activate a virtual env (example: venv / conda)
python3 -m venv .venv && source .venv/bin/activate

# (2) install system deps (macOS) for VMEC++ I/O used by constellaration
# Requires Homebrew: https://brew.sh
brew install netcdf

# (3) install the package (editable) + extras for baselines
pip install -e ".[dev,bo]"

# (4) sanity check
constelx --help
constelx data fetch --nfp 3 --limit 32
constelx eval forward --example  # runs an example boundary through metrics
constelx eval forward --random --nfp 3 --seed 0
```

Notes:
- Python: this repo targets Python 3.10+ (`requires-python >=3.10`). Use `python3` on macOS.
- Apple Silicon (arm64): PyTorch is optional in this starter and not auto-installed on macOS arm64. If you plan to train surrogates, install a compatible PyTorch build from pytorch.org.
- Linux users: on Ubuntu, install `libnetcdf-dev` (and `cmake`) instead of Homebrew `netcdf`.

## What’s inside

- **CLI (`constelx`)**: `data` (fetch/filter/csv), `eval` (forward metrics, scoring), `opt` (baselines), `surrogate` (train/serve simple models), `agent` (multi-step propose→simulate→select loop).
- **Physics wrappers**: thin adapters around the `constellaration` package for metrics and VMEC++ boundary objects.
- **Optimization**: CMA-ES and (optional) BoTorch Bayesian optimization stubs.
- **Models**: simple MLP baseline + placeholders for FNO/transformers.
- **Hard-constraint tooling**: PCFM projection helpers and a PBFM conflict-free gradient update for physics-aware generation and training.

See `docs/ROADMAP.md` for the suggested implementation path.

## CLI usage

Evaluation
- `constelx eval forward --example`
- `constelx eval forward --random --nfp 3 --seed 0`
- `constelx eval score --metrics-json examples/metrics_small.json`

Optimization
- Toy sphere: `constelx opt cmaes --toy --budget 20 --seed 0`
- Boundary mode: `constelx opt cmaes --nfp 3 --budget 50 --seed 0`

Agent
- Random search: `constelx agent run --nfp 3 --budget 6 --seed 0 --runs-dir runs`
- CMA-ES (falls back to random if cma missing):
  `constelx agent run --nfp 3 --budget 20 --algo cmaes --seed 0`
- Resume a run: `constelx agent run --nfp 3 --budget 10 --resume runs/<ts>`
 - PCFM correction (examples):
   - Norm equality: constrain helical amplitude to a circle of radius 0.06
     - JSON: `examples/pcfm_norm.json`
     - Run: `constelx agent run --nfp 3 --budget 4 --correction pcfm --constraints-file examples/pcfm_norm.json`
   - Ratio equality: enforce `z_sin[1][5] / r_cos[1][5] = -1.25`
     - JSON: `examples/pcfm_ratio.json`
     - Run: `constelx agent run --nfp 3 --budget 4 --correction pcfm --constraints-file examples/pcfm_ratio.json`
   - Product equality: enforce `r_cos[1][5] * z_sin[1][5] = 0.003`
     - JSON: `examples/pcfm_product.json`
     - Run: `constelx agent run --nfp 3 --budget 4 --correction pcfm --constraints-file examples/pcfm_product.json`
   - Tuning: `--pcfm-gn-iters 3 --pcfm-damping 1e-6 --pcfm-tol 1e-8` or via top-level keys in the constraints JSON: `{gn_iters,damping,tol}`

Artifacts (written under `runs/<timestamp>/`)
- `config.yaml`: run config, env info, git SHA, package versions
- `proposals.jsonl`: proposals with seeds and boundaries
- `metrics.csv`: metrics for each proposal with computed `score`
- `best.json`: best score + metrics + boundary
- `README.md`: how the run was launched and environment details

## Development

Quality gates
- Format: `ruff format .`
- Lint: `ruff check .`
- Types: `mypy src/constelx`
- Tests: `pytest -q`

Pre-commit
- [![pre-commit enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
- Install hooks: `pip install -e ".[dev]" && pre-commit install`
- Run on all files: `pre-commit run -a`
- Update hook versions: `pre-commit autoupdate`
- Local equivalent of hooks: `ruff format . && ruff check . --fix && mypy --strict src/constelx && pytest -q`

CI
- Core job: ruff (format+lint), mypy, pytest — fast, Python-only path.
- Physics job: installs NetCDF system libs and `.[physics]`, then runs full tests.

## Physics stack installation

To run with the real evaluator dependencies:

- Ubuntu (CI/local):
  - `sudo apt-get update`
  - `sudo apt-get install -y libnetcdf-dev libnetcdf-cxx-legacy-dev cmake ninja-build`
  - `pip install -e ".[dev,physics]"`
- macOS (Homebrew):
  - `brew install netcdf`
  - `pip install -e ".[dev,physics]"`
- Windows: recommended via conda-forge:
  - `conda install -c conda-forge netcdf-c netcdf-cxx4 cmake ninja`
  - `pip install -e ".[dev,physics]"`

When physics extras are unavailable, the CLI and tests use lightweight placeholder evaluators that avoid native builds.

## Citing

- Proxima + HF dataset + tools: see the dataset card and repo linked below.
- If you use PCFM/PBFM modules, please cite their respective arXiv preprints.

## Links

- Dataset: https://huggingface.co/datasets/proxima-fusion/constellaration
- Code/evaluator: https://github.com/proximafusion/constellaration
- VMEC++ docs: https://proximafusion.github.io/vmecpp/

License: MIT
