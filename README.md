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
- **Hard-constraint tooling**: placeholders for PCFM and PBFM so you can enforce physical constraints during generation/training.

See `docs/ROADMAP.md` for the suggested implementation path.

## Citing

- Proxima + HF dataset + tools: see the dataset card and repo linked below.
- If you use PCFM/PBFM modules, please cite their respective arXiv preprints.

## Links

- Dataset: https://huggingface.co/datasets/proxima-fusion/constellaration
- Code/evaluator: https://github.com/proximafusion/constellaration
- VMEC++ docs: https://proximafusion.github.io/vmecpp/

License: MIT
