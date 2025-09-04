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