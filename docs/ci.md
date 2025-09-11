CI Performance and Coverage

Overview
- Core CI runs lint, type-checks, and tests on a lightweight environment.
- Physics CI installs system dependencies and runs tests that exercise the official evaluator.
- Test execution uses pytest-xdist to parallelize and reduce wall-clock time.

Key Choices
- Torch optionality: PyTorch is moved to an optional extra (surrogate) to avoid installing a large wheel on the core job, since surrogate tests are already guarded with skip-if when torch is absent.
- Physics gating: Physics tests are explicitly enabled in CI via the environment variable CONSTELX_RUN_PHYSICS_TESTS=1. Without this, the heavy physics dependencies would be installed yet the tests would remain skipped.
- Parallel tests: pytest-xdist (-n auto) is used in both jobs to parallelize collection and execution across available CPUs.
- Lean system deps: The physics job installs NetCDF/CMake/Ninja with --no-install-recommends to keep the footprint minimal.

How to Run Locally
- Core (no physics):
  - python -m venv .venv && source .venv/bin/activate
  - pip install -e ".[dev]"
  - ruff format --check . && ruff check . && mypy src/constelx && pytest -q -n auto
- Physics-enabled:
  - export CONSTELX_RUN_PHYSICS_TESTS=1
  - pip install -e ".[dev,physics]"
  - pytest -q -n auto

Surrogate (optional)
- Surrogate training requires PyTorch. To exercise surrogate tests locally:
  - pip install -e ".[dev,surrogate]"
  - pytest -q -n auto tests/test_cli_surrogate.py

Future Improvements
- If needed, introduce a dedicated surrogate CI job that installs the surrogate extra and runs only surrogate tests to keep the core job fast while still validating surrogate functionality.
- Consider using a prebuilt container image for the physics job with NetCDF/CMake/Ninja preinstalled to reduce bootstrap time further.

