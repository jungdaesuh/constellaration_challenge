# Dev Fixtures

Synthetic assets for fast unit tests and local smoke checks.

- `boundary.json`: deterministic synthetic boundary for placeholder path.
- `metrics_small.json`: minimal metrics table used by scoring tests.

These files require `CONSTELX_DEV=1` (or explicit `--allow-dev`) when run
through CLI commands that guard placeholder paths.
