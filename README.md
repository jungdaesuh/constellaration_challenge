# constelx — Starter repo for the ConStellaration challenge

**Goal:** provide a clean, extensible Python codebase for ML + physics-based optimization of stellarator plasma boundaries using the [ConStellaration dataset] and the [`constellaration`] evaluation tools.

> This is a skeleton repo: modules and CLIs are stubbed with TODOs and simple working examples so you can plug in your models and agents immediately.

## Quick start (macOS)

```bash
# (1) create and activate a virtual env (example: venv / conda)
python3 -m venv .venv && source .venv/bin/activate

# (2) install system deps (macOS) for VMEC++ I/O used by constellaration
# Requires Homebrew: https://brew.sh
brew install netcdf

# (3) install the package (editable) + extras for baselines
pip install -e ".[dev,bo,evolution]"

# (4) sanity check
constelx --help
constelx data fetch --nfp 3 --limit 32
constelx data fetch --source hf --nfp 3 --limit 128  # real dataset → Parquet cache
constelx eval forward --example  # runs an example boundary through metrics
constelx eval forward --random --nfp 3 --seed 0
constelx data prior-train data/dataset.jsonl --out models/seeds_prior.joblib
constelx data prior-sample models/seeds_prior.joblib --count 16 --nfp 3 --min-feasibility 0.3
```

Notes:
- Python: this repo targets Python 3.10+ (`requires-python >=3.10`). Use `python3` on macOS.
- Apple Silicon (arm64): PyTorch is optional in this starter and not auto-installed on macOS arm64. If you plan to train surrogates, install a compatible PyTorch build from pytorch.org.
- Linux users: on Ubuntu, install `libnetcdf-dev` (and `cmake`) instead of Homebrew `netcdf`.

## What’s inside

- CLI (`constelx`): `data` (fetch/filter/csv), `eval` (forward metrics, scoring), `opt` (baselines), `surrogate` (train/serve simple models), `agent` (multi-step propose→simulate→select loop).
- Physics wrappers: thin adapters around the `constellaration` package for metrics and VMEC++ boundary objects, plus bounded Boozer-space QS/QI proxies (`constelx.physics.booz_proxy`).
- Proxy metrics: Boozer-space quasi-symmetry/isodynamic proxies with bounded residuals (`constelx.eval.boozer`). Placeholder + real evaluator paths populate `proxy_qs_*` / `proxy_qi_*` fields for early-stage guards.
- Optimization: CMA-ES and (optional) BoTorch Bayesian optimization stubs.
- Models: simple MLP baseline + placeholders for FNO/transformers.
- Hard-constraint tooling: PCFM projection helpers and a PBFM conflict-free gradient update for physics-aware generation and training.

See `docs/ROADMAP.md` (engineering roadmap) and `docs/STRATEGY.md`
(research/agent strategy) for the suggested path and background.

## Docs

- Engineering roadmap: `docs/ROADMAP.md`
- Strategy and agent playbook: `docs/STRATEGY.md`
- Contributor runbook: `AGENTS.md`
- Background/guide: `docs/GUIDELINE.md`

## CLI usage

Evaluation
- `constelx eval forward --example`
- `constelx eval forward --random --nfp 3 --seed 0`
- Near-axis QS/QI-friendly seed: `constelx eval forward --near-axis --nfp 3 --seed 0`
- `constelx eval score --metrics-json examples/metrics_small.json`

Evaluator knobs & parity
- `constelx eval problems` lists each challenge problem together with the metrics the
  evaluator expects so you can sanity-check parity before long runs.
- Timeout/backoff knobs: `CONSTELX_REAL_TIMEOUT_MS`, `CONSTELX_REAL_RETRIES`, and
  `CONSTELX_REAL_BACKOFF` tune the real evaluator timeout loop.
- Logging: set `CONSTELX_EVAL_LOG_DIR=/path/to/logs` to capture one JSON file per
  evaluation (boundary fingerprints, metrics, VMEC knobs). Combine with
  `CONSTELX_VMEC_VERBOSE=1` when you need verbose VMEC++ traces.
- Physics parity: install the `physics` extras, export
  `CONSTELX_RUN_PHYSICS_TESTS=1`, then run `pytest -q -k scoring_parity` to confirm the
  local scorer matches the official ConStellaration aggregation.

Optimization
- Toy sphere: `constelx opt cmaes --toy --budget 20 --seed 0`
- Boundary mode: `constelx opt cmaes --nfp 3 --budget 50 --seed 0`

Optimization baselines (trust‑constr / ALM / NGOpt)
- Trust‑constr (2D helical coefficients):
  `constelx opt run --baseline trust-constr --nfp 3 --budget 10`
- Augmented‑Lagrangian (simple penalty outer loop):
  `constelx opt run --baseline alm --nfp 3 --budget 10`
- DESC trust-region (DESC gradients + resolution ladder):
  `constelx opt run --baseline desc-trust --nfp 3 --budget 10`
  Requires `pip install -e ".[desc]"` (or `constelx[desc]`) and runs a two-stage
  DESC trust-region SQP pass (coarse M=8/N=8 then refined M=12/N=12) before
  scoring via the shared evaluator.
- Nevergrad NGOpt (augmented-Lagrangian polisher):
  `constelx opt run --baseline ngopt --nfp 3 --budget 10`
- With physics path (requires problem id):
  `constelx opt run --baseline trust-constr --nfp 3 --budget 10 --use-physics --problem p1`
- Pareto sweep (Problem 3 placeholder):
  `constelx opt pareto --budget 16 --sweeps 5 --seed 0 --json-out pareto.json`
  When `--use-physics` is set, metrics and scoring route through the official evaluator
  if available; otherwise the command falls back to the placeholder path.
  (Install the `evolution` extra to enable CMA-ES and NGOpt: `pip install -e ".[evolution]"`.)

Agent
- Random search: `constelx agent run --nfp 3 --budget 6 --seed 0 --runs-dir runs`
- Near-axis seeding: `constelx agent run --nfp 3 --budget 6 --seed-mode near-axis`
- Data prior seeding: `constelx agent run --nfp 3 --budget 10 --seed-mode prior --seed-prior models/seeds_prior.joblib`
- CMA-ES (falls back to random if cma missing):
  `constelx agent run --nfp 3 --budget 20 --algo cmaes --seed 0`
- Resume a run: `constelx agent run --nfp 3 --budget 10 --resume runs/<ts>`
- Multi-start NFP exploration (round-robin across values):
  `constelx agent run --nfp-list "3,4,5" --budget 12 --seed 0`
  The budget is shared across NFPs and proposals are allocated in a round-robin fashion. Artifacts include an `nfp` field in `proposals.jsonl`, `metrics.csv`, and in `boundaries.jsonl` (when packing submissions with `--top-k`).
- Physics knobs: `constelx agent run --use-physics --problem p1 --vmec-level medium --vmec-hot-restart --vmec-restart-key run001`
  (`--vmec-level`, `--vmec-hot-restart`, and `--vmec-restart-key` also work with `constelx eval forward` and `constelx opt run`).
- Geometry guard: `--guard-geom-validate` pre-screens invalid shapes and logs
  `fail_reason=invalid_geometry` without spending evaluator calls.
   - Thresholds can be tuned: `--guard-r0-min`, `--guard-r0-max`, and
     `--guard-helical-ratio-max`.

Novelty gating (skip near-duplicates)
- Enable skip: `--novelty-skip` to avoid spending evaluator calls on proposals too close to recent ones.
- Tuning:
  - Metric: `--novelty-metric l2|cosine|allclose` (default `l2`)
  - Threshold: `--novelty-eps <float>` (distance <= eps → duplicate; default `1e-6`)
  - Window: `--novelty-window <N>` recent proposals kept per NFP (default `128`)
  - Persistence: `--novelty-db path.jsonl` to persist novelty across runs (defaults to `runs/<ts>/novelty.jsonl` when enabled)
- Logging: duplicates are recorded to `metrics.csv` with `feasible=False` and `fail_reason=duplicate_novelty` without consuming budget.

Surrogate screening (proxy gating before evaluator)
- Enable: `--surrogate-screen --surrogate-model outputs/surrogates/mlp/mlp.pt`
  - Optional: `--surrogate-metadata PATH` (defaults to `metadata.json` alongside the model)
- Tuning:
  - Threshold: `--surrogate-threshold <float>` keeps candidates with predicted score ≤ threshold.
  - Quantile: `--surrogate-quantile <q>` (default `0.5`) when no threshold is set; keep-lowest fraction per batch.
  - Cap survivors: `--surrogate-keep-max <K>` to bound the number forwarded to the evaluator.
- Logging: filtered entries appear in `metrics.csv` with `fail_reason=filtered_surrogate`, `phase=surrogate`, and a `surrogate_score` column.


Multi-fidelity proxy gating
- Enable: `--mf-proxy` to evaluate a cheap Boozer-space proxy before expensive calls.
- Flags: `--mf-proxy [--mf-threshold <t> | --mf-quantile <q>] [--mf-max-high K] [--mf-proxy-metric metric]`
  with metrics drawn from the unified facade (`score`, `placeholder_metric`, `qs_residual`,
  `qi_residual`, `helical_energy`, `mirror_ratio`).
  - Metrics: `metric` defaults to `score`; choose `qs_residual`, `qi_residual`, or `helical_energy` to gate on individual proxies.
- Behavior: computes proxy metrics for each batch, keeps survivors by threshold/quantile on the selected metric, optionally caps them to `K`, then routes survivors to real/placeholder evaluation.
- Provenance: proxy rows in `metrics.csv` include `phase=proxy`, the `proxy_metric` name, and the numeric `proxy_score`. Survivors evaluated downstream retain existing fields with `phase=real` when physics is enabled.
- Caching: proxy evaluations are cached separately with a `:proxy` suffix so they do not collide with real results.

### PCFM correction (Gauss–Newton projection)

Projects boundaries onto the constraint manifold before evaluation using a damped Gauss–Newton
step with clamped updates and geometry revalidation at every iteration. This keeps proposals
feasible without diverging when the Jacobian is ill conditioned.

**Example spec**

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

Launch with:

`constelx agent run --nfp 3 --budget 4 --correction pcfm --constraints-file examples/pcfm_norm.json`

**Ready-to-use specs**

- Norm equality: constrain helical amplitude to a circle of radius 0.06 -> `examples/pcfm_norm.json`
- Ratio equality: enforce `z_sin[1][5] / r_cos[1][5] = -1.25` -> `examples/pcfm_ratio.json`
- Product equality: enforce `r_cos[1][5] * z_sin[1][5] = 0.003` -> `examples/pcfm_product.json`
- Aspect-ratio band: keep `R0/a` within `[4, 8]` -> `examples/pcfm_ar_band.json`
- Edge iota ratio: target helical ratio ~= 0.1 -> `examples/pcfm_edge_iota.json`
- QS residual band: clamp `qs_residual` <= 0.2 with Boozer proxies -> `examples/pcfm_qs_band.json`
- Clearance floor: ensure `|R0| - ||m=1|| >= 0.95` -> `examples/pcfm_clearance.json`

Tuning via CLI flags (`--pcfm-gn-iters 3 --pcfm-damping 1e-6 --pcfm-tol 1e-8`) or top-level JSON
keys `{gn_iters,damping,tol}`.
Artifacts (written under `runs/<timestamp>/`)
- `config.yaml`: run config, env info, git SHA, package versions
- `proposals.jsonl`: proposals with seeds and boundaries
- `metrics.csv`: metrics for each proposal with computed `score`
- `best.json`: best score + metrics + boundary
- `README.md`: how the run was launched and environment details

Submission
- Pack a completed run into a submission zip:
  `constelx submit pack runs/<ts> --out submissions/run_<ts>.zip`
  Includes `boundary.json`, `best.json` (if present), and a small `metadata.json`.
 - Include the top-K boundaries by aggregate score:
   `constelx submit pack runs/<ts> --out submissions/run_<ts>.zip --top-k 5`
   This adds `boundaries.jsonl` with records of the form `{iteration,index,agg_score,evaluator_score,feasible,fail_reason,source,scoring_version,boundary}`.

Ablation
- Quick component ablations: `constelx ablate run --nfp 3 --budget 6 --components guard_simple,mf_proxy`
  - Also supports `correction=eci_linear` and `correction=pcfm` toggles (uses simple defaults).
  - Writes a timestamped folder under `runs/ablations/` with per-variant subfolders and `summary.csv`/`summary.json` at the root.
 - Spec-driven plan with multi-seed aggregation:
   - Create `plan.json`:
     `{ "base": {"nfp":3, "budget":3}, "seeds": [0,1], "variants": [{"name":"baseline","overrides":{}},{"name":"eci_linear","overrides":{"correction":"eci_linear","constraints":[{"rhs":0.0,"coeffs":[{"field":"r_cos","i":1,"j":5,"c":1.0},{"field":"z_sin","i":1,"j":5,"c":1.0}]}]}}] }`
   - Run: `constelx ablate run --spec plan.json --runs-dir runs/ablations`
   - Produces `details.csv` (per-variant/seed) and `summary.csv` (best and mean agg_score per variant).

## Development

Quality gates
- Format: `ruff format .`
- Lint: `ruff check .`
- Types: `mypy src/constelx`
- Tests: `pytest -q`

Physics parity tests
- Parity tests for P1–P3 are included and gated by physics deps to keep CI fast.
- Enable locally with real evaluator installed: `export CONSTELX_RUN_PHYSICS_TESTS=1` then run `pytest -q -k scoring_parity`.
- These tests assert our `eval.score(..., problem=...)` matches the official aggregator for each problem.

Pre-commit
- [![pre-commit enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
- Install hooks: `pip install -e ".[dev]" && pre-commit install`
- Run on all files: `pre-commit run -a`
- Update hook versions: `pre-commit autoupdate`
- Local equivalent of hooks: `ruff format . && ruff check . --fix && mypy --strict src/constelx && pytest -q`

CI
- Core job: ruff (format+lint), mypy, pytest — fast, Python-only path.
- Physics job: installs NetCDF system libs and `.[physics]`, then runs full tests (including P1–P3 parity).

## Physics stack installation

To run with the real evaluator dependencies:

- Ubuntu (CI/local):
  - `sudo apt-get update`
  - `sudo apt-get install -y libnetcdf-dev libnetcdf-cxx-legacy-dev cmake ninja-build`
  - `pip install -e ".[dev,physics]"`
- macOS (Homebrew):
  - `brew install netcdf`
  - `pip install -e ".[dev,physics]"`

  macOS notes (booz_xform/NetCDF):
  - Some prebuilt macOS wheels for `booz_xform` may look for an older Homebrew NetCDF dylib (e.g., `libnetcdf.19.dylib`) while Homebrew provides a newer one (e.g., `libnetcdf.22.dylib`). If you see an ImportError mentioning `libnetcdf.*.dylib` when importing `booz_xform`, reinstall `booz_xform` from source linked against your local NetCDF:
    - `brew install netcdf cmake ninja`
    - `export NETCDF_DIR=/opt/homebrew/opt/netcdf`
    - `export LDFLAGS="-L/opt/homebrew/opt/netcdf/lib"`
    - `export CPPFLAGS="-I/opt/homebrew/opt/netcdf/include"`
    - `PIP_NO_BINARY=booz_xform pip install -v git+https://github.com/hiddenSymmetries/booz_xform.git`
  - Optional: to avoid thread oversubscription during real evaluations, set:
    - `export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
- Windows: recommended via conda-forge:
  - `conda install -c conda-forge netcdf-c netcdf-cxx4 cmake ninja`
  - `pip install -e ".[dev,physics]"`
- DESC trust-region baseline (gradient inner loop with DESC):
  - `pip install -e ".[dev,desc]"` or `pip install constelx[desc]`
  - Includes `desc-opt` and its JAX stack; reuse the NetCDF guidance above when
    building on macOS or Linux to satisfy DESC’s geometric I/O dependencies.

When physics extras are unavailable, the CLI and tests use lightweight placeholder evaluators that avoid native builds.

## Environment (.env) and evaluator knobs

The CLI and evaluator auto-load a local `.env` if `python-dotenv` is installed (included in the `dev` extra). Useful variables:

- `CONSTELX_USE_REAL_EVAL`: `1|true` to route eval to the real physics path when available.
- `CONSTELX_ALLOW_PARALLEL_REAL`: `1|true` to enable parallel real-eval in `forward_many`.
- `CONSTELX_REAL_TIMEOUT_MS`: per-call timeout in milliseconds (default `20000`).
- `CONSTELX_REAL_RETRIES`: number of retries on timeout/error (default `1`).
- `CONSTELX_REAL_BACKOFF`: multiplicative backoff factor between retries (default `1.5`).
- `CONSTELX_CACHE_TTL_SECONDS`: optional TTL (in seconds) for evaluator cache entries when
  `diskcache` is available (install with `.[cache]`). After TTL, cached results expire and will be
  recomputed. JSON fallback ignores TTL.
- `CONSTELX_VMEC_LEVEL`: choose VMEC resolution ladder (`auto|low|medium|high`).
- `CONSTELX_VMEC_HOT_RESTART`: `1|true` to enable VMEC hot restart when supported.
- `CONSTELX_VMEC_RESTART_KEY`: identifier used to reuse VMEC restart states across calls.

Caching
- Evaluator results are cached to speed up repeated calls. Default cache dir is `.cache/eval/`.
- To keep repositories clean and runs isolated, prefer per-run caches:
  pass `--cache-dir runs/<ts>/cache` to `constelx eval/agent` commands. The `runs/` tree is ignored.

Artifacts now include clear scoring and provenance fields:
- CSV: `nfp`, `evaluator_score`, `agg_score`, `elapsed_ms`, `feasible`, `fail_reason`, `source`.
- `best.json`: `agg_score`, optional `evaluator_score`, and metrics without a conflicting `score` key.

## Citing

- Proxima + HF dataset + tools: see the dataset card and repo linked below.
- If you use PCFM/PBFM modules, please cite their respective arXiv preprints.

## Links

- Dataset: https://huggingface.co/datasets/proxima-fusion/constellaration
- Code/evaluator: https://github.com/proximafusion/constellaration
- VMEC++ docs: https://proximafusion.github.io/vmecpp/

License: MIT
