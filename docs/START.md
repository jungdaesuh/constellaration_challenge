# Start Here — 24h Winning Runbook

This playbook gets you from a fresh shell to competitive, physics‑on
results for P1/P2/P3 in under a day. It is designed for humans and AI coding
agents: every step has an explicit command you can copy‑paste.

## 0) Prerequisites

```bash
# From repo root
python3 -m venv .venv && source .venv/bin/activate
brew install netcdf || sudo apt-get install -y libnetcdf-dev netcdf-bin
pip install -e '.[physics,bo]'
```

Performance toggles (recommended):

```bash
export CONSTELX_ALLOW_PARALLEL_REAL=1      # allow parallel VMEC in forward_many
export CONSTELX_REAL_TIMEOUT_MS=20000      # 20s per call
export CONSTELX_REAL_RETRIES=0             # no retries on stall
```

## 1) Seeds prior (optional but recommended)

```bash
constelx data fetch --nfp 3 --limit 1024
constelx data prior-train data/cache/subset.parquet \
  --out models/seeds_prior_hf_gmm.joblib
```

You can sample to inspect:

```bash
constelx data prior-sample models/seeds_prior_hf_gmm.joblib --count 8 --nfp 3 \
  --out data/prior_seeds.jsonl
```

## 2) Fast physics checks

Sanity smoke (real evaluator):

```bash
constelx eval forward --near-axis --use-physics --problem p1 --json \
  > runs/smoke_best.json
```

## 3) P1/P2 single‑objective — FuRBO (Trust‑Region BO)

Low‑fidelity search + hot restart, parallel VMEC workers:

```bash
# P1 — geometric
constelx opt run --baseline furbo \
  --nfp 3 --budget 60 --use-physics --problem p1 \
  --tr-init 0.2 --tr-min 0.02 --tr-max 0.5 \
  --tr-gamma-inc 1.6 --tr-gamma-dec 0.5 \
  --batch 2 --vmec-level low --vmec-hot-restart \
  --cache-dir .cache/eval

# P2 — simple-to-build QI (add PCFM QS band)
constelx opt run --baseline furbo \
  --nfp 3 --budget 80 --use-physics --problem p2 \
  --tr-init 0.2 --batch 2 --vmec-level low --vmec-hot-restart \
  --cache-dir .cache/eval \
  --correction pcfm --constraints-file examples/pcfm_qs_band.json
```

Tips:
- Combine with agent gating when exploring broader spaces: surrogate screen,
  Boozer proxy (`--mf-proxy`), novelty skip.
- Keep search at `--vmec-level low`; re‑score finalists at `medium`.

## 4) P3 multi‑objective — quick Pareto then refine

Quick sweep (physics‑on):

```bash
constelx opt pareto --budget 24 --sweeps 7 --use-physics --problem p3 \
  --json-out runs/p3_front.json
```

Refine promising regions with FuRBO (operates on scalar score for now):

```bash
constelx opt run --baseline furbo \
  --nfp 3 --budget 100 --use-physics --problem p3 \
  --tr-init 0.2 --batch 2 --vmec-level low --vmec-hot-restart \
  --cache-dir .cache/eval
```

## 5) Re‑score at higher fidelity & package

```bash
# Re-run eval forward on your top boundaries with --vmec-level medium
# (Example assumes you saved the best boundary as boundary.json)
constelx eval forward --boundary-json runs/<ts>/boundary.json \
  --use-physics --problem p1 --vmec-level medium --json \
  > runs/<ts>/best_rescored.json

# Pack submission
constelx submit pack runs/<ts> --out submissions/run_<ts>.zip --top-k 5
```

## 6) Performance knobs (recap)

- Parallel physics: `CONSTELX_ALLOW_PARALLEL_REAL=1` + `--max-workers` (agent)
- Timeouts: `CONSTELX_REAL_TIMEOUT_MS=20000`, `CONSTELX_REAL_RETRIES=0`
- Fidelity: search `--vmec-level low --vmec-hot-restart`, re‑score at `medium`
- Caching: always pass `--cache-dir .cache/eval`
- Gating: `--mf-proxy`, `--surrogate-screen`, `--novelty-skip`

## 7) Where to read more

- FuRBO optimizer: docs/FURBO.md
- Architecture map: docs/ARCHITECTURE.md
- Strategy and physics context: docs/STRATEGY.md, papers/ (benchmark and PCFM)
- Full “Win‑Tomorrow” checklist: TODO.md

## 8) Troubleshooting

- VMEC stalls or times out: keep 20s timeout and 0 retries; reduce `--batch`;
  ensure cache dir is on a fast disk; verify NetCDF and evaluator install.
- Low feasible hit‑rate: raise prior threshold (0.6–0.8), add PCFM QS band,
  gate harder on `--mf-proxy --mf-quantile 0.25`.
- BO explores too far: shrink TR (`--tr-gamma-dec 0.5`), reduce `--tr-max`.

```
# One-liner to verify CLI wiring
constelx opt run --baseline furbo --nfp 3 --budget 6 --problem p1 --no-use-physics
```

