# FuRBO — Constrained Trust‑Region Bayesian Optimization

This optimizer proposes boundary updates with a constrained qNEI acquisition
wrapped in a trust region. It models the (scalar) objective and a feasibility
signal, and adapts the search box based on success/failure. It is designed to
reduce expensive VMEC calls while improving the feasible hit‑rate.

## Usage

Single‑objective problems (P1/P2):

```
constelx opt run --baseline furbo \
  --nfp 3 --budget 60 --use-physics --problem p1 \
  --tr-init 0.2 --tr-min 0.02 --tr-max 0.5 \
  --tr-gamma-inc 1.6 --tr-gamma-dec 0.5 \
  --batch 2 --vmec-level low --vmec-hot-restart \
  --cache-dir .cache/eval
```

Multi‑objective (P3): start with `opt pareto` for a quick front; a qNEHVI
variant will land next. Meanwhile, `furbo` will operate on the scalar score.

## How it maps to boundaries

For quick runs, the search space is a 2D vector mapped to the main helical
pair: `r_cos[1][5] = -|x0|`, `z_sin[1][5] = +|x1|` with global bounds
`[-0.2, 0.2]` per coordinate. Full boundary‑mode support can be added once
GP conditioning and scaling are extended.

## Constraints

The optimizer consumes a vector of normalized constraints `c_tilde` where the
convention is `<= 0` is feasible. When per‑constraint details are unavailable,
it falls back to a single feasibility indicator derived from evaluator metrics.

Code: `src/constelx/problems/constraints.py`.

## Performance tips

- Enable parallel physics: `export CONSTELX_ALLOW_PARALLEL_REAL=1` and
  pass `--max-workers N` to agent/eval.
- Keep search at `--vmec-level low` with `--vmec-hot-restart`, then re‑score
  finalists at higher fidelity.
- Combine with proxy/surrogate gating in the agent for fastest end‑to‑end runs.

