# Implementation roadmap

This repository ships with stubs and guardrails so you can iterate quickly. A recommended path:

1. **Dataset access**: `constelx data fetch` to materialize a filtered subset (e.g., NFP=3) into Parquet.
2. **Forward model sanity checks**: `constelx eval forward --example` to verify metrics computation works on your machine.
3. **Baselines**:
   - Start with `constelx opt baseline --algo cma-es` to optimize a few scalar metrics with simple shape parametrizations.
   - Try `constelx surrogate train` (MLP/DiT/FNO stubs) and `constelx surrogate eval` to replace the forward call in the inner loop.
4. **Agent loop**: glue propose→simulate→select in `constelx agent run`.
5. **Physics-constrained generation (optional)**:
   - Use `constelx.physics.pcfm` to project generated fields onto hard constraints (e.g., mass/flux consistency).
   - Use `constelx.physics.pbfm` training utilities to add residuals without conflicting with distribution learning.
6. **Submission packaging**: `constelx submit pack` to export your boundaries to the leaderboard format.
