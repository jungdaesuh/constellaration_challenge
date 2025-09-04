"""Physics-Based Flow Matching (PBFM) — placeholder.

Training utilities to add a physics residual term to the FM objective with *conflict-free* updates:
- Compute g_FM and g_R (residual) and align them using ConFIG-style projection.
- Support temporal unrolling to obtain accurate x_1 for residual evaluation.
- Optionally try a stochastic sampler at inference.
"""

from __future__ import annotations

from typing import Any


def conflict_free_update(g_fm: Any, g_r: Any) -> Any:
    """Return an update direction that improves both losses."""
    # TODO: implement ConFIG-style orthogonalization and unit-vector mix
    return g_fm
