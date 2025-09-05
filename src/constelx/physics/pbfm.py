"""Physics-Based Flow Matching (PBFM) — placeholder.

Training utilities to add a physics residual term to the FM objective with *conflict-free* updates:
- Compute g_FM and g_R (residual) and align them using ConFIG-style projection.
- Support temporal unrolling to obtain accurate x_1 for residual evaluation.
- Optionally try a stochastic sampler at inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def conflict_free_update(g_fm: ArrayLike, g_r: ArrayLike) -> NDArray[np.floating[Any]]:
    """Return an update direction that improves both losses.

    Aligns the residual gradient ``g_r`` with the flow-matching gradient
    ``g_fm`` using a ConFIG-style projection. The two unit-normalized
    directions are summed and normalized again to yield a single unit
    direction that balances both objectives.

    Parameters
    ----------
    g_fm, g_r:
        Gradient vectors for the flow-matching and residual losses. Both must
        have the same shape.

    Returns
    -------
    numpy.ndarray
        Combined unit direction for ``g_fm`` and the orthogonalized ``g_r``.

    Examples
    --------
    >>> conflict_free_update([1.0, 0.0], [-1.0, 1.0])
    array([0.70710678, 0.70710678])
    """

    g_fm_arr = np.asarray(g_fm, dtype=float)
    g_r_arr = np.asarray(g_r, dtype=float)
    if g_fm_arr.shape != g_r_arr.shape:
        raise ValueError("g_fm and g_r must have the same shape")

    dot = float(np.dot(g_fm_arr, g_r_arr))
    if dot < 0.0:
        proj = (dot / np.dot(g_fm_arr, g_fm_arr)) * g_fm_arr
        g_r_arr = g_r_arr - proj

    def _unit(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        norm = float(np.linalg.norm(x))
        return x / norm if norm > 0 else x

    u_fm = _unit(g_fm_arr)
    u_r = _unit(g_r_arr)
    return _unit(u_fm + u_r)
