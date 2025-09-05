"""Boundary parameterization and validation utilities.

This module provides a tiny, fixed-basis parameterization of stellarator-symmetric
`SurfaceRZFourier` boundaries and validation helpers suitable for quick experiments
and tests. It avoids importing third-party types to keep type checking strict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

Coeff2D = List[List[float]]


def _zeros(shape: Tuple[int, int]) -> Coeff2D:
    m, n = shape
    return [[0.0 for _ in range(n)] for _ in range(m)]


def sample_random(nfp: int, seed: int, shape: Tuple[int, int] = (5, 9)) -> Dict[str, Any]:
    """Generate a random, bounded boundary dict in a small fixed basis.

    - Stellarator-symmetric: r_sin, z_cos present but set to zeros.
    - Base circle (R0) is set positive; small helical perturbations added.

    Parameters
    - nfp: Number of field periods (positive integer).
    - seed: RNG seed for determinism.
    - shape: (m, n) truncation for Fourier coefficient grids.

    Returns
    - A JSON-serializable dict matching `SurfaceRZFourier` fields.
    """

    if nfp <= 0:
        raise ValueError("nfp must be positive")

    import numpy as np  # local import to keep module import light

    rng = np.random.default_rng(seed)
    m, n = shape
    r_cos = _zeros(shape)
    z_sin = _zeros(shape)
    r_sin = _zeros(shape)
    z_cos = _zeros(shape)

    # Set a base major radius on the (m=0, n=nfp+1) term (index nfp+2 considering zero-index)
    # Using index 4 like the example (works for n >= 5)
    idx_n = min(4, n - 1)
    r_cos[0][idx_n] = 1.0

    # Add small helical perturbations in (m=1, n=idx_n+1) if available
    if m > 1 and n > idx_n + 1:
        amp = 0.05
        r_cos[1][idx_n + 1] = float(-abs(rng.uniform(0, amp)))
        z_sin[1][idx_n + 1] = float(abs(rng.uniform(0, amp)))

    return {
        "r_cos": r_cos,
        "r_sin": r_sin,
        "z_cos": z_cos,
        "z_sin": z_sin,
        "n_field_periods": int(nfp),
        "is_stellarator_symmetric": True,
    }


def _check_same_shape(a: Coeff2D, b: Coeff2D) -> bool:
    return len(a) == len(b) and all(len(ai) == len(bi) for ai, bi in zip(a, b))


def validate(boundary: Mapping[str, Any], *, coeff_abs_max: float = 2.0) -> None:
    """Validate a boundary dict for basic shape and bounds.

    Checks
    - Required keys exist and have expected types.
    - Coefficient arrays are 2D lists of floats with matching shapes across fields.
    - Absolute value bound on all coefficients.
    - Base radius term positive (heuristic sanity check).

    Raises
    - ValueError with a descriptive message on first failed check.
    """

    required = ["r_cos", "r_sin", "z_cos", "z_sin", "n_field_periods", "is_stellarator_symmetric"]
    for k in required:
        if k not in boundary:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(boundary["n_field_periods"], int) or boundary["n_field_periods"] <= 0:
        raise ValueError("n_field_periods must be a positive int")
    if boundary["is_stellarator_symmetric"] is not True:
        raise ValueError("is_stellarator_symmetric must be True in this starter")

    def as_coeff(name: str) -> Coeff2D:
        v = boundary[name]
        if not isinstance(v, list) or not v or not isinstance(v[0], list):
            raise ValueError(f"{name} must be a 2D list")
        try:
            return [[float(x) for x in row] for row in v]
        except Exception as e:  # noqa: BLE001 - convert to ValueError with context
            raise ValueError(f"{name} must contain numeric values: {e}")

    r_cos = as_coeff("r_cos")
    r_sin = as_coeff("r_sin")
    z_cos = as_coeff("z_cos")
    z_sin = as_coeff("z_sin")

    # Shapes must all match
    if not (
        _check_same_shape(r_cos, r_sin)
        and _check_same_shape(r_cos, z_cos)
        and _check_same_shape(r_cos, z_sin)
    ):
        raise ValueError("Coefficient arrays must have the same shape")

    # Bounds
    for name, arr in ("r_cos", r_cos), ("r_sin", r_sin), ("z_cos", z_cos), ("z_sin", z_sin):
        for row in arr:
            for x in row:
                if abs(x) > coeff_abs_max:
                    raise ValueError(f"{name} has coefficient out of bounds: {x}")

    # Heuristic: base radius positive
    if r_cos[0][min(4, len(r_cos[0]) - 1)] <= 0.0:
        raise ValueError("Base radius term (r_cos[0][4]) must be positive")
