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


def sample_random(
    nfp: int,
    seed: int,
    shape: Tuple[int, int] = (5, 9),
    *,
    use_fourier: bool = True,
) -> Dict[str, Any]:
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
    if use_fourier:
        # Build via BoundaryFourier to centralize indexing conventions
        from .boundary_fourier import BoundaryFourier

        n_offset = n // 2
        bf = BoundaryFourier.empty(nfp=int(nfp), m_dim=m, n_dim=n, n_offset=n_offset)
        # Base radius at validator’s conventional column j=4
        j_base = min(4, n - 1)
        bf.r_cos[0][j_base] = 1.0
        # Helical pair at (m=1, j=5) when available
        if m > 1 and n > j_base + 1:
            amp = 0.05
            bf.r_cos[1][j_base + 1] = float(-abs(rng.uniform(0, amp)))
            bf.z_sin[1][j_base + 1] = float(abs(rng.uniform(0, amp)))
        b = bf.to_surface_rz_fourier_dict()
        return b

    r_cos = _zeros(shape)
    z_sin = _zeros(shape)
    # For stellarator-symmetric case, r_sin and z_cos should be None
    r_sin: Coeff2D | None = None
    z_cos: Coeff2D | None = None

    # Set a base major radius on column index 4 (when available)
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


def sample_near_axis_qs(
    nfp: int,
    seed: int,
    shape: Tuple[int, int] = (5, 9),
    *,
    r0: float = 1.0,
    epsilon: float = 0.06,
    ellipticity: float = 0.0,
) -> Dict[str, Any]:
    """Generate a near-axis-inspired QS/QI-friendly boundary seed.

    This constructs a simple stellarator-symmetric boundary consistent with a
    small-ε near-axis picture: a circular base of radius ``r0`` and a modest
    helical perturbation at (m=1, n=+1). Optionally adds a tiny m=2 ellipticity.

    The construction is intentionally lightweight and deterministic, avoiding
    heavy near-axis libraries. It is suitable for seeding optimizers and for
    quick tests where QS/QI-friendly starts are desired.

    Parameters
    - nfp: Number of field periods (positive integer).
    - seed: RNG seed for determinism (used only to introduce tiny, bounded
      jitter on amplitudes to avoid duplicated seeds across runs).
    - shape: (m, n) truncation for Fourier coefficient grids.
    - r0: Base major radius term at (m=0, n=0) column index ~4.
    - epsilon: Relative helical amplitude as a fraction of r0 (0 < ε << 1).
    - ellipticity: Optional relative m=2 shaping magnitude (fraction of r0).

    Returns
    - A JSON-serializable dict matching `SurfaceRZFourier` fields.
    """
    if nfp <= 0:
        raise ValueError("nfp must be positive")

    import numpy as np  # local import to keep module import light

    rng = np.random.default_rng(seed)
    m_dim, n_dim = shape

    from .boundary_fourier import BoundaryFourier

    n_offset = n_dim // 2
    bf = BoundaryFourier.empty(nfp=int(nfp), m_dim=m_dim, n_dim=n_dim, n_offset=n_offset)

    # Base radius at a conventional column j≈4, clipped to grid
    j0 = min(4, n_dim - 1)
    bf.r_cos[0][j0] = float(max(1e-6, r0))

    # Helical pair at (m=1, n=+1) with QS/QI-friendly sign convention
    # r_cos(1, +1) negative, z_sin(1, +1) positive, magnitudes ~ ε*r0.
    if m_dim > 1 and n_dim > j0 + 1:
        amp = float(abs(epsilon) * bf.r_cos[0][j0])
        # add tiny deterministic jitter (±5%) from seed to diversify seeds
        jitter = 1.0 + float(rng.uniform(-0.05, 0.05))
        a = max(0.0, min(2.0 * amp, amp * jitter))
        bf.r_cos[1][j0 + 1] = -a
        bf.z_sin[1][j0 + 1] = +a

    # Optional mild ellipticity at (m=2, n=0)
    if ellipticity and m_dim > 2:
        e = float(abs(ellipticity) * bf.r_cos[0][j0])
        e = min(e, 0.2 * bf.r_cos[0][j0])
        try:
            i2, j2 = bf.idx(2, 0)
            bf.r_cos[i2][j2] = e
            # z_sin(2,0) often set 0 in stellarator-symmetric setups; keep 0.0
        except Exception:
            pass

    return bf.to_surface_rz_fourier_dict()


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

    required = ["r_cos", "z_sin", "n_field_periods", "is_stellarator_symmetric"]
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
    z_sin = as_coeff("z_sin")

    # r_sin and z_cos may be None for stellarator-symmetric boundaries
    r_sin_val = boundary.get("r_sin", None)
    z_cos_val = boundary.get("z_cos", None)
    r_sin: Coeff2D | None = None
    z_cos: Coeff2D | None = None
    if r_sin_val is not None:
        r_sin = as_coeff("r_sin")
    if z_cos_val is not None:
        z_cos = as_coeff("z_cos")

    # Shapes must all match
    # If provided, shapes must match; otherwise only check r_cos vs z_sin
    if not _check_same_shape(r_cos, z_sin):
        raise ValueError("Coefficient arrays r_cos and z_sin must have the same shape")
    if r_sin is not None and not _check_same_shape(r_cos, r_sin):
        raise ValueError(
            "Coefficient arrays r_cos and r_sin must have the same shape (if provided)"
        )
    if z_cos is not None and not _check_same_shape(r_cos, z_cos):
        raise ValueError(
            "Coefficient arrays r_cos and z_cos must have the same shape (if provided)"
        )

    # Bounds
    for name, arr in ("r_cos", r_cos), ("z_sin", z_sin):
        for row in arr:
            for x in row:
                if abs(x) > coeff_abs_max:
                    raise ValueError(f"{name} has coefficient out of bounds: {x}")
    if r_sin is not None:
        for row in r_sin:
            for x in row:
                if abs(x) > coeff_abs_max:
                    raise ValueError("r_sin has coefficient out of bounds: {x}")
    if z_cos is not None:
        for row in z_cos:
            for x in row:
                if abs(x) > coeff_abs_max:
                    raise ValueError("z_cos has coefficient out of bounds: {x}")

    # Heuristic: base radius positive
    if r_cos[0][min(4, len(r_cos[0]) - 1)] <= 0.0:
        raise ValueError("Base radius term (r_cos[0][4]) must be positive")
