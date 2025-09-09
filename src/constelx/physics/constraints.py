"""Constraint placeholders for physics-aware optimization/generation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _base_and_minor(boundary: Dict[str, Any]) -> Tuple[float, float]:
    """Return approximations of (major radius R0, minor radius a)."""
    r_cos = boundary.get("r_cos")
    z_sin = boundary.get("z_sin")
    if not (
        isinstance(r_cos, list)
        and r_cos
        and isinstance(r_cos[0], list)
        and isinstance(z_sin, list)
        and len(z_sin) > 1
        and isinstance(z_sin[1], list)
        and len(r_cos) > 1
        and isinstance(r_cos[1], list)
    ):
        return 0.0, 0.0
    j0 = min(4, len(r_cos[0]) - 1, len(r_cos[1]) - 1, len(z_sin[1]) - 1)
    try:
        r0 = float(r_cos[0][j0])
        r1 = float(r_cos[1][j0])
        z1 = float(z_sin[1][j0])
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0
    minor = float(np.hypot(r1, z1))
    return r0, minor


def aspect_ratio(boundary: Dict[str, Any]) -> float:
    """Approximate aspect ratio as ``R0 / a`` from Fourier coefficients.

    ``R0`` is taken from the base radius coefficient and ``a`` from the
    magnitude of the ``m=1`` terms. Returns ``0.0`` when unavailable or
    degenerate.
    """
    r0, minor = _base_and_minor(boundary)
    if minor <= 0.0:
        return 0.0
    return abs(r0) / minor


def curvature_smoothness(boundary: Dict[str, Any]) -> float:
    """Rough smoothness proxy based on second differences of ``m=1`` terms."""
    r_cos = boundary.get("r_cos")
    z_sin = boundary.get("z_sin")
    if not (
        isinstance(r_cos, list)
        and len(r_cos) > 1
        and isinstance(r_cos[1], list)
        and isinstance(z_sin, list)
        and len(z_sin) > 1
        and isinstance(z_sin[1], list)
    ):
        return 0.0
    r1 = np.asarray(r_cos[1], dtype=float)
    z1 = np.asarray(z_sin[1], dtype=float)
    if r1.size < 3 or z1.size < 3:
        return 0.0
    dr2 = np.diff(r1, n=2)
    dz2 = np.diff(z1, n=2)
    curv = dr2 * dr2 + dz2 * dz2
    return float(np.mean(curv)) if curv.size else 0.0


__all__ = ["aspect_ratio", "curvature_smoothness"]
