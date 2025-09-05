from __future__ import annotations

from typing import Any, cast

from constellaration.geometry import surface_rz_fourier


def example_boundary() -> dict[str, Any]:
    """Return a tiny example stellarator-symmetric boundary in SurfaceRZFourier JSON form."""
    # 5x9 truncation for R_cos and Z_sin coefficients with NFP=3
    r_cos = [[0.0] * 9 for _ in range(5)]
    z_sin = [[0.0] * 9 for _ in range(5)]
    # minimal circle with small helical perturbation
    r_cos[0][4] = 1.0
    r_cos[1][5] = -0.05
    z_sin[1][5] = 0.05
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        r_sin=None,  # None for stellarator symmetric case
        z_cos=None,  # None for stellarator symmetric case
        z_sin=z_sin,
        n_field_periods=3,
        is_stellarator_symmetric=True,
    )
    return cast(dict[str, Any], boundary.model_dump())


def evaluate_boundary(boundary_json: dict[str, Any]) -> dict[str, Any]:
    """Compute placeholder metrics derived from boundary coefficients.

    This starter returns simple norms and a combined placeholder metric to keep
    tests fast and avoid heavy dependencies during CI. Replace with real physics
    evaluation when integrating with VMEC++ and full constellaration metrics.
    """
    boundary = surface_rz_fourier.SurfaceRZFourier.model_validate(boundary_json)

    r_cos_norm = float((boundary.r_cos**2).sum()) if boundary.r_cos is not None else 0.0
    z_sin_norm = float((boundary.z_sin**2).sum()) if boundary.z_sin is not None else 0.0

    return {
        "r_cos_norm": r_cos_norm,
        "z_sin_norm": z_sin_norm,
        "nfp": boundary.n_field_periods,
        "stellarator_symmetric": boundary.is_stellarator_symmetric,
        "placeholder_metric": r_cos_norm + z_sin_norm,
    }
