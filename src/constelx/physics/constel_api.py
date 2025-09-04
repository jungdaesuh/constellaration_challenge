from __future__ import annotations

import math
from typing import Dict, Any

from constellaration.utils import vmec_utils, surface_rz_fourier
from constellaration.metrics import scoring

def example_boundary() -> dict:
    """Return a tiny example axisymmetric boundary in SurfaceRZFourier JSON form."""
    # 5x9 truncation for R_cos and Z_sin coefficients with NFP=3
    r_cos = [[0.0]*9 for _ in range(5)]
    z_sin = [[0.0]*9 for _ in range(5)]
    # minimal circle with small helical perturbation
    r_cos[0][4] = 1.0
    r_cos[1][5] = -0.05
    z_sin[1][5] = 0.05
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        r_sin=[[0.0]*9 for _ in range(5)],
        z_cos=[[0.0]*9 for _ in range(5)],
        z_sin=z_sin,
        n_field_periods=3,
        is_stellarator_symmetric=True,
    )
    return boundary.model_dump()

def evaluate_boundary(boundary_json: dict) -> Dict[str, Any]:
    """Compute a few metrics via constellaration's scoring helpers."""
    boundary = surface_rz_fourier.SurfaceRZFourier.model_validate(boundary_json)
    # Minimal VMEC++ "WOut" surrogate from boundary only
    # For richer metrics you can sample from the HF dataset and reuse its vmecpp_wout
    vmecpp_wout = vmec_utils.minimal_wout_from_boundary(boundary)
    # Example: smoothness/compactness proxy and simple shape descriptors
    values = scoring.geom_metrics(boundary, vmecpp_wout)
    return values
