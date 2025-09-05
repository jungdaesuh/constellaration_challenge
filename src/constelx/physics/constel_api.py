from __future__ import annotations

from typing import Any, Optional, cast


def _try_import_surface_rz_fourier() -> Optional[Any]:
    try:
        from constellaration.geometry import surface_rz_fourier

        return surface_rz_fourier
    except Exception:
        return None


def example_boundary() -> dict[str, Any]:
    """Return a tiny example stellarator-symmetric boundary in SurfaceRZFourier JSON form."""
    # 5x9 truncation for R_cos and Z_sin coefficients with NFP=3
    r_cos = [[0.0] * 9 for _ in range(5)]
    z_sin = [[0.0] * 9 for _ in range(5)]
    # minimal circle with small helical perturbation
    r_cos[0][4] = 1.0
    r_cos[1][5] = -0.05
    z_sin[1][5] = 0.05
    srf = _try_import_surface_rz_fourier()
    if srf is not None:
        boundary = srf.SurfaceRZFourier(
            r_cos=r_cos,
            r_sin=None,
            z_cos=None,
            z_sin=z_sin,
            n_field_periods=3,
            is_stellarator_symmetric=True,
        )
        return cast(dict[str, Any], boundary.model_dump())
    # Fallback: return plain dict without pydantic model
    return {
        "r_cos": r_cos,
        "r_sin": None,
        "z_cos": None,
        "z_sin": z_sin,
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }


def evaluate_boundary(boundary_json: dict[str, Any]) -> dict[str, Any]:
    """Compute placeholder metrics derived from boundary coefficients.

    This starter returns simple norms and a combined placeholder metric to keep
    tests fast and avoid heavy dependencies during CI. Replace with real physics
    evaluation when integrating with VMEC++ and full constellaration metrics.
    """
    srf = _try_import_surface_rz_fourier()
    if srf is not None:
        boundary = srf.SurfaceRZFourier.model_validate(boundary_json)
        r_cos_norm = float((boundary.r_cos**2).sum()) if boundary.r_cos is not None else 0.0
        z_sin_norm = float((boundary.z_sin**2).sum()) if boundary.z_sin is not None else 0.0
        return {
            "r_cos_norm": r_cos_norm,
            "z_sin_norm": z_sin_norm,
            "nfp": boundary.n_field_periods,
            "stellarator_symmetric": boundary.is_stellarator_symmetric,
            "placeholder_metric": r_cos_norm + z_sin_norm,
        }
    # Fallback: compute norms from plain lists
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore[assignment]

    r_cos = boundary_json.get("r_cos")
    z_sin = boundary_json.get("z_sin")
    if np is not None:
        r_cos_norm = float((np.asarray(r_cos) ** 2).sum()) if r_cos is not None else 0.0
        z_sin_norm = float((np.asarray(z_sin) ** 2).sum()) if z_sin is not None else 0.0
    else:

        def _norm2(a: Any) -> float:
            return float(sum(sum((float(x) ** 2 for x in row)) for row in a))

        r_cos_norm = _norm2(r_cos) if r_cos is not None else 0.0
        z_sin_norm = _norm2(z_sin) if z_sin is not None else 0.0
    return {
        "r_cos_norm": r_cos_norm,
        "z_sin_norm": z_sin_norm,
        "nfp": int(boundary_json.get("n_field_periods", 0)),
        "stellarator_symmetric": bool(boundary_json.get("is_stellarator_symmetric", True)),
        "placeholder_metric": r_cos_norm + z_sin_norm,
    }
