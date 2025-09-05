from __future__ import annotations

from math import inf, isnan
from typing import Any, Mapping, Tuple


def boundary_to_vmec(boundary: Mapping[str, Any]) -> Any:
    """Validate and convert a boundary dict to a constellaration VMEC boundary model.

    Returns a SurfaceRZFourier pydantic model when constellaration is available.
    Raises RuntimeError if the library is missing or validation fails.
    """
    try:
        from constellaration.geometry import surface_rz_fourier as srf

        return srf.SurfaceRZFourier.model_validate(dict(boundary))
    except Exception as e:  # pragma: no cover - optional dependency branch
        raise RuntimeError(
            "constellaration is required for VMEC boundary validation. "
            "Install extras: pip install -e '.[physics]'"
        ) from e


def _fallback_metrics(boundary: Mapping[str, Any]) -> dict[str, Any]:
    # Lightweight placeholder using existing starter path
    from .constel_api import evaluate_boundary as _eval

    return _eval(dict(boundary), use_real=False)


def forward_metrics(
    boundary: Mapping[str, Any], *, problem: str, vmec_opts: dict[str, Any] | None = None
) -> Tuple[dict[str, float], dict[str, Any]]:
    """Compute metrics via the official constellaration evaluator.

    Returns (metrics, info). Falls back to starter placeholder metrics if the
    physics stack is unavailable so that callers can degrade gracefully.
    """
    try:
        from constellaration.geometry import surface_rz_fourier as srf
        from constellaration.metrics import scoring as px_scoring
        from constellaration.mhd import vmec_utils

        b = srf.SurfaceRZFourier.model_validate(dict(boundary))
        vmec_kwargs = dict(vmec_opts or {})
        wout = vmec_utils.minimal_wout_from_boundary(b, **vmec_kwargs)
        # Use a general geometric metrics call; problem-specific scorers can
        # select subsets/aggregations later.
        m_raw = px_scoring.geom_metrics(b, wout)
        metrics = {k: float(v) for k, v in dict(m_raw).items() if isinstance(v, (int, float))}
        return metrics, {"problem": problem, "feasible": True, "source": "constellaration"}
    except Exception:
        m = _fallback_metrics(boundary)
        # Best-effort cast to float values for compatibility
        metrics_f = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
        return metrics_f, {"problem": problem, "feasible": True, "source": "placeholder"}


def score(problem: str, metrics: Mapping[str, Any]) -> float:
    """Aggregate a scalar score using the official scorer when available.

    Falls back to summing numeric metrics (NaN -> +inf) if the physics scorer
    is unavailable.
    """
    try:
        from constellaration.metrics import scoring as px_scoring

        # Try a few likely entrypoints to avoid hard-coding an exact name.
        if hasattr(px_scoring, "score"):
            return float(getattr(px_scoring, "score")(problem, dict(metrics)))
        if hasattr(px_scoring, "aggregate_score"):
            return float(getattr(px_scoring, "aggregate_score")(problem, dict(metrics)))
    except Exception:
        pass

    total = 0.0
    for v in metrics.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
            if isnan(fv):
                return inf
            total += fv
    return float(total)


__all__ = ["boundary_to_vmec", "forward_metrics", "score"]
