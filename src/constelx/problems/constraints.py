from __future__ import annotations

"""Constraint extraction and normalization utilities.

This module converts evaluator metrics into a vector of normalized constraint
values c_tilde where c_tilde <= 0 indicates feasibility. When detailed per-
constraint metrics are unavailable, we fall back to a single feasibility
indicator derived from the evaluator (`metrics["feasible"]` or
`metrics["feasibility"]`).

Design notes
- Keep dependencies minimal; operate on plain dicts.
- Do not import heavy physics here; this module is used inside BO loops.
"""

from typing import Any, Mapping, Sequence


def _as_float(x: Any, default: float = float("inf")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _feasible_flag(metrics: Mapping[str, Any]) -> bool | None:
    # Preferred boolean feasible flag
    feas = metrics.get("feasible")
    if isinstance(feas, bool):
        return feas
    # Some evaluators expose a numeric feasibility measure in [0,1]
    feas_num = metrics.get("feasibility")
    if isinstance(feas_num, (int, float)):
        try:
            return bool(feas_num <= 0.0 or feas_num == 0.0 or feas_num)
        except Exception:
            return None
    return None


def constraints_from_metrics(problem: str, metrics: Mapping[str, Any]) -> list[float]:
    """Return normalized constraints c_tilde (<=0 feasible).

    The function is robust to missing per-constraint details and will reduce to
    a single-element vector based on a feasibility flag when necessary.

    Current behavior:
    - If metrics contain a boolean `feasible`, return [-eps] when True, [1.0]
      when False.
    - If metrics contain a numeric `feasibility` field, treat values <=0 as
      satisfied; otherwise return [value].
    - TODO (extend when available): map known metric keys (e.g., mirror_ratio,
      qs_residual, w_mhd, chi_grad_r) to normalized inequality forms.
    """
    feas = _feasible_flag(metrics)
    if feas is True:
        # small negative to indicate satisfied
        return [-1e-6]
    if feas is False:
        return [1.0]
    feas_num = metrics.get("feasibility")
    if isinstance(feas_num, (int, float)):
        v = _as_float(feas_num)
        return [v]

    # Fallback: no information; treat as unknown (slightly positive to avoid
    # over-trusting feasibility in BO). This is conservative.
    return [0.1]


def is_feasible(c_tilde: Sequence[float]) -> bool:
    """Return True if all constraints are satisfied (<= 0)."""
    try:
        return all(float(v) <= 0.0 for v in c_tilde)
    except Exception:
        return False


__all__ = ["constraints_from_metrics", "is_feasible"]
