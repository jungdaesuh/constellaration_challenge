"""Lightweight geometry validity checks.

These checks are heuristic by design and aim to cheaply discard obviously
invalid or pathological boundaries before expensive evaluations. They are
intended for use in pre-screening (agent guard) and not as authoritative
physics constraints.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple


def _get_base_radius(boundary: Mapping[str, Any]) -> float:
    r_cos = boundary.get("r_cos")
    if not isinstance(r_cos, list) or not r_cos or not isinstance(r_cos[0], list):
        return 0.0
    j0 = min(4, len(r_cos[0]) - 1)
    try:
        return float(r_cos[0][j0])
    except Exception:
        return 0.0


def _helical_m1_mag(boundary: Mapping[str, Any]) -> float:
    r_cos = boundary.get("r_cos")
    z_sin = boundary.get("z_sin")
    if not (
        isinstance(r_cos, list)
        and isinstance(z_sin, list)
        and len(r_cos) > 1
        and len(z_sin) > 1
        and isinstance(r_cos[1], list)
        and isinstance(z_sin[1], list)
    ):
        return 0.0
    total = 0.0
    ncols = min(len(r_cos[1]), len(z_sin[1]))
    for j in range(ncols):
        try:
            total += abs(float(r_cos[1][j])) + abs(float(z_sin[1][j]))
        except Exception:
            continue
    return float(total)


def quick_geometry_validate(
    boundary: Mapping[str, Any],
    *,
    r0_min: float = 0.05,
    r0_max: float = 5.0,
    helical_ratio_max: float = 0.5,
) -> Tuple[bool, str]:
    """Quick validity checks.

    - Base radius R0 in [r0_min, r0_max]
    - Sum of |m=1| helical amplitudes <= helical_ratio_max * R0

    Returns (ok, reason). Reason is non-empty when ok is False.
    """
    try:
        r0 = _get_base_radius(boundary)
        if not (r0_min <= r0 <= r0_max):
            return False, "invalid_r0"
        helical = _helical_m1_mag(boundary)
        if helical > helical_ratio_max * max(r0, 1e-9):
            return False, "helical_exceeds_ratio"
        return True, ""
    except Exception:
        return False, "geometry_check_error"
