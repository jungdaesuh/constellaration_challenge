from __future__ import annotations

from constelx.eval.boundary_param import sample_near_axis_qs, validate
from constelx.eval.geometry import quick_geometry_validate


def test_sample_near_axis_is_deterministic() -> None:
    b1 = sample_near_axis_qs(nfp=3, seed=123)
    b2 = sample_near_axis_qs(nfp=3, seed=123)
    assert b1 == b2
    validate(b1)


def test_near_axis_passes_quick_geometry() -> None:
    b = sample_near_axis_qs(nfp=3, seed=0, r0=1.0, epsilon=0.06)
    validate(b)
    ok, reason = quick_geometry_validate(b, r0_min=0.05, r0_max=5.0, helical_ratio_max=0.5)
    assert ok, f"near-axis seed should pass quick geometry guard, got: {reason}"
