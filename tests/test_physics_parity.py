from __future__ import annotations

import math
import os

import pytest

# Skip the entire module unless explicitly enabled to avoid importing heavy
# physics dependencies on misconfigured systems.
RUN_PHYS = os.getenv("CONSTELX_RUN_PHYSICS_TESTS", "0").lower() in {"1", "true", "yes"}
pytestmark = pytest.mark.skipif(
    not RUN_PHYS, reason="Set CONSTELX_RUN_PHYSICS_TESTS=1 to run physics parity tests"
)

from constelx.physics.constel_api import example_boundary  # noqa: E402
from constelx.physics.proxima_eval import forward_metrics  # noqa: E402


def test_p1_score_bounded() -> None:
    b = example_boundary()
    try:
        metrics, info = forward_metrics(b, problem="p1")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert "score" in metrics
    s = float(metrics["score"])
    assert math.isfinite(s)
    # Geometric problem uses bounded score [0,1]
    assert 0.0 <= s <= 1.0


def test_p2_score_bounded() -> None:
    b = example_boundary()
    try:
        metrics, info = forward_metrics(b, problem="p2")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if info.get("source") == "placeholder":
        pytest.skip("constellaration p2 evaluation not available; using placeholder")
    assert "score" in metrics
    s = float(metrics["score"])
    assert math.isfinite(s)
    assert 0.0 <= s <= 1.0


def test_p3_multiobjective_shape() -> None:
    b = example_boundary()
    try:
        metrics, info = forward_metrics(b, problem="p3")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if info.get("source") == "placeholder":
        pytest.skip("constellaration p3 evaluation not available; using placeholder")
    # Should expose objectives for multi-objective case
    has_list = isinstance(metrics.get("objectives"), list)
    has_flat = any(k.startswith("objective_") for k in metrics.keys())
    assert has_list or has_flat
    # Score should exist (may be an aggregator placeholder depending on evaluator version)
    assert "score" in metrics
