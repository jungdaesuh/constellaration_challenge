from __future__ import annotations

import math

import pytest

pytest.importorskip("constellaration")

from constelx.physics.constel_api import example_boundary
from constelx.physics.proxima_eval import forward_metrics


def test_p1_score_bounded() -> None:
    b = example_boundary()
    metrics, info = forward_metrics(b, problem="p1")
    assert "score" in metrics
    s = float(metrics["score"])
    assert math.isfinite(s)
    # Geometric problem uses bounded score [0,1]
    assert 0.0 <= s <= 1.0


def test_p2_score_bounded() -> None:
    b = example_boundary()
    metrics, info = forward_metrics(b, problem="p2")
    if info.get("source") == "placeholder":
        pytest.skip("constellaration p2 evaluation not available; using placeholder")
    assert "score" in metrics
    s = float(metrics["score"])
    assert math.isfinite(s)
    assert 0.0 <= s <= 1.0


def test_p3_multiobjective_shape() -> None:
    b = example_boundary()
    metrics, info = forward_metrics(b, problem="p3")
    if info.get("source") == "placeholder":
        pytest.skip("constellaration p3 evaluation not available; using placeholder")
    # Should expose objectives for multi-objective case
    has_list = isinstance(metrics.get("objectives"), list)
    has_flat = any(k.startswith("objective_") for k in metrics.keys())
    assert has_list or has_flat
    # Score should exist (may be an aggregator placeholder depending on evaluator version)
    assert "score" in metrics
