from __future__ import annotations

import math

from constelx.eval import score


def test_score_sums_numeric_ignores_non_numeric():
    metrics = {
        "a": 1.5,
        "b": 2.5,
        "note": "info",
        "flag": True,  # should be ignored (not a pure float/int in practical scoring)
    }
    # Only pure numeric values are summed; bool is subclass of int in Python
    # but we purposefully ignore non-float-like indicators by checking instance.
    # Here, we expect 1.5 + 2.5 = 4.0
    assert math.isfinite(score(metrics))
    assert score(metrics) == 4.0


def test_score_nan_returns_inf():
    m_bad = {"valid": 1.0, "invalid": float("nan")}
    s = score(m_bad)
    assert math.isinf(s) and s > 0
