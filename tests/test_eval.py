from __future__ import annotations

import math
import sys
import types

from constelx.eval import score


def test_score_sums_numeric_ignores_non_numeric() -> None:
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


def test_score_nan_returns_inf() -> None:
    m_bad = {"valid": 1.0, "invalid": float("nan")}
    s = score(m_bad)
    assert math.isinf(s) and s > 0


def test_score_records_scorer_fallback(monkeypatch, caplog) -> None:
    # Stub constellaration scorer to raise during invocation so we hit fallback path.
    scoring_mod = types.ModuleType("constellaration.metrics")

    def _boom(problem: str, metrics: dict[str, float]) -> float:
        raise RuntimeError("scorer exploded")

    scoring_mod.score = _boom  # type: ignore[attr-defined]
    pkg = types.ModuleType("constellaration")
    pkg.metrics = scoring_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "constellaration", pkg)
    monkeypatch.setitem(sys.modules, "constellaration.metrics", scoring_mod)

    caplog.set_level("WARNING", logger="constelx.physics.proxima_eval")

    metrics: dict[str, float] = {"a": 1.0, "b": 2.0}
    value = score(metrics, problem="p1")

    assert math.isfinite(value)
    assert metrics.get("scorer_fallback") is True
    warning = metrics.get("scorer_warning")
    assert isinstance(warning, str) and "scorer_import_failed" in warning
    assert any("scorer unavailable" in rec.message for rec in caplog.records)
