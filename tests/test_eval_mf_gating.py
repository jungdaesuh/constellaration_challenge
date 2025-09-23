from __future__ import annotations

import math
from typing import Any

import pytest

from constelx.eval import MF_PROXY_METRICS, forward_many
from constelx.eval import score as eval_score
from constelx.physics.constel_api import example_boundary


def _make_boundary(scale: float) -> dict[str, Any]:
    b = example_boundary()
    # Increase/decrease helical amplitude to modulate placeholder_metric deterministically
    b["r_cos"][1][5] = float(-abs(scale))
    b["z_sin"][1][5] = float(abs(scale))
    return b


def test_mf_gating_selects_by_threshold_and_sets_phase() -> None:
    # Build a small batch with clearly separated proxy scores
    bad = _make_boundary(0.20)  # larger amplitudes -> worse placeholder_metric
    mid = _make_boundary(0.10)
    good = _make_boundary(0.02)
    batch = [bad, mid, good]

    # Run with mf_proxy enabled and gate on the QS residual proxy.
    results = forward_many(
        batch,
        max_workers=1,
        cache_dir=None,
        prefer_vmec=False,
        use_real=False,
        problem="p1",
        mf_proxy=True,
        mf_threshold=0.2,
        mf_quantile=None,
        mf_max_high=None,
        mf_metric="qs_residual",
    )

    proxy_scores = [float(r.get("proxy_score", float("nan"))) for r in results]
    assert all(ps >= 0.0 for ps in proxy_scores)
    metrics_used = {r.get("proxy_metric") for r in results}
    assert metrics_used == {"qs_residual"}
    for rec in results:
        assert "qs_residual" in rec, "proxy rows must expose the QS residual value"
        assert "qi_residual" in rec, "proxy rows should include all Boozer proxies"
        assert "helical_energy" in rec, "proxy rows should include all Boozer proxies"
        assert "mirror_ratio" in rec, "proxy rows should include all Boozer proxies"
        assert rec.get("source") == "proxy"
        assert rec.get("feasible") is None
        assert "fail_reason" not in rec
    # Expect three results (real for survivors is not used here since use_real=False)
    assert len(results) == 3
    # All rows must carry a phase field when mf_proxy=True on placeholder path
    phases = [r.get("phase") for r in results]
    assert all(p == "proxy" for p in phases)
    # Scores should be finite and ordered by our construction
    scores = [float(eval_score(r, problem=None)) for r in results]
    assert all(math.isfinite(s) for s in scores)


def test_forward_many_rejects_unknown_proxy_metric() -> None:
    batch = [example_boundary() for _ in range(2)]
    with pytest.raises(ValueError):
        forward_many(
            batch,
            mf_proxy=True,
            mf_metric="not_a_metric",
        )


def test_mf_proxy_metric_options_cover_qs_proxies() -> None:
    available = {m.strip().lower() for m in MF_PROXY_METRICS}
    expected = {
        "score",
        "placeholder_metric",
        "qs_residual",
        "qi_residual",
        "helical_energy",
        "mirror_ratio",
    }
    assert expected.issubset(available)
