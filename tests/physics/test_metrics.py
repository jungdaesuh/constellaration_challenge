from __future__ import annotations

from constelx.physics.constel_api import evaluate_boundary, example_boundary
from constelx.physics.metrics import compute, enrich


def test_metrics_compute_placeholder_metadata() -> None:
    boundary = example_boundary()
    result = compute(boundary, use_real=False, attach_proxies=False)

    assert result["source"] == "placeholder"
    assert result["feasible"] is True
    assert "fail_reason" in result
    assert result["fail_reason"] == ""


def test_metrics_enrich_adds_defaults_and_proxies() -> None:
    boundary = example_boundary()
    base_metrics = evaluate_boundary(dict(boundary), use_real=False)
    # Remove geometry defaults to verify enrichment adds them deterministically.
    base_metrics.pop("nfp", None)
    base_metrics.pop("stellarator_symmetric", None)

    enriched = enrich(base_metrics, boundary)

    # Existing values remain untouched.
    assert enriched["placeholder_metric"] == base_metrics["placeholder_metric"]

    assert enriched["nfp"] == boundary.get("n_field_periods", 0)
    assert enriched["stellarator_symmetric"] is True

    for key in ("qs_residual", "qi_residual", "helical_energy", "mirror_ratio"):
        assert key in enriched
        assert 0.0 <= enriched[key] <= 1.0
