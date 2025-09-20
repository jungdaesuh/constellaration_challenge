from __future__ import annotations

import math

import pytest

from constelx.eval.boozer import compute_boozer_proxies
from constelx.eval.boundary_fourier import BoundaryFourier
from constelx.physics.constel_api import evaluate_boundary, example_boundary


def test_boozer_proxies_bounded() -> None:
    proxies = compute_boozer_proxies(example_boundary())
    assert math.isfinite(proxies.b_mean)
    assert 0.0 <= proxies.b_std_fraction <= 1.0
    assert 0.0 <= proxies.mirror_ratio <= 1.0
    assert 0.0 <= proxies.qs_residual <= 1.0
    assert 0.0 <= proxies.qi_residual <= 1.0
    assert proxies.qs_quality == pytest.approx(1.0 - proxies.qs_residual)
    assert proxies.qi_quality == pytest.approx(1.0 - proxies.qi_residual)


def test_boozer_proxies_prefix_dict() -> None:
    proxies = compute_boozer_proxies(example_boundary())
    prefixed = proxies.to_dict(prefix="proxy_")
    assert all(key.startswith("proxy_") for key in prefixed)
    assert prefixed["proxy_qs_quality"] == proxies.qs_quality


def test_boozer_residuals_respond_to_deformation() -> None:
    base = example_boundary()
    base_proxies = compute_boozer_proxies(base)

    bf = BoundaryFourier.from_surface_dict(base)
    i, j = bf.idx(1, 2)
    bf.r_cos[i][j] = 0.4  # introduce additional helical content
    bf.z_sin[i][j] = 0.4
    distorted = bf.to_surface_rz_fourier_dict()
    distorted_proxies = compute_boozer_proxies(distorted)

    assert distorted_proxies.qs_residual > base_proxies.qs_residual
    assert distorted_proxies.qi_residual >= base_proxies.qi_residual


def test_evaluate_boundary_adds_proxy_metrics() -> None:
    metrics = evaluate_boundary(example_boundary(), use_real=False)
    assert "proxy_qs_residual" in metrics
    assert "proxy_qi_quality" in metrics
