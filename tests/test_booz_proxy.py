from __future__ import annotations

import math

from constelx.physics.booz_proxy import BoozerProxy, compute_proxies
from constelx.physics.constel_api import example_boundary


def test_example_boundary_has_low_residuals() -> None:
    proxy = compute_proxies(example_boundary(), use_real=False)
    assert isinstance(proxy, BoozerProxy)
    assert proxy.qs_residual < 0.2
    assert proxy.qi_residual < 0.4
    assert proxy.helical_energy > 0.0
    assert 0.0 <= proxy.mirror_ratio <= 1.0


def test_helical_energy_increases_with_large_modes() -> None:
    boundary = example_boundary()
    boundary["r_cos"][2][3] = 0.3
    boundary["z_sin"][3][2] = 0.25
    proxy = compute_proxies(boundary, use_real=False)
    assert proxy.qs_residual > 0.0
    assert proxy.qi_residual > 0.04
    assert proxy.helical_energy > 0.02


def test_handles_missing_coefficients_gracefully() -> None:
    proxy = compute_proxies({"n_field_periods": 3}, use_real=False)
    values = proxy.as_dict()
    assert set(values.keys()) == {"qs_residual", "qi_residual", "helical_energy", "mirror_ratio"}
    for value in values.values():
        assert not math.isnan(value)
        assert 0.0 <= value <= 1.0
