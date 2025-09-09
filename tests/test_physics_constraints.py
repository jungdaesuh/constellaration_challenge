import copy

import pytest

from constelx.physics.constraints import aspect_ratio, curvature_smoothness


def _base_boundary() -> dict:
    b = {
        "r_cos": [[0.0 for _ in range(6)] for _ in range(2)],
        "r_sin": None,
        "z_cos": None,
        "z_sin": [[0.0 for _ in range(6)] for _ in range(2)],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    for j in range(6):
        b["r_cos"][1][j] = 0.1
        b["z_sin"][1][j] = 0.2
    b["r_cos"][0][4] = 1.0
    return b


def test_aspect_ratio_simple() -> None:
    b = _base_boundary()
    ar = aspect_ratio(b)
    assert ar == pytest.approx(4.4721, rel=1e-3)


def test_curvature_smoothness_variation() -> None:
    b_const = _base_boundary()
    smooth0 = curvature_smoothness(b_const)
    assert smooth0 == pytest.approx(0.0)

    b_var = copy.deepcopy(b_const)
    b_var["r_cos"][1][1] = 0.05
    b_var["z_sin"][1][2] = 0.1
    assert curvature_smoothness(b_var) > smooth0


def test_aspect_ratio_missing_coeffs() -> None:
    b = {"r_cos": [[0.0]], "z_sin": [[0.0]]}
    assert aspect_ratio(b) == 0.0


def test_curvature_smoothness_short_series() -> None:
    b = {
        "r_cos": [[0.0 for _ in range(2)] for _ in range(2)],
        "z_sin": [[0.0 for _ in range(2)] for _ in range(2)],
    }
    assert curvature_smoothness(b) == 0.0
