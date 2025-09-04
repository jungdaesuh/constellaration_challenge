from __future__ import annotations

import pytest

from constelx.eval.boundary_param import sample_random, validate


def test_sample_random_is_deterministic():
    b1 = sample_random(nfp=3, seed=42)
    b2 = sample_random(nfp=3, seed=42)
    assert b1 == b2


def test_validate_rejects_bad_inputs():
    b = sample_random(nfp=3, seed=0)
    # Good boundary validates
    validate(b)
    # Break coefficient bounds
    b_bad = dict(b)
    b_bad["r_cos"] = [row[:] for row in b["r_cos"]]
    b_bad["r_cos"][0][0] = 99.0
    with pytest.raises(ValueError):
        validate(b_bad)
    # Missing key
    b_missing = dict(b)
    b_missing.pop("z_sin")
    with pytest.raises(ValueError):
        validate(b_missing)
    # Shape mismatch
    b_shape = dict(b)
    b_shape["z_cos"] = [[0.0]]
    with pytest.raises(ValueError):
        validate(b_shape)
