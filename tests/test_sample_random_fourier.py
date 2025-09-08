from __future__ import annotations

from constelx.eval.boundary_param import sample_random, validate


def test_sample_random_fourier_opt_in_validates_and_deterministic() -> None:
    b1 = sample_random(nfp=3, seed=123, use_fourier=True)
    b2 = sample_random(nfp=3, seed=123, use_fourier=True)
    assert b1 == b2
    validate(b1)
