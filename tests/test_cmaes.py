from __future__ import annotations

import importlib.util
import math

import pytest


def cma_available() -> bool:
    return importlib.util.find_spec("cma") is not None


@pytest.mark.skipif(not cma_available(), reason="cma is not installed")
def test_cmaes_optimizes_sphere() -> None:
    from constelx.optim.cmaes import optimize

    def sphere(x: list[float]) -> float:
        return float(sum(v * v for v in x))

    best_x, hist = optimize(sphere, x0=[0.7, -0.4], bounds=(-1.0, 1.0), budget=20, sigma0=0.3, seed=0)
    assert len(hist) == 20
    # Should get reasonably close to zero
    assert sum(v * v for v in best_x) < 1e-2

