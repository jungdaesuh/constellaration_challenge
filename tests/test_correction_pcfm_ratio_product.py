from __future__ import annotations

import math

from constelx.agents.corrections.eci_linear import Variable as LinVar
from constelx.agents.corrections.pcfm import (
    PcfmSpec,
    ProductEq,
    RatioEq,
    Var,
    make_hook,
)
from constelx.physics.constel_api import example_boundary


def test_pcfm_ratio_eq_projects_to_target() -> None:
    b = example_boundary()
    v1 = LinVar("r_cos", 1, 5)
    v2 = LinVar("z_sin", 1, 5)
    # Start away from target ratio
    b["r_cos"][1][5] = float(-0.10)
    b["z_sin"][1][5] = float(0.02)

    con = RatioEq(num=Var("z_sin", 1, 5), den=Var("r_cos", 1, 5), target=-1.5, eps=1e-6)
    spec = PcfmSpec(variables=[v1, v2], constraints=[con], gn_iters=3)
    hook = make_hook(spec)
    b2 = hook(b)
    x = float(b2["r_cos"][1][5])
    y = float(b2["z_sin"][1][5])
    ratio = y / (x + 1e-6)
    assert math.isfinite(ratio)
    # Allow small GN approximation error
    assert abs(ratio - (-1.5)) < 2e-3


def test_pcfm_product_eq_projects_to_target() -> None:
    b = example_boundary()
    v1 = LinVar("r_cos", 1, 5)
    v2 = LinVar("z_sin", 1, 5)
    # Start away from target product
    b["r_cos"][1][5] = float(-0.12)
    b["z_sin"][1][5] = float(0.02)

    target = 0.003
    con = ProductEq(a=Var("r_cos", 1, 5), b=Var("z_sin", 1, 5), target=target)
    spec = PcfmSpec(variables=[v1, v2], constraints=[con], gn_iters=3)
    hook = make_hook(spec)
    b2 = hook(b)
    x = float(b2["r_cos"][1][5])
    y = float(b2["z_sin"][1][5])
    prod = x * y
    assert math.isfinite(prod)
    assert abs(prod - target) < 1e-6
