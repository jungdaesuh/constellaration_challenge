from __future__ import annotations

import math

import numpy as np

from constelx.agents.corrections.eci_linear import Variable as LinVar
from constelx.agents.corrections.pcfm import (
    NormEq,
    PcfmSpec,
    Term,
    make_hook,
)
from constelx.physics.constel_api import example_boundary


def test_pcfm_norm_eq_projects_to_circle_min_change() -> None:
    b = example_boundary()
    # Variables: r_cos[1][5] and z_sin[1][5]
    v1 = LinVar("r_cos", 1, 5)
    v2 = LinVar("z_sin", 1, 5)

    # Build a norm equality: w1 x1^2 + w2 x2^2 = r^2
    radius = 0.08
    con = NormEq(terms=[Term("r_cos", 1, 5, 1.0), Term("z_sin", 1, 5, 1.0)], radius=radius)
    spec = PcfmSpec(variables=[v1, v2], constraints=[con], gn_iters=3)
    hook = make_hook(spec)

    # Perturb away from the circle
    b["r_cos"][1][5] = float(-0.12)
    b["z_sin"][1][5] = float(0.02)
    b2 = hook(b)
    x1 = float(b2["r_cos"][1][5])
    x2 = float(b2["z_sin"][1][5])
    # Residual should be close to zero
    r = (x1 * x1 + x2 * x2) - (radius * radius)
    assert abs(r) < 1e-6
    # Update direction should move toward radial projection (angle preserved)
    theta0 = math.atan2(0.02, -0.12)
    theta = math.atan2(x2, x1)
    assert abs((theta - theta0 + math.pi) % (2 * math.pi) - math.pi) < 1e-3

