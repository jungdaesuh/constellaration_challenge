from __future__ import annotations

import math

from constelx.agents.corrections.eci_linear import Variable as LinVar
from constelx.agents.corrections.pcfm import (
    ArBand,
    ClearanceMin,
    EdgeIotaEq,
    NormEq,
    PcfmSpec,
    Term,
    Var,
    make_hook,
)
from constelx.physics.constel_api import example_boundary
from constelx.physics.constraints import aspect_ratio


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


def _common_variables() -> tuple[LinVar, LinVar, LinVar]:
    return LinVar("r_cos", 0, 4), LinVar("r_cos", 1, 5), LinVar("z_sin", 1, 5)


def test_pcfm_ar_band_projects_within_band() -> None:
    b = example_boundary()
    v0, v1, v2 = _common_variables()
    con = ArBand(
        major=Var("r_cos", 0, 4),
        minor=(Var("r_cos", 1, 5), Var("z_sin", 1, 5)),
        amin=4.0,
        amax=8.0,
    )
    spec = PcfmSpec(variables=[v0, v1, v2], constraints=[con], gn_iters=5)
    hook = make_hook(spec)

    b2 = hook(b)
    A = aspect_ratio(b2)
    assert 4.0 - 1e-3 <= A <= 8.0 + 1e-3


def test_pcfm_edge_iota_hits_target() -> None:
    b = example_boundary()
    v0, v1, v2 = _common_variables()
    target = 0.1
    con = EdgeIotaEq(
        major=Var("r_cos", 0, 4),
        helical=(Var("r_cos", 1, 5), Var("z_sin", 1, 5)),
        target=target,
    )
    spec = PcfmSpec(variables=[v0, v1, v2], constraints=[con], gn_iters=5)
    hook = make_hook(spec)

    b2 = hook(b)
    r0 = abs(float(b2["r_cos"][0][4]))
    rc = float(b2["r_cos"][1][5])
    zs = float(b2["z_sin"][1][5])
    helical = math.sqrt(rc * rc + zs * zs)
    iota = helical / (r0 + 1e-6)
    assert abs(iota - target) < 5e-3


def test_pcfm_clearance_minimum_respected() -> None:
    b = example_boundary()
    v0, v1, v2 = _common_variables()
    minimum = 0.95
    con = ClearanceMin(
        major=Var("r_cos", 0, 4),
        helical=(Var("r_cos", 1, 5), Var("z_sin", 1, 5)),
        minimum=minimum,
    )
    spec = PcfmSpec(variables=[v0, v1, v2], constraints=[con], gn_iters=5)
    hook = make_hook(spec)

    b2 = hook(b)
    r0 = abs(float(b2["r_cos"][0][4]))
    rc = float(b2["r_cos"][1][5])
    zs = float(b2["z_sin"][1][5])
    helical = math.sqrt(rc * rc + zs * zs)
    clearance = r0 - helical
    assert clearance >= minimum - 1e-3
