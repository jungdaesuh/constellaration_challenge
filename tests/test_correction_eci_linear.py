from __future__ import annotations

import math

from constelx.agents.corrections.eci_linear import (
    EciLinearSpec,
    LinearConstraint,
    Variable,
    make_hook,
)
from constelx.physics.constel_api import example_boundary


def test_eci_linear_projection_enforces_constraint_min_change() -> None:
    b = example_boundary()
    # Variables: r_cos[1][5] and z_sin[1][5]
    v1 = Variable("r_cos", 1, 5)
    v2 = Variable("z_sin", 1, 5)
    # Constraint: v1 + v2 = 0 (sum zero)
    constraint = LinearConstraint(coeffs=[(v1, 1.0), (v2, 1.0)], rhs=0.0)
    spec = EciLinearSpec(variables=[v1, v2], constraints=[constraint])
    hook = make_hook(spec)

    b2 = hook(b)
    x1 = b2["r_cos"][1][5]
    x2 = b2["z_sin"][1][5]
    assert math.isclose(x1 + x2, 0.0, abs_tol=1e-9)
    # Validate minimal change: mean stays same
    x1_0 = b["r_cos"][1][5]
    x2_0 = b["z_sin"][1][5]
    assert math.isclose((x1 + x2) / 2.0, 0.0, abs_tol=1e-9)
    # And symmetry: moved by equal and opposite amounts toward midpoint
    assert math.isclose(x1 - x1_0, -(x2 - x2_0), rel_tol=1e-7, abs_tol=1e-9)
