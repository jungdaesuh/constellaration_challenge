from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, minimize

from ..eval import forward as eval_forward
from ..eval import score as eval_score
from ..eval.boundary_param import validate as validate_boundary
from ..physics.constel_api import example_boundary


@dataclass(frozen=True)
class BaselineConfig:
    nfp: int = 3
    budget: int = 50
    seed: int = 0
    use_physics: bool = False
    problem: str = "p1"


def _make_boundary(x: Sequence[float], nfp: int) -> dict[str, Any]:
    b = example_boundary()
    b["n_field_periods"] = int(nfp)
    # Map 2D params to two helical coefficients; clamp softly
    r = float(np.clip(-abs(x[0]), -0.2, 0.2))
    z = float(np.clip(abs(x[1]), -0.2, 0.2))
    b["r_cos"][1][5] = r
    b["z_sin"][1][5] = z
    validate_boundary(b)
    return b


def _objective(x: np.ndarray, cfg: BaselineConfig) -> float:
    b = _make_boundary(x, cfg.nfp)
    m = eval_forward(
        b,
        prefer_vmec=cfg.use_physics,
        use_real=cfg.use_physics,
        problem=cfg.problem,
    )
    # Prefer evaluator-provided score when present and problem is known via eval_score
    return float(eval_score(m, problem=cfg.problem if cfg.use_physics else None))


def run_trust_constr(cfg: BaselineConfig) -> Tuple[np.ndarray, float]:
    """Trust-constr baseline over two helical coefficients.

    Returns (x_best, best_score).
    """
    x0 = np.asarray([0.05, 0.05], dtype=float)
    bounds = Bounds([-0.2, -0.2], [0.2, 0.2], keep_feasible=False)
    res = minimize(
        lambda x: _objective(x, cfg),
        x0,
        method="trust-constr",
        bounds=bounds,
        options={"maxiter": int(cfg.budget), "gtol": 1e-8, "xtol": 1e-8, "verbose": 0},
    )
    return np.asarray(res.x, dtype=float), float(res.fun)


def run_alm(cfg: BaselineConfig) -> Tuple[np.ndarray, float]:
    """Simple augmented-Lagrangian-like penalty loop.

    This placeholder uses a feasibility signal when available; otherwise it
    reduces to plain trust-constr on the aggregated objective.
    """
    rho = 10.0
    x = np.asarray([0.05, 0.05], dtype=float)
    bounds = Bounds([-0.2, -0.2], [0.2, 0.2], keep_feasible=False)

    def penalized_obj(xv: np.ndarray) -> float:
        b = _make_boundary(xv, cfg.nfp)
        m = eval_forward(
            b,
            prefer_vmec=cfg.use_physics,
            use_real=cfg.use_physics,
            problem=cfg.problem,
        )
        base = float(eval_score(m, problem=cfg.problem if cfg.use_physics else None))
        pen = 0.0
        # If evaluator reports feasibility metric, add it; if marks infeasible, add a large penalty
        feas_val = m.get("feasibility")
        if isinstance(feas_val, (int, float)):
            pen += float(max(0.0, feas_val))
        feas_flag = m.get("feasible")
        if isinstance(feas_flag, bool) and not feas_flag:
            pen += 100.0
        return base + rho * pen

    # A few outer updates of rho; each inner is trust-constr for a handful of steps
    outer = max(1, int(np.ceil(cfg.budget / 10)))
    inner_iter = max(1, cfg.budget // outer)
    for _ in range(outer):
        res = minimize(
            penalized_obj,
            x,
            method="trust-constr",
            bounds=bounds,
            options={"maxiter": int(inner_iter), "gtol": 1e-8, "xtol": 1e-8, "verbose": 0},
        )
        x = np.asarray(res.x, dtype=float)
        rho *= 2.0
    # Final score reported is the unpenalized objective
    final = float(_objective(x, cfg))
    return x, final


__all__ = ["BaselineConfig", "run_trust_constr", "run_alm"]

