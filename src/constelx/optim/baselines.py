from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
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
    cache_dir: Path = Path(".cache/eval")
    vmec_level: str | None = None
    vmec_hot_restart: bool | None = None
    vmec_restart_key: str | None = None


def _make_boundary(x: Sequence[float] | NDArray[np.floating[Any]], nfp: int) -> dict[str, Any]:
    b = example_boundary()
    b["n_field_periods"] = int(nfp)
    # Map 2D params to two helical coefficients; clamp softly
    r = float(np.clip(-abs(x[0]), -0.2, 0.2))
    z = float(np.clip(abs(x[1]), -0.2, 0.2))
    b["r_cos"][1][5] = r
    b["z_sin"][1][5] = z
    validate_boundary(b)
    return b


def _constraint_penalty(metrics: Mapping[str, Any]) -> float:
    pen = 0.0
    feas_val = metrics.get("feasibility")
    if isinstance(feas_val, (int, float)):
        pen += float(max(0.0, feas_val))
    feas_flag = metrics.get("feasible")
    if isinstance(feas_flag, bool) and not feas_flag:
        pen += 100.0
    return pen


def _score_and_penalty(
    x: Sequence[float] | NDArray[np.floating[Any]], cfg: BaselineConfig
) -> tuple[float, float]:
    boundary = _make_boundary(x, cfg.nfp)
    metrics = eval_forward(
        boundary,
        prefer_vmec=cfg.use_physics,
        use_real=cfg.use_physics,
        cache_dir=cfg.cache_dir,
        problem=cfg.problem,
        vmec_level=cfg.vmec_level,
        vmec_hot_restart=cfg.vmec_hot_restart,
        vmec_restart_key=cfg.vmec_restart_key,
    )
    score = float(eval_score(metrics, problem=cfg.problem if cfg.use_physics else None))
    penalty = _constraint_penalty(metrics)
    return score, penalty


def _objective(x: np.ndarray, cfg: BaselineConfig) -> float:
    score, _ = _score_and_penalty(x, cfg)
    return score


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
        base, pen = _score_and_penalty(xv, cfg)
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


def run_ngopt(cfg: BaselineConfig) -> Tuple[np.ndarray, float]:
    """Nevergrad NGOpt baseline with a simple augmented-Lagrangian outer loop."""

    try:
        import nevergrad as ng
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Nevergrad is required for the NGOpt baseline; install 'constelx[evolution]'"
        ) from exc

    lower = [-0.2, -0.2]
    upper = [0.2, 0.2]
    x = np.asarray([0.05, 0.05], dtype=float)
    rho = 10.0
    lam = 0.0

    outer = max(1, int(np.ceil(cfg.budget / 20)))
    inner_budget = max(1, cfg.budget // outer)

    best_score = float("inf")
    best_x = np.asarray(x, dtype=float)

    for outer_idx in range(outer):
        parametrization = ng.p.Array(init=np.asarray(x, dtype=float), lower=lower, upper=upper)
        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=inner_budget)
        optimizer.parametrization.random_state.seed(cfg.seed + outer_idx)

        def augmented_loss(arr: np.ndarray) -> float:
            values = np.asarray(arr, dtype=float)
            base, penalty = _score_and_penalty(values, cfg)
            return base + lam * penalty + 0.5 * rho * penalty * penalty

        recommendation = optimizer.minimize(augmented_loss)
        # Update iterate and bookkeeping
        x = np.asarray(recommendation.value, dtype=float)
        score, penalty = _score_and_penalty(x, cfg)
        if score < best_score:
            best_score = score
            best_x = np.asarray(x, dtype=float)
        lam = max(0.0, lam + rho * penalty)
        rho *= 2.0

    return best_x, best_score


__all__ = ["BaselineConfig", "run_trust_constr", "run_alm", "run_ngopt"]
