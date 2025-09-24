from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, cast

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


def _objective(x: NDArray[np.float64], cfg: BaselineConfig) -> float:
    score, _ = _score_and_penalty(x, cfg)
    return score


def run_trust_constr(cfg: BaselineConfig) -> Tuple[NDArray[np.float64], float]:
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


def run_alm(cfg: BaselineConfig) -> Tuple[NDArray[np.float64], float]:
    """Simple augmented-Lagrangian-like penalty loop.

    This dev-friendly path uses a feasibility signal when available; otherwise it
    reduces to plain trust-constr on the aggregated objective.
    """
    rho = 10.0
    x = np.asarray([0.05, 0.05], dtype=float)
    bounds = Bounds([-0.2, -0.2], [0.2, 0.2], keep_feasible=False)

    def penalized_obj(xv: NDArray[np.float64]) -> float:
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


def run_ngopt(cfg: BaselineConfig) -> Tuple[NDArray[np.float64], float]:
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

        def augmented_loss(arr: NDArray[np.float64]) -> float:
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


__all__ = [
    "BaselineConfig",
    "run_trust_constr",
    "run_alm",
    "run_ngopt",
    "run_botorch_qnei",
]


def run_botorch_qnei(cfg: BaselineConfig) -> Tuple[NDArray[np.float64], float]:
    """BoTorch qNEI baseline with feasibility-aware acquisition.

    The search space matches the other baselines (two helical coefficients bounded
    in ``[-0.2, 0.2]``) and uses the shared evaluator helpers for score + penalty.
    """

    try:
        import torch
        from botorch import settings as botorch_settings
        from botorch.acquisition import qNoisyExpectedImprovement
        from botorch.acquisition.objective import ConstrainedMCObjective
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling import SobolQMCNormalSampler
        from gpytorch.mlls import SumMarginalLogLikelihood

        try:
            from botorch.acquisition import (
                qLogNoisyExpectedImprovement as _qLogNoisyExpectedImprovement,
            )

            qLogNoisyExpectedImprovement = _qLogNoisyExpectedImprovement
        except ImportError:  # pragma: no cover - older BoTorch fallback
            qLogNoisyExpectedImprovement = None
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "BoTorch is required for the qNEI baseline; install 'constelx[bo]'"
        ) from exc

    torch.manual_seed(int(cfg.seed))
    device = torch.device("cpu")
    dtype = torch.double

    bounds = torch.tensor([[-0.2, -0.2], [0.2, 0.2]], dtype=dtype, device=device)
    dim = bounds.shape[1]

    def evaluate_candidate(x_arr: NDArray[np.float64]) -> tuple[float, float]:
        score, penalty = _score_and_penalty(x_arr, cfg)
        if not np.isfinite(score):
            score = float(1e6)
        if not np.isfinite(penalty):
            penalty = float(1e2)
        return float(score), float(penalty)

    def _sobol_draw(n: int) -> "torch.Tensor":
        # Local wrapper to keep mypy strict: treat SobolEngine as Any
        engine_cls = cast(Any, torch.quasirandom.SobolEngine)
        engine = engine_cls(dimension=dim, scramble=True, seed=int(cfg.seed))
        return cast("torch.Tensor", engine.draw(n).to(dtype=dtype, device=device))

    def sample_sobol(num: int) -> list[NDArray[np.float64]]:
        if num <= 0:
            return []
        sobol_raw = _sobol_draw(num)
        sobol_scaled = bounds[0] + (bounds[1] - bounds[0]) * sobol_raw
        return [row.cpu().numpy().astype(np.float64, copy=True) for row in sobol_scaled]

    initial_points: list[NDArray[np.float64]] = [np.asarray([0.05, 0.05], dtype=np.float64)]
    remaining = max(0, min(max(4, dim + 1), int(cfg.budget)) - len(initial_points))
    initial_points.extend(sample_sobol(remaining))
    if not initial_points:
        # Budget was zero; nothing to optimize.
        return np.zeros(2, dtype=float), float("inf")

    train_x_list: list[NDArray[np.float64]] = []
    obj_vals: list[float] = []
    pen_vals: list[float] = []

    best_x = np.asarray(initial_points[0], dtype=float)
    best_score = float("inf")
    best_penalty = float("inf")

    def update_best(x_val: NDArray[np.float64], score: float, penalty: float) -> None:
        nonlocal best_x, best_score, best_penalty
        feasible = penalty <= 0.0
        if feasible:
            if best_penalty > 0.0 or score < best_score:
                best_x = np.asarray(x_val, dtype=float)
                best_score = float(score)
                best_penalty = float(penalty)
        elif best_penalty > 0.0 and penalty < best_penalty:
            best_x = np.asarray(x_val, dtype=float)
            best_score = float(score)
            best_penalty = float(penalty)

    for pt in initial_points:
        score, penalty = evaluate_candidate(pt)
        train_x_list.append(np.asarray(pt, dtype=np.float64))
        obj_vals.append(float(score))
        pen_vals.append(float(penalty))
        update_best(pt, score, penalty)
        if len(train_x_list) >= cfg.budget:
            return np.asarray(best_x, dtype=float), float(best_score)

    # Persistent Sobol engine for fallback/random proposals to preserve state
    _sobol_engine_cls = cast(Any, torch.quasirandom.SobolEngine)
    _sobol_fallback_engine = _sobol_engine_cls(
        dimension=dim, scramble=True, seed=int(cfg.seed) + 7919
    )

    def next_random() -> NDArray[np.float64]:
        random_draw = cast(
            "torch.Tensor",
            _sobol_fallback_engine.draw(1).to(dtype=dtype, device=device)[0],
        )
        candidate = bounds[0] + (bounds[1] - bounds[0]) * random_draw
        return candidate.cpu().numpy().astype(np.float64, copy=True)

    try:
        sampler = SobolQMCNormalSampler(num_samples=256)
    except TypeError:
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size((256,)))

    while len(train_x_list) < cfg.budget:
        train_x = torch.as_tensor(np.stack(train_x_list, axis=0), dtype=dtype, device=device)

        obj_arr = np.asarray(obj_vals, dtype=np.float64)
        obj_mean = float(obj_arr.mean())
        obj_std = float(obj_arr.std())
        if not np.isfinite(obj_std) or obj_std < 1e-12:
            obj_std = 1.0
        obj_norm = (obj_arr - obj_mean) / obj_std

        pen_arr = np.asarray(pen_vals, dtype=np.float64)
        pen_mean = float(pen_arr.mean())
        pen_std = float(pen_arr.std())
        if not np.isfinite(pen_std) or pen_std < 1e-12:
            pen_std = 1.0
        pen_norm = (pen_arr - pen_mean) / pen_std

        train_obj = torch.as_tensor(obj_norm[:, None], dtype=dtype, device=device)
        train_con = torch.as_tensor(pen_norm[:, None], dtype=dtype, device=device)

        with botorch_settings.validate_input_scaling(False):
            obj_model = SingleTaskGP(
                train_x,
                train_obj,
                input_transform=Normalize(d=dim),
                outcome_transform=Standardize(m=1),
            )
            con_model = SingleTaskGP(
                train_x,
                train_con,
                input_transform=Normalize(d=dim),
            )
        model = ModelListGP(obj_model, con_model)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        def objective(samples: torch.Tensor) -> torch.Tensor:
            # Map back to the original scale and minimize the aggregate score.
            unscaled = samples[..., 0] * obj_std + obj_mean
            return -unscaled

        def constraint(samples: torch.Tensor) -> torch.Tensor:
            # Penalty <= 0 is feasible; >0 is violation (original scale).
            return samples[..., 1] * pen_std + pen_mean

        constrained_obj = ConstrainedMCObjective(
            objective=objective,
            constraints=[constraint],
            infeasible_cost=-1e6,
        )

        try:
            acq_cls = (
                qLogNoisyExpectedImprovement
                if qLogNoisyExpectedImprovement is not None
                else qNoisyExpectedImprovement
            )
            candidate, _ = optimize_acqf(
                acq_cls(
                    model=model,
                    X_baseline=train_x,
                    sampler=sampler,
                    objective=constrained_obj,
                ),
                bounds=bounds,
                q=1,
                num_restarts=8,
                raw_samples=64,
            )
            candidate_np = candidate.detach().cpu().numpy()[0].astype(np.float64)
        except Exception:
            candidate_np = next_random()

        if any(np.allclose(candidate_np, prev, atol=1e-6) for prev in train_x_list):
            candidate_np = next_random()

        score, penalty = evaluate_candidate(candidate_np)
        train_x_list.append(candidate_np)
        obj_vals.append(float(score))
        pen_vals.append(float(penalty))
        update_best(candidate_np, score, penalty)

    return np.asarray(best_x, dtype=float), float(best_score)
