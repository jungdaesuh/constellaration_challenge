from __future__ import annotations

"""Constrained Trust-Region Bayesian Optimization (FuRBO-style) baselines.

This module implements a lightweight trust-region wrapper around constrained
qNEI for single-objective problems (P1/P2) and a scalarized fallback for P3.

Notes
- We model one objective (aggregate score) and one or more constraints derived
  from evaluator metrics. When detailed constraints are not available we fall
  back to a single feasibility indicator.
- The parameterization matches other baselines: a 2D vector mapping to the
  m=1 helical coefficients r_cos[1][5], z_sin[1][5] in [-0.2, 0.2].
"""

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from ..eval import forward as eval_forward
from ..eval import score as eval_score
from ..eval.boundary_param import validate as validate_boundary
from ..physics.constel_api import example_boundary
from .baselines import BaselineConfig


@dataclass(frozen=True)
class FurboConfig(BaselineConfig):
    tr_init: float = 0.2
    tr_min: float = 0.02
    tr_max: float = 0.5
    tr_gamma_inc: float = 1.6
    tr_gamma_dec: float = 0.5
    batch: int = 1  # number of candidates per BO step


def _make_boundary(x: NDArray[np.float64], nfp: int) -> dict[str, Any]:
    b = example_boundary()
    b["n_field_periods"] = int(nfp)
    # Clamp and map to helical pair with fixed sign convention
    r = float(np.clip(-abs(float(x[0])), -0.2, 0.2))
    z = float(np.clip(+abs(float(x[1])), -0.2, 0.2))
    b["r_cos"][1][5] = r
    b["z_sin"][1][5] = z
    validate_boundary(b)
    return b


def _evaluate(x: NDArray[np.float64], cfg: FurboConfig) -> Tuple[float, NDArray[np.float64]]:
    """Return (objective, constraints_vector). Lower objective is better.

    Constraints vector uses c_tilde<=0 as feasible. Falls back to a single
    feasibility indicator when detailed constraints are unavailable.
    """
    from ..problems.constraints import constraints_from_metrics

    b = _make_boundary(x, cfg.nfp)
    metrics = eval_forward(
        b,
        prefer_vmec=cfg.use_physics,
        use_real=cfg.use_physics,
        cache_dir=cfg.cache_dir,
        problem=cfg.problem,
        vmec_level=cfg.vmec_level,
        vmec_hot_restart=cfg.vmec_hot_restart,
        vmec_restart_key=cfg.vmec_restart_key,
    )
    obj = float(eval_score(metrics, problem=cfg.problem if cfg.use_physics else None))
    c_vec = np.asarray(constraints_from_metrics(cfg.problem, metrics), dtype=float)
    return obj, c_vec


def _bounds() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    lo = np.asarray([-0.2, -0.2], dtype=np.float64)
    hi = np.asarray([+0.2, +0.2], dtype=np.float64)
    return lo, hi


def _truncate_to_box(
    x: NDArray[np.float64], lo: NDArray[np.float64], hi: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.minimum(hi, np.maximum(lo, x))


def run_furbo(cfg: FurboConfig) -> Tuple[NDArray[np.float64], float]:
    """Run constrained TR-BO over the 2D helical coefficients.

    Returns (x_best, best_score).
    """
    # Lazy import BoTorch/torch to keep optional dependency
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
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "BoTorch is required for the FuRBO baseline; install 'constelx[bo]'"
        ) from exc

    torch.manual_seed(int(cfg.seed))
    device = torch.device("cpu")
    dtype = torch.double

    lower_global, upper_global = _bounds()
    # Initialize center near a modest amplitude
    x_center = np.asarray([0.05, 0.05], dtype=np.float64)
    tr_radius = float(max(cfg.tr_min, min(cfg.tr_init, cfg.tr_max)))

    train_x_list: list[NDArray[np.float64]] = []
    objs: list[float] = []
    feas: list[bool] = []

    best_x = x_center.copy()
    best_obj = float("inf")
    best_feasible = False

    # BoTorch version compatibility
    try:
        sampler = SobolQMCNormalSampler(num_samples=256)
    except TypeError:  # older/newer API variant
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size((256,)))

    def _acquire_in_box(lo: NDArray[np.float64], hi: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(train_x_list) < 4:
            # Bootstrap with Sobol
            d = lo.shape[0]
            engine = torch.quasirandom.SobolEngine(
                dimension=d, scramble=True, seed=int(cfg.seed) + len(train_x_list)
            )
            pts = engine.draw(max(cfg.batch, 1)).to(dtype=dtype, device=device)
            cand = torch.from_numpy(lo).to(dtype=dtype, device=device) + (
                torch.from_numpy(hi - lo).to(dtype=dtype, device=device) * pts
            )
            return cand.detach().cpu().numpy()

        X = torch.as_tensor(np.asarray(train_x_list), dtype=dtype, device=device)
        y_obj = torch.as_tensor(np.asarray(objs)[:, None], dtype=dtype, device=device)
        # Construct a single feasibility pseudo-constraint: <=0 feasible
        # Using observed c_tilde min across vector if multiple were present in eval
        # Here we collapse to one dimension using min across the vector constraint.
        penalties = []
        for i in range(len(train_x_list)):
            # negative => feasible, positive => violation
            penalties.append(0.0 if feas[i] else 1.0)
        y_pen = torch.as_tensor(np.asarray(penalties)[:, None], dtype=dtype, device=device)

        with botorch_settings.validate_input_scaling(False):
            obj_model = SingleTaskGP(
                X,
                y_obj,
                input_transform=Normalize(d=X.shape[1]),
                outcome_transform=Standardize(m=1),
            )
            con_model = SingleTaskGP(
                X,
                y_pen,
                input_transform=Normalize(d=X.shape[1]),
            )
        model = ModelListGP(obj_model, con_model)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        def objective(samples: torch.Tensor) -> torch.Tensor:
            return -samples[..., 0]

        def constraint(samples: torch.Tensor) -> torch.Tensor:
            # <= 0 is feasible; > 0 violates
            return samples[..., 1]

        constrained_obj = ConstrainedMCObjective(
            objective=objective,
            constraints=[constraint],
            infeasible_cost=-1e6,
        )

        lo_t = torch.as_tensor(lo, dtype=dtype, device=device)
        hi_t = torch.as_tensor(hi, dtype=dtype, device=device)
        bounds = torch.stack([lo_t, hi_t])

        candidate, _ = optimize_acqf(
            qNoisyExpectedImprovement(
                model=model, X_baseline=X, sampler=sampler, objective=constrained_obj
            ),
            bounds=bounds,
            q=max(cfg.batch, 1),
            num_restarts=8,
            raw_samples=64,
        )
        return candidate.detach().cpu().numpy()

    budget = int(cfg.budget)
    it = 0
    while len(train_x_list) < budget:
        # Current TR box
        lo_tr = _truncate_to_box(x_center - tr_radius, lower_global, upper_global)
        hi_tr = _truncate_to_box(x_center + tr_radius, lower_global, upper_global)
        # Acquire q candidates in the current TR
        xq = _acquire_in_box(lo_tr, hi_tr)
        if xq.ndim == 1:
            xq = xq[None, :]
        success = False
        for row in xq:
            if len(train_x_list) >= budget:
                break
            x = _truncate_to_box(row.astype(np.float64), lo_tr, hi_tr)
            obj, c_vec = _evaluate(x, FurboConfig(**vars(cfg)))
            feat = bool(np.all(c_vec <= 0.0))
            train_x_list.append(x)
            objs.append(float(obj))
            feas.append(feat)
            # Track best feasible
            if feat and (not best_feasible or obj < best_obj):
                best_feasible = True
                best_obj = float(obj)
                best_x = x.copy()
                x_center = best_x.copy()
                success = True
        # TR update
        if success:
            tr_radius = min(cfg.tr_max, tr_radius * float(cfg.tr_gamma_inc))
        else:
            tr_radius = max(cfg.tr_min, tr_radius * float(cfg.tr_gamma_dec))
        it += 1

    return np.asarray(best_x, dtype=float), float(best_obj)


__all__ = ["FurboConfig", "run_furbo"]
