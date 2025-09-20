from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..eval import forward as eval_forward
from ..eval import score as eval_score
from ..eval.boundary_param import validate as validate_boundary
from .baselines import BaselineConfig


@dataclass(frozen=True)
class DescResolution:
    """Resolution and trust-region settings for a DESC optimization stage."""

    L: int = 0
    M: int = 8
    N: int = 8
    maxiter: int = 25
    initial_radius: float = 1e-3
    max_radius: float = 0.1


_DEFAULT_DESC_LADDER: tuple[DescResolution, ...] = (
    DescResolution(L=0, M=8, N=8, maxiter=20, initial_radius=1e-3, max_radius=0.08),
    DescResolution(L=0, M=12, N=12, maxiter=30, initial_radius=7.5e-4, max_radius=0.05),
)


@dataclass(frozen=True)
class DescTrustRegionConfig(BaselineConfig):
    """Config for the DESC trust-region baseline."""

    ladder: tuple[DescResolution, ...] = field(default_factory=lambda: _DEFAULT_DESC_LADDER)
    method: str = "fmin-auglag"
    prefer_vmec_validation: bool = False


@dataclass(frozen=True)
class _DescModules:
    Equilibrium: Any
    FourierSurface: Any
    ObjectiveFunction: Any
    QuasisymmetryTwoTerm: Any
    ForceBalance: Any
    Optimizer: Any


def _load_desc() -> _DescModules:
    try:
        eq_mod = importlib.import_module("desc.equilibrium")
        geom_mod = importlib.import_module("desc.geometry")
        objectives = importlib.import_module("desc.objectives")
        opt_mod = importlib.import_module("desc.optimize")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "DESC is not installed; install 'constelx[desc]' to enable the trust-region baseline"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for exotic import errors
        raise RuntimeError(f"Failed to import DESC components: {exc}") from exc

    try:
        return _DescModules(
            Equilibrium=getattr(eq_mod, "Equilibrium"),
            FourierSurface=getattr(geom_mod, "FourierRZToroidalSurface"),
            ObjectiveFunction=getattr(objectives, "ObjectiveFunction"),
            QuasisymmetryTwoTerm=getattr(objectives, "QuasisymmetryTwoTerm"),
            ForceBalance=getattr(objectives, "ForceBalance"),
            Optimizer=getattr(opt_mod, "Optimizer"),
        )
    except AttributeError as exc:  # pragma: no cover - only if DESC API changes unexpectedly
        raise RuntimeError(f"DESC API missing expected attribute: {exc}") from exc


def _iter_modes(grid: Sequence[Sequence[Any] | None]) -> Iterable[tuple[int, int, float]]:
    for m, row in enumerate(grid or []):
        if row is None:
            continue
        for n, val in enumerate(row):
            if val is None:
                continue
            try:
                coeff = float(val)
            except (TypeError, ValueError):
                continue
            if coeff == 0.0:
                continue
            yield m, n, coeff


def _boundary_to_surface(boundary: Mapping[str, Any], mods: _DescModules) -> Any:
    r_cos = boundary.get("r_cos") or []
    z_sin = boundary.get("z_sin") or []

    R_lmn: list[float] = []
    modes_R: list[list[int]] = []
    for m, n, coeff in _iter_modes(r_cos):
        modes_R.append([m, n])
        R_lmn.append(coeff)

    if not R_lmn:
        R_lmn = [1.0]
        modes_R = [[0, 0]]

    Z_lmn: list[float] = []
    modes_Z: list[list[int]] = []
    for m, n, coeff in _iter_modes(z_sin):
        modes_Z.append([m, n])
        Z_lmn.append(coeff)

    if not Z_lmn:
        Z_lmn = [0.0]
        modes_Z = [[0, 0]]

    surface = mods.FourierSurface(
        R_lmn=np.asarray(R_lmn, dtype=float),
        Z_lmn=np.asarray(Z_lmn, dtype=float),
        modes_R=np.asarray(modes_R, dtype=int),
        modes_Z=np.asarray(modes_Z, dtype=int),
        NFP=int(boundary.get("n_field_periods", 3)),
        sym=bool(boundary.get("is_stellarator_symmetric", True)),
    )
    return surface


def _update_boundary_from_surface(boundary: dict[str, Any], surface: Any) -> None:
    modes_R = getattr(surface, "R_basis").modes
    coeffs_R = np.asarray(getattr(surface, "R_lmn"), dtype=float)
    r_cos = boundary.get("r_cos")

    for idx, mode in enumerate(modes_R):
        if mode.shape[0] < 3:
            continue
        m = int(mode[-2])
        n = int(mode[-1])
        if m < 0 or n < 0:
            continue
        if (
            isinstance(r_cos, list)
            and m < len(r_cos)
            and isinstance(r_cos[m], list)
            and n < len(r_cos[m])
        ):
            r_cos[m][n] = float(coeffs_R[idx])

    modes_Z = getattr(surface, "Z_basis").modes
    coeffs_Z = np.asarray(getattr(surface, "Z_lmn"), dtype=float)
    z_sin = boundary.get("z_sin")

    for idx, mode in enumerate(modes_Z):
        if mode.shape[0] < 3:
            continue
        m = int(mode[-2])
        n = int(mode[-1])
        if m < 0 or n < 0:
            continue
        if (
            isinstance(z_sin, list)
            and m < len(z_sin)
            and isinstance(z_sin[m], list)
            and n < len(z_sin[m])
        ):
            z_sin[m][n] = float(coeffs_Z[idx])


def _helical_coeffs_from_boundary(boundary: Mapping[str, Any]) -> NDArray[np.float64]:
    r_cos = boundary.get("r_cos") or []
    z_sin = boundary.get("z_sin") or []
    r_val = (
        float(r_cos[1][5])
        if len(r_cos) > 1 and isinstance(r_cos[1], list) and len(r_cos[1]) > 5
        else 0.0
    )
    z_val = (
        float(z_sin[1][5])
        if len(z_sin) > 1 and isinstance(z_sin[1], list) and len(z_sin[1]) > 5
        else 0.0
    )
    return np.asarray([r_val, z_val], dtype=float)


def run_desc_trust_region(
    cfg: DescTrustRegionConfig,
) -> tuple[NDArray[np.float64], float]:
    """Run the gradient trust-region baseline using DESC with a resolution ladder."""

    mods = _load_desc()
    if not cfg.ladder:
        raise ValueError("DescTrustRegionConfig.ladder must contain at least one stage")
    # Prepare initial boundary and optional VMEC validation
    boundary = _make_initial_boundary(cfg)
    if cfg.prefer_vmec_validation:
        try:
            from ..eval import boundary_to_vmec as _boundary_to_vmec

            _boundary_to_vmec(boundary)
        except Exception:
            pass

    surface = _boundary_to_surface(boundary, mods)
    eq = mods.Equilibrium(
        surface=surface,
        Psi=1.0,
        NFP=int(cfg.nfp),
        L=cfg.ladder[0].L,
        M=cfg.ladder[0].M,
        N=cfg.ladder[0].N,
    )

    for stage in cfg.ladder:
        eq.change_resolution(L=stage.L, M=stage.M, N=stage.N)
        objective = mods.ObjectiveFunction(mods.QuasisymmetryTwoTerm(eq=eq))
        constraints = (mods.ForceBalance(eq=eq),)
        optimizer = mods.Optimizer(cfg.method)
        options = {
            "initial_trust_radius": float(stage.initial_radius),
            "max_trust_radius": float(stage.max_radius),
        }
        things, result = optimizer.optimize(
            eq,
            objective=objective,
            constraints=constraints,
            verbose=0,
            maxiter=int(stage.maxiter),
            options=options,
        )
        eq = things[0]
        # In some DESC versions optimize returns a tuple of Optimizable, guard to copy back
        if getattr(eq, "surface", None) is None and hasattr(things[0], "surface"):
            eq = things[0]
        if getattr(result, "success", True) is False:
            break

    optimized_surface = eq.surface if hasattr(eq, "surface") else surface
    updated_boundary = copy.deepcopy(boundary)
    _update_boundary_from_surface(updated_boundary, optimized_surface)
    validate_boundary(updated_boundary)

    metrics = eval_forward(
        updated_boundary,
        prefer_vmec=cfg.use_physics,
        use_real=cfg.use_physics,
        cache_dir=cfg.cache_dir,
        problem=cfg.problem,
    )
    best_score = float(eval_score(metrics, problem=cfg.problem if cfg.use_physics else None))
    return _helical_coeffs_from_boundary(updated_boundary), best_score


def _make_initial_boundary(cfg: DescTrustRegionConfig) -> dict[str, Any]:
    from ..physics.constel_api import example_boundary

    boundary = example_boundary()
    boundary["n_field_periods"] = int(cfg.nfp)
    return boundary


__all__ = [
    "DescResolution",
    "DescTrustRegionConfig",
    "run_desc_trust_region",
]
