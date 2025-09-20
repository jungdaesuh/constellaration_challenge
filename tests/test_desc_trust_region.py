from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from constelx.optim.desc_trust_region import (
    DescResolution,
    DescTrustRegionConfig,
    run_desc_trust_region,
)


class DummySurface:
    def __init__(
        self,
        R_lmn: np.ndarray,
        Z_lmn: np.ndarray,
        modes_R: np.ndarray,
        modes_Z: np.ndarray,
        NFP: int = 3,
        sym: bool = True,
    ) -> None:
        self.R_lmn = np.asarray(R_lmn, dtype=float)
        self.Z_lmn = np.asarray(Z_lmn, dtype=float)
        modes_R = np.asarray(modes_R, dtype=int)
        modes_Z = np.asarray(modes_Z, dtype=int)
        if modes_R.ndim == 2 and modes_R.shape[1] == 2:
            modes_R = np.column_stack([np.zeros(len(modes_R), dtype=int), modes_R])
        if modes_Z.ndim == 2 and modes_Z.shape[1] == 2:
            modes_Z = np.column_stack([np.zeros(len(modes_Z), dtype=int), modes_Z])
        self.R_basis = SimpleNamespace(modes=modes_R)
        self.Z_basis = SimpleNamespace(modes=modes_Z)
        self.NFP = int(NFP)
        self.sym = bool(sym)

    def copy(self) -> "DummySurface":
        return DummySurface(
            self.R_lmn.copy(),
            self.Z_lmn.copy(),
            self.R_basis.modes.copy(),
            self.Z_basis.modes.copy(),
            NFP=self.NFP,
            sym=self.sym,
        )


class DummyEquilibrium:
    def __init__(
        self,
        surface: DummySurface,
        Psi: float = 1.0,
        NFP: int = 3,
        L: int = 0,
        M: int = 8,
        N: int = 8,
    ) -> None:
        self.surface = surface
        self.Psi = float(Psi)
        self.NFP = int(NFP)
        self.L = int(L)
        self.M = int(M)
        self.N = int(N)

    def change_resolution(
        self, L: int | None = None, M: int | None = None, N: int | None = None
    ) -> None:
        if L is not None:
            self.L = int(L)
        if M is not None:
            self.M = int(M)
        if N is not None:
            self.N = int(N)


class DummyObjective:
    def __init__(self, eq: DummyEquilibrium) -> None:
        self.eq = eq


class DummyObjectiveFunction:
    def __init__(self, objective: DummyObjective) -> None:
        self.objective = objective


class DummyForceBalance:
    def __init__(self, eq: DummyEquilibrium) -> None:
        self.eq = eq


class DummyOptimizer:
    calls: list[tuple[int, int, int, int, float, float]] = []

    def __init__(self, method: str) -> None:
        self.method = method

    def optimize(
        self,
        eq: DummyEquilibrium,
        objective: DummyObjectiveFunction,
        constraints: tuple[DummyForceBalance, ...] = (),
        verbose: int = 0,
        maxiter: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[DummyEquilibrium], SimpleNamespace]:
        opts = options or {}
        DummyOptimizer.calls.append(
            (
                eq.L,
                eq.M,
                eq.N,
                int(maxiter) if maxiter is not None else 0,
                float(opts.get("initial_trust_radius", 0.0)),
                float(opts.get("max_trust_radius", 0.0)),
            )
        )
        eq.surface.R_lmn = np.asarray(eq.surface.R_lmn, dtype=float) - 0.01
        eq.surface.Z_lmn = np.asarray(eq.surface.Z_lmn, dtype=float) - 0.01
        return [eq], SimpleNamespace(success=True)


class DummyModules(SimpleNamespace):
    pass


@pytest.fixture(name="dummy_modules")
def _dummy_modules_fixture(monkeypatch: pytest.MonkeyPatch) -> DummyModules:
    from constelx.optim import desc_trust_region as mod

    DummyOptimizer.calls = []

    modules = DummyModules(
        Equilibrium=DummyEquilibrium,
        FourierSurface=DummySurface,
        ObjectiveFunction=DummyObjectiveFunction,
        QuasisymmetryTwoTerm=DummyObjective,
        ForceBalance=DummyForceBalance,
        Optimizer=DummyOptimizer,
    )

    monkeypatch.setattr(mod, "_load_desc", lambda: modules)
    monkeypatch.setattr(mod, "eval_forward", lambda *args, **kwargs: {"placeholder_metric": 0.0})
    monkeypatch.setattr(mod, "eval_score", lambda metrics, problem=None: 0.123)
    return modules


def test_desc_baseline_requires_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    from constelx.optim import desc_trust_region as mod

    monkeypatch.setattr(mod, "_load_desc", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    cfg = DescTrustRegionConfig()
    with pytest.raises(RuntimeError, match="missing"):
        run_desc_trust_region(cfg)


def test_desc_ladder_runs_stages(
    monkeypatch: pytest.MonkeyPatch, dummy_modules: DummyModules
) -> None:
    ladder = (
        DescResolution(L=0, M=4, N=4, maxiter=3, initial_radius=0.2, max_radius=0.4),
        DescResolution(L=0, M=6, N=6, maxiter=5, initial_radius=0.1, max_radius=0.3),
    )
    cfg = DescTrustRegionConfig(ladder=ladder)

    x, score = run_desc_trust_region(cfg)

    assert DummyOptimizer.calls == [
        (0, 4, 4, 3, 0.2, 0.4),
        (0, 6, 6, 5, 0.1, 0.3),
    ]
    # Heuristic update subtracts 0.01 per stage from the helical coefficients
    assert np.allclose(x, np.asarray([-0.07, 0.03]))
    assert score == pytest.approx(0.123)
