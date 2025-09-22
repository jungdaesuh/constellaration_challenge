from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ProblemSpec:
    """Lightweight problem specification for P1–P3.

    Fields are intentionally minimal and decoupled from the external evaluator
    so ConStelX can document expectations and perform gentle validation without
    adding hard dependencies.
    """

    pid: str
    name: str
    required_metrics: Tuple[str, ...]
    optional_metrics: Tuple[str, ...] = ()
    description: str = ""

    def missing_keys(self, metrics: Mapping[str, object]) -> List[str]:
        present = set(metrics.keys())
        return [k for k in self.required_metrics if k not in present]


def _p1() -> ProblemSpec:
    # Geometric single-objective; external evaluator typically returns
    # at least {objective, score, feasibility}. Keep requirements minimal.
    return ProblemSpec(
        pid="p1",
        name="Geometric shaping",
        required_metrics=("score",),
        optional_metrics=("objective", "feasibility"),
        description="Single-objective geometry task; lower is better.",
    )


def _p2() -> ProblemSpec:
    return ProblemSpec(
        pid="p2",
        name="Simple-to-build QI",
        required_metrics=("score",),
        optional_metrics=("objective", "feasibility"),
        description="QI with coil-simplicity proxy; single-objective.",
    )


def _p3() -> ProblemSpec:
    # Multi-objective; external returns a scalar score and a list of objectives
    # or separate objective_k entries. Require score; advise objectives.
    return ProblemSpec(
        pid="p3",
        name="MHD-stable QI multi-objective",
        required_metrics=("score",),
        optional_metrics=("objectives", "objective_0", "objective_1", "feasibility"),
        description="Multi-objective with Pareto analysis; scalar score present for ranking.",
    )


_REGISTRY: Dict[str, ProblemSpec] = {
    "p1": _p1(),
    "geom": _p1(),
    "geometric": _p1(),
    "geometrical": _p1(),
    "p2": _p2(),
    "simple": _p2(),
    "qi_simple": _p2(),
    "simple_qi": _p2(),
    "p3": _p3(),
    "multi": _p3(),
    "mhd": _p3(),
    "qi_stable": _p3(),
}


def get_spec(problem: str) -> Optional[ProblemSpec]:
    return _REGISTRY.get(problem.lower().strip())


def list_specs() -> Iterable[ProblemSpec]:
    # Return canonical P1, P2, P3 in order
    yield _REGISTRY["p1"]
    yield _REGISTRY["p2"]
    yield _REGISTRY["p3"]


__all__ = ["ProblemSpec", "get_spec", "list_specs"]
