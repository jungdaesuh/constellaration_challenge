from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ...eval.boundary_param import validate as validate_boundary
from ...physics.pcfm import project_gauss_newton
from .eci_linear import Variable as LinVariable  # reuse simple variable tuple


@dataclass(frozen=True)
class Term:
    field: str
    i: int
    j: int
    w: float = 1.0


@dataclass(frozen=True)
class NormEq:
    terms: List[Term]
    radius: float
    weight: float = 1.0


Constraint = NormEq  # minimal starter: support norm equality


@dataclass(frozen=True)
class PcfmSpec:
    variables: List[LinVariable]
    constraints: List[Constraint]
    coeff_abs_max: float = 2.0
    gn_iters: int = 2
    damping: float = 1e-6
    tol: float = 1e-8


def _flatten(
    boundary: Mapping[str, Any],
    variables: Sequence[LinVariable],
) -> NDArray[np.floating[Any]]:
    xs: List[float] = []
    for v in variables:
        arr = np.asarray(boundary[v.field])
        xs.append(float(arr[v.i, v.j]))
    return np.asarray(xs, dtype=float)


def _unflatten(
    boundary: Mapping[str, Any], variables: Sequence[LinVariable], x: NDArray[np.floating[Any]]
) -> Dict[str, Any]:
    b = dict(boundary)
    for v, val in zip(variables, x.tolist()):
        arr = np.asarray(b[v.field]).copy()
        # clamp to keep validation safe
        arr[v.i, v.j] = float(max(-10.0, min(10.0, val)))
        b[v.field] = arr.tolist()
    return b


def _build_residual_and_jac(
    variables: Sequence[LinVariable], constraints: Sequence[Constraint]
) -> Tuple[
    Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
]:
    # Map variable to index
    index: Dict[Tuple[str, int, int], int] = {
        (v.field, int(v.i), int(v.j)): k for k, v in enumerate(variables)
    }

    def residual(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        rs: List[float] = []
        for con in constraints:
            # r = sum w_i x_i^2 - r^2
            s = 0.0
            for t in con.terms:
                k = index[(t.field, int(t.i), int(t.j))]
                s += float(t.w) * float(x[k]) * float(x[k])
            rs.append(con.weight * (s - float(con.radius) * float(con.radius)))
        return np.asarray(rs, dtype=float)

    def jacobian(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        m = len(constraints)
        n = len(variables)
        J = np.zeros((m, n), dtype=float)
        for r, con in enumerate(constraints):
            for t in con.terms:
                k = index[(t.field, int(t.i), int(t.j))]
                J[r, k] += con.weight * (2.0 * float(t.w) * float(x[k]))
        return J

    return residual, jacobian


def make_hook(spec: PcfmSpec) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    residual, jac = _build_residual_and_jac(spec.variables, spec.constraints)

    def hook(boundary: Mapping[str, Any]) -> Dict[str, Any]:
        x0 = _flatten(boundary, spec.variables)
        x = project_gauss_newton(
            x0,
            residual,
            jacobian=jac,
            max_iters=spec.gn_iters,
            tol=spec.tol,
            damping=spec.damping,
        )
        b = _unflatten(boundary, spec.variables, x)
        # Clamp only touched fields for safety; then validate (best effort)
        try:
            validate_boundary(b, coeff_abs_max=spec.coeff_abs_max)
        except Exception:
            # If validation fails, revert to original boundary
            return dict(boundary)
        return b

    return hook


def build_spec_from_json(data: Sequence[Dict[str, Any]]) -> PcfmSpec:
    # Collect variables in first-appearance order
    var_index: Dict[Tuple[str, int, int], int] = {}
    variables: List[LinVariable] = []

    def get_var(field: str, i: int, j: int) -> LinVariable:
        key = (field, int(i), int(j))
        if key not in var_index:
            var_index[key] = len(variables)
            variables.append(LinVariable(field=field, i=int(i), j=int(j)))
        return variables[var_index[key]]

    constraints: List[Constraint] = []
    for item in data:
        t = str(item.get("type", "norm_eq")).lower()
        if t != "norm_eq":
            raise ValueError("Only 'norm_eq' is supported in the starter")
        terms_in = item.get("terms", [])
        terms: List[Term] = []
        for td in terms_in:
            field = str(td["field"])
            i = int(td["i"])  # required
            j = int(td["j"])  # required
            w = float(td.get("w", 1.0))
            get_var(field, i, j)  # register order
            terms.append(Term(field=field, i=i, j=j, w=w))
        radius = float(item["radius"])  # required
        weight = float(item.get("weight", 1.0))
        constraints.append(NormEq(terms=terms, radius=radius, weight=weight))

    return PcfmSpec(variables=variables, constraints=constraints)


__all__ = [
    "Term",
    "NormEq",
    "PcfmSpec",
    "make_hook",
    "build_spec_from_json",
]
