from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

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


@dataclass(frozen=True)
class Var:
    field: str
    i: int
    j: int


@dataclass(frozen=True)
class RatioEq:
    num: Var
    den: Var
    target: float
    eps: float = 1e-6
    weight: float = 1.0


@dataclass(frozen=True)
class ProductEq:
    a: Var
    b: Var
    target: float
    weight: float = 1.0


@dataclass(frozen=True)
class ArBand:
    major: Var
    minor: Tuple[Var, ...]
    amin: float
    amax: float
    weight: float = 1.0


@dataclass(frozen=True)
class EdgeIotaEq:
    major: Var
    helical: Tuple[Var, ...]
    target: float
    eps: float = 1e-6
    weight: float = 1.0


@dataclass(frozen=True)
class ClearanceMin:
    major: Var
    helical: Tuple[Var, ...]
    minimum: float
    weight: float = 1.0


Constraint = Union[NormEq, RatioEq, ProductEq, ArBand, EdgeIotaEq, ClearanceMin]


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

    total_rows = sum(2 if isinstance(con, ArBand) else 1 for con in constraints)

    def _value_at(var: Var, x: NDArray[np.floating[Any]]) -> float:
        return float(x[index[(var.field, int(var.i), int(var.j))]])

    def _values_at(vars: Sequence[Var], x: NDArray[np.floating[Any]]) -> np.ndarray:
        return np.asarray([_value_at(v, x) for v in vars], dtype=float)

    def _aspect_ratio_and_grad(
        con: ArBand, x: NDArray[np.floating[Any]]
    ) -> Tuple[float, Dict[int, float]]:
        major = _value_at(con.major, x)
        minor_vals = _values_at(con.minor, x)
        minor_norm = float(np.linalg.norm(minor_vals))
        denom = minor_norm + 1e-8
        aspect = abs(major) / denom
        grad: Dict[int, float] = {}
        sign_major = 0.0 if major == 0.0 else (1.0 if major > 0.0 else -1.0)
        grad[index[(con.major.field, int(con.major.i), int(con.major.j))]] = sign_major / denom
        if minor_norm > 1e-8:
            factor = -abs(major) / (denom * denom)
            inv_norm = 1.0 / minor_norm
            for v, val in zip(con.minor, minor_vals.tolist()):
                grad[index[(v.field, int(v.i), int(v.j))]] = factor * inv_norm * float(val)
        else:
            for v in con.minor:
                grad[index[(v.field, int(v.i), int(v.j))]] = 0.0
        return aspect, grad

    def _edge_iota_and_grad(
        con: EdgeIotaEq, x: NDArray[np.floating[Any]]
    ) -> Tuple[float, Dict[int, float]]:
        major = _value_at(con.major, x)
        helical_vals = _values_at(con.helical, x)
        helical_norm = float(np.linalg.norm(helical_vals))
        denom = abs(major) + float(con.eps)
        iota = helical_norm / denom
        grad: Dict[int, float] = {}
        sign_major = 0.0 if major == 0.0 else (1.0 if major > 0.0 else -1.0)
        helical_norm_eff = float(con.eps) if helical_norm <= 1e-8 else helical_norm
        grad[index[(con.major.field, int(con.major.i), int(con.major.j))]] = (
            -helical_norm_eff * sign_major / (denom * denom)
        )
        if helical_norm <= 1e-8 and con.helical:
            fallback = 1.0 / math.sqrt(len(con.helical))
            for v in con.helical:
                grad[index[(v.field, int(v.i), int(v.j))]] = fallback / denom
        else:
            coeff = 1.0 / (denom * helical_norm_eff)
            for v, val in zip(con.helical, helical_vals.tolist()):
                grad[index[(v.field, int(v.i), int(v.j))]] = coeff * float(val)
        return iota, grad

    def _clearance_and_grad(
        con: ClearanceMin, x: NDArray[np.floating[Any]]
    ) -> Tuple[float, Dict[int, float]]:
        major = _value_at(con.major, x)
        helical_vals = _values_at(con.helical, x)
        helical_norm = float(np.linalg.norm(helical_vals))
        clearance = abs(major) - helical_norm
        grad: Dict[int, float] = {}
        sign_major = 0.0 if major == 0.0 else (1.0 if major > 0.0 else -1.0)
        grad[index[(con.major.field, int(con.major.i), int(con.major.j))]] = sign_major
        if helical_norm > 1e-8:
            coeff = -1.0 / helical_norm
            for v, val in zip(con.helical, helical_vals.tolist()):
                grad[index[(v.field, int(v.i), int(v.j))]] = coeff * float(val)
        else:
            for v in con.helical:
                grad[index[(v.field, int(v.i), int(v.j))]] = 0.0
        return clearance, grad

    def residual(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        rs: List[float] = []
        for con in constraints:
            if isinstance(con, NormEq):
                s = 0.0
                for t in con.terms:
                    k = index[(t.field, int(t.i), int(t.j))]
                    s += float(t.w) * float(x[k]) * float(x[k])
                rs.append(con.weight * (s - float(con.radius) * float(con.radius)))
            elif isinstance(con, RatioEq):
                kn = index[(con.num.field, int(con.num.i), int(con.num.j))]
                kd = index[(con.den.field, int(con.den.i), int(con.den.j))]
                denom = float(x[kd]) + float(con.eps)
                rs.append(con.weight * ((float(x[kn]) / denom) - float(con.target)))
            elif isinstance(con, ProductEq):
                ka = index[(con.a.field, int(con.a.i), int(con.a.j))]
                kb = index[(con.b.field, int(con.b.i), int(con.b.j))]
                rs.append(con.weight * (float(x[ka]) * float(x[kb]) - float(con.target)))
            elif isinstance(con, ArBand):
                aspect, _ = _aspect_ratio_and_grad(con, x)
                hi = max(0.0, aspect - float(con.amax))
                lo = max(0.0, float(con.amin) - aspect)
                rs.append(con.weight * hi)
                rs.append(con.weight * lo)
            elif isinstance(con, EdgeIotaEq):
                iota, _ = _edge_iota_and_grad(con, x)
                rs.append(con.weight * (iota - float(con.target)))
            else:  # ClearanceMin
                clearance, _ = _clearance_and_grad(con, x)
                rs.append(con.weight * max(0.0, float(con.minimum) - clearance))
        return np.asarray(rs, dtype=float)

    def jacobian(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        m = total_rows
        n = len(variables)
        J = np.zeros((m, n), dtype=float)
        row = 0
        for con in constraints:
            if isinstance(con, NormEq):
                for t in con.terms:
                    k = index[(t.field, int(t.i), int(t.j))]
                    J[row, k] += con.weight * (2.0 * float(t.w) * float(x[k]))
                row += 1
            elif isinstance(con, RatioEq):
                kn = index[(con.num.field, int(con.num.i), int(con.num.j))]
                kd = index[(con.den.field, int(con.den.i), int(con.den.j))]
                denom = float(x[kd]) + float(con.eps)
                J[row, kn] += con.weight * (1.0 / denom)
                J[row, kd] += con.weight * (-(float(x[kn]) / (denom * denom)))
                row += 1
            elif isinstance(con, ProductEq):
                ka = index[(con.a.field, int(con.a.i), int(con.a.j))]
                kb = index[(con.b.field, int(con.b.i), int(con.b.j))]
                J[row, ka] += con.weight * float(x[kb])
                J[row, kb] += con.weight * float(x[ka])
                row += 1
            elif isinstance(con, ArBand):
                aspect, grad = _aspect_ratio_and_grad(con, x)
                if aspect > float(con.amax):
                    for k, g in grad.items():
                        J[row, k] += con.weight * g
                row += 1
                if aspect < float(con.amin):
                    for k, g in grad.items():
                        J[row, k] += -con.weight * g
                row += 1
            elif isinstance(con, EdgeIotaEq):
                _, grad = _edge_iota_and_grad(con, x)
                for k, g in grad.items():
                    J[row, k] += con.weight * g
                row += 1
            else:  # ClearanceMin
                clearance, grad = _clearance_and_grad(con, x)
                if clearance < float(con.minimum):
                    for k, g in grad.items():
                        J[row, k] += -con.weight * g
                row += 1
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

    def parse_var(data: Mapping[str, Any]) -> Var:
        field = str(data["field"])  # required
        i = int(data["i"])  # required
        j = int(data["j"])  # required
        get_var(field, i, j)
        return Var(field=field, i=i, j=j)

    constraints: List[Constraint] = []
    for item in data:
        t = str(item.get("type", "norm_eq")).lower()
        if t == "norm_eq":
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
        elif t == "ratio_eq":
            num = parse_var(item["num"])  # required
            den = parse_var(item["den"])  # required
            target = float(item["target"])  # required
            eps = float(item.get("eps", 1e-6))
            weight = float(item.get("weight", 1.0))
            constraints.append(RatioEq(num=num, den=den, target=target, eps=eps, weight=weight))
        elif t == "product_eq":
            a = parse_var(item["a"])  # required
            b = parse_var(item["b"])  # required
            target = float(item["target"])  # required
            weight = float(item.get("weight", 1.0))
            constraints.append(ProductEq(a=a, b=b, target=target, weight=weight))
        elif t == "ar_band":
            major = parse_var(item["major"])  # required
            minor_seq = item.get("minor") or item.get("minor_terms")
            if not minor_seq:
                raise ValueError("ar_band constraint requires 'minor' terms")
            minor = tuple(parse_var(d) for d in minor_seq)
            amin = float(item["amin"])  # required
            amax = float(item["amax"])  # required
            weight = float(item.get("weight", 1.0))
            constraints.append(
                ArBand(major=major, minor=minor, amin=amin, amax=amax, weight=weight)
            )
        elif t == "edge_iota":
            major = parse_var(item["major"])  # required
            helical_seq = item.get("helical")
            if not helical_seq:
                raise ValueError("edge_iota constraint requires 'helical' terms")
            helical = tuple(parse_var(d) for d in helical_seq)
            target = float(item["target"])  # required
            eps = float(item.get("eps", 1e-6))
            weight = float(item.get("weight", 1.0))
            constraints.append(
                EdgeIotaEq(major=major, helical=helical, target=target, eps=eps, weight=weight)
            )
        elif t == "clearance_min":
            major = parse_var(item["major"])  # required
            helical_seq = item.get("helical")
            if not helical_seq:
                raise ValueError("clearance_min constraint requires 'helical' terms")
            helical = tuple(parse_var(d) for d in helical_seq)
            minimum = float(item["min"])  # required
            weight = float(item.get("weight", 1.0))
            constraints.append(
                ClearanceMin(major=major, helical=helical, minimum=minimum, weight=weight)
            )
        else:
            raise ValueError("Unsupported PCFM constraint type: " + t)

    return PcfmSpec(variables=variables, constraints=constraints)


__all__ = [
    "Term",
    "NormEq",
    "Var",
    "RatioEq",
    "ProductEq",
    "ArBand",
    "EdgeIotaEq",
    "ClearanceMin",
    "PcfmSpec",
    "make_hook",
    "build_spec_from_json",
]
