from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Variable:
    field: str  # e.g., "r_cos" or "z_sin"
    i: int
    j: int


@dataclass(frozen=True)
class LinearConstraint:
    coeffs: List[Tuple[Variable, float]]  # sum c_k * x[var_k] = rhs
    rhs: float


@dataclass(frozen=True)
class EciLinearSpec:
    variables: List[Variable]
    constraints: List[LinearConstraint]


def _flatten(
    boundary: Mapping[str, Any], variables: Sequence[Variable]
) -> NDArray[np.floating[Any]]:
    x = []
    for v in variables:
        arr = boundary[v.field]
        try:
            a = np.asarray(arr)
            x.append(float(a[v.i, v.j]))
        except Exception:
            raise AssertionError("Unsupported boundary coefficient array type")
    return np.asarray(x, dtype=float)


def _unflatten(
    boundary: Mapping[str, Any], variables: Sequence[Variable], x: NDArray[np.floating[Any]]
) -> Dict[str, Any]:
    b = dict(boundary)
    # deep copy only touched fields minimally
    for v, val in zip(variables, x.tolist()):
        arr = np.asarray(b[v.field]).copy()
        arr[v.i, v.j] = float(val)
        b[v.field] = arr.tolist()
    return b


def _build_A_b(
    variables: Sequence[Variable], constraints: Sequence[LinearConstraint]
) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    m = len(constraints)
    n = len(variables)
    A = np.zeros((m, n), dtype=float)
    b = np.zeros((m,), dtype=float)
    for r, con in enumerate(constraints):
        b[r] = float(con.rhs)
        for var, c in con.coeffs:
            try:
                k = variables.index(var)
            except ValueError:
                raise ValueError("Constraint references unknown variable")
            A[r, k] = float(c)
    return A, b


def project_linear(
    x0: NDArray[np.floating[Any]],
    A: NDArray[np.floating[Any]],
    b: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Project x0 onto the affine subspace {x | A x = b} minimizing ||x - x0||_2.

    Uses x* = x0 - A^T (A A^T)^-1 (A x0 - b). Handles rank deficiency with pinv.
    """
    r = A @ x0 - b
    if np.allclose(r, 0.0):
        return x0
    M = A @ A.T
    try:
        Minv: NDArray[np.floating[Any]] = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        Minv = np.linalg.pinv(M)
    return cast(NDArray[np.floating[Any]], x0 - A.T @ (Minv @ r))


def make_hook(spec: EciLinearSpec) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    def hook(boundary: Mapping[str, Any]) -> Dict[str, Any]:
        x0 = _flatten(boundary, spec.variables)
        A, b = _build_A_b(spec.variables, spec.constraints)
        x = project_linear(x0, A, b)
        return _unflatten(boundary, spec.variables, x)

    return hook


__all__ = [
    "Variable",
    "LinearConstraint",
    "EciLinearSpec",
    "project_linear",
    "make_hook",
]
