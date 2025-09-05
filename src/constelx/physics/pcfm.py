"""Physics-Constrained Flow Matching (PCFM) — core projection utilities.

This module provides a small, dependency-light Gauss–Newton projector to
enforce nonlinear equality constraints of the form ``h(u) = 0`` on a vector of
parameters ``u``. It is used by the agent's optional "pcfm" correction hook to
adjust boundary Fourier coefficients before evaluation.

The implementation uses a damped least-squares Gauss–Newton step with optional
finite-difference Jacobians, backtracking line search, and safety fallbacks.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray


def finite_diff_jacobian(
    fun: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    u: NDArray[np.floating[Any]],
    *,
    eps: Optional[float] = None,
) -> NDArray[np.floating[Any]]:
    """Compute a finite-difference Jacobian of ``fun`` at ``u``.

    Uses simple forward differences with element-wise step sizes scaled to the
    magnitude of ``u``. This is sufficient for the small dimensionalities used
    in the starter and keeps dependencies minimal.
    """
    u = np.asarray(u, dtype=float)
    f0 = np.asarray(fun(u), dtype=float)
    m = int(f0.size)
    n = int(u.size)
    J = np.zeros((m, n), dtype=float)
    # Per-coordinate step
    for j in range(n):
        h = (1e-6 if eps is None else float(eps)) * (1.0 + abs(float(u[j])))
        uj = float(u[j])
        u[j] = uj + h
        fj = np.asarray(fun(u), dtype=float)
        u[j] = uj
        J[:, j] = (fj - f0) / h
    return J


def project_gauss_newton(
    u: NDArray[np.floating[Any]],
    residual: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    jacobian: Optional[Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]] = None,
    *,
    max_iters: int = 3,
    tol: float = 1e-8,
    damping: float = 1e-6,
    backtracking: bool = True,
    max_backtracks: int = 5,
) -> NDArray[np.floating[Any]]:
    """Project ``u`` toward the manifold ``{u | h(u)=0}`` via damped Gauss–Newton.

    Solves at each iteration the damped least-squares subproblem::

        minimize 0.5||J du + h||^2 + 0.5*lambda*||du||^2

    using a stable least-squares formulation. Performs a short line-search to
    ensure residual reduction. Returns the updated vector (never raises).
    """
    x = np.asarray(u, dtype=float).copy()
    lam = float(damping)
    for _ in range(max_iters):
        h = np.asarray(residual(x), dtype=float)
        norm_h = float(np.linalg.norm(h))
        if not np.isfinite(norm_h) or norm_h <= tol:
            break
        J = (
            finite_diff_jacobian(residual, x)
            if jacobian is None
            else np.asarray(jacobian(x), dtype=float)
        )
        m, n = J.shape
        # Build damped least-squares system: [J; sqrt(lam) I] du = [-h; 0]
        A = np.vstack([J, np.sqrt(lam) * np.eye(n, dtype=float)])
        b = np.concatenate([-h, np.zeros(n, dtype=float)])
        try:
            du, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            du = -np.linalg.pinv(J) @ h

        # Backtracking line search on residual decrease
        alpha = 1.0
        accepted = False
        for _bt in range(max_backtracks if backtracking else 1):
            x_new = x + alpha * du
            h_new = np.asarray(residual(x_new), dtype=float)
            if np.linalg.norm(h_new) < norm_h:
                x = x_new
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            # Increase damping and try next GN iteration from same x
            lam *= 10.0
        # Convergence check next loop via h(x)
    return x


def guided_step(u: Any, model: Any, t: float, dt: float, constraint: Callable[[Any], Any]) -> Any:
    """Placeholder for a PCFM-guided update step (not used in starter)."""
    # Intentionally left minimal; end-to-end uses the projector above.
    return u


__all__ = ["finite_diff_jacobian", "project_gauss_newton", "guided_step"]
