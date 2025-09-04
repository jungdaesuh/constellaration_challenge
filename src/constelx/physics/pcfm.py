"""Physics-Constrained Flow Matching (PCFM) — placeholder

Use this module to integrate a pretrained flow model while projecting intermediate states onto
hard constraints (equality type), e.g., global conservation, boundary values, local flux constraints.

Key ideas to implement:
- Forward 'shoot' to t=1, Gauss–Newton projection on constraint manifold.
- Reverse update via OT displacement interpolant (stable approx of reverse flow).
- Optional relaxed correction with penalty on residuals.
- Final projection to enforce h(u1)=0 to numerical precision.
"""

from typing import Callable, Dict, Any, Tuple

def project_gauss_newton(u, residual: Callable, jacobian: Callable):
    """One Gauss–Newton projection step onto linearized constraints."""
    # TODO: implement
    return u

def guided_step(u, model, t, dt, constraint):
    """One PCFM-guided update step."""
    # TODO: implement
    return u
