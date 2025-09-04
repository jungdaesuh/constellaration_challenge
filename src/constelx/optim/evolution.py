from __future__ import annotations

import math
from typing import Any, Dict

try:
    import cma
except Exception:  # optional dep
    cma = None

from ..physics.constel_api import evaluate_boundary, example_boundary


def score_from_boundary(b: Dict[str, Any]) -> float:
    """Toy score: minimize a weighted sum of geom metrics (placeholder)."""
    m = evaluate_boundary(b)
    # Example: smaller is better (this metric dict is placeholder-friendly)
    return float(m.get("compactness", 0.0)) + 0.1 * float(m.get("smoothness", 0.0))


def run_cma_es_baseline(steps: int = 50) -> float:
    if cma is None:
        raise RuntimeError(
            "Install extra 'evolution' (pip install -e '.[evolution]') to use CMA-ES."
        )

    # Parametrize a tiny subspace: two coefficients controlling helical perturbation
    x0 = [0.05, 0.05]
    sigma0 = 0.02
    es = cma.CMAEvolutionStrategy(x0, sigma0, {"bounds": [-0.2, 0.2]})

    def make_boundary(x: list[float]) -> Dict[str, Any]:
        b = example_boundary()
        # helical (m=1, n=1) perturbations
        b["r_cos"][1][5] = float(-abs(x[0]))
        b["z_sin"][1][5] = float(abs(x[1]))
        return b

    best = math.inf
    for _ in range(steps):
        xs = es.ask()
        scores = []
        for x in xs:
            s = score_from_boundary(make_boundary(x))
            scores.append(s)
            best = min(best, s)
        es.tell(xs, scores)
    return best
