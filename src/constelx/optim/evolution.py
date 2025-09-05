from __future__ import annotations

import math
from typing import Any, Dict

try:
    import cma
except Exception:  # optional dep
    cma = None

from ..physics.constel_api import evaluate_boundary, example_boundary


def score_from_boundary(b: Dict[str, Any]) -> float:
    """Toy score based on available placeholder metrics.

    Prefer the combined placeholder metric; fall back to the sum of norms.
    Lower is better.
    """
    m = evaluate_boundary(b)
    rc = float(m.get("r_cos_norm", 0.0))
    zs = float(m.get("z_sin_norm", 0.0))
    return float(m.get("placeholder_metric", rc + zs))


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
