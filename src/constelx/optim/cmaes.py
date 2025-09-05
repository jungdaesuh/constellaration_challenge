from __future__ import annotations

from typing import Callable, Sequence, Tuple

try:
    import cma
except Exception:  # pragma: no cover - optional dependency
    cma = None


def optimize(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    bounds: Tuple[float, float],
    budget: int,
    *,
    sigma0: float = 0.2,
    seed: int | None = None,
) -> tuple[list[float], list[float]]:
    """Run a simple CMA-ES optimization loop.

    Returns (best_x, history_scores).
    """
    if cma is None:
        raise RuntimeError("CMA-ES not installed. Install extra: pip install cma")

    opts: dict = {"bounds": list(bounds)}
    if seed is not None:
        opts["seed"] = int(seed)

    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    best_score = float("inf")
    best_x = list(x0)
    history: list[float] = []
    steps = max(1, int(budget))
    for _ in range(steps):
        xs = es.ask()
        scores = []
        for x in xs:
            s = float(f(x))
            if s < best_score:
                best_score = s
                best_x = [float(v) for v in x]
            scores.append(s)
        history.append(min(scores))
        es.tell(xs, scores)
    return best_x, history
