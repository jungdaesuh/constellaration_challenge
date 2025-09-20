from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Callable, List, Mapping, Sequence, Tuple, TypeVar

_T = TypeVar("_T")


def _as_float_tuple(values: Sequence[object]) -> Tuple[float, ...]:
    out: List[float] = []
    for v in values:
        if isinstance(v, bool):
            raise ValueError("Boolean values are not valid objectives")
        if not isinstance(v, (int, float)):
            raise ValueError("Objective values must be numeric")
        fv = float(v)
        if isnan(fv):
            raise ValueError("Objective values cannot be NaN")
        out.append(fv)
    return tuple(out)


def _broadcast(values: Sequence[float], *, dim: int, name: str) -> Tuple[float, ...]:
    if len(values) == dim:
        return tuple(values)
    if len(values) == 1:
        return tuple(values[0] for _ in range(dim))
    raise ValueError(f"{name} length {len(values)} does not match objective dimension {dim}")


def _normalize_weights(weights: Sequence[float], *, dim: int) -> Tuple[float, ...]:
    w = _broadcast(weights, dim=dim, name="weights")
    if any(val < 0.0 for val in w):
        raise ValueError("Weights must be non-negative")
    total = sum(w)
    if total <= 0.0:
        raise ValueError("At least one weight must be positive")
    return tuple(val / total for val in w)


@dataclass(frozen=True)
class ScalarizationConfig:
    """Configuration for scalarizing a multi-objective vector.

    Methods:
        - ``weighted_sum``: simple convex combination ``sum(w_i * f_i)``.
        - ``weighted_chebyshev``: ``max_i w_i * |f_i - r_i| + rho * sum_i w_i * |f_i - r_i|``.
    """

    method: str
    weights: Tuple[float, ...]
    reference_point: Tuple[float, ...] | None = None
    rho: float = 1e-6

    def __post_init__(self) -> None:  # pragma: no cover - trivial guard
        if not self.weights:
            raise ValueError("ScalarizationConfig.weights cannot be empty")
        method_norm = self.method.lower().strip()
        if method_norm not in {"weighted_sum", "weighted_chebyshev"}:
            raise ValueError(f"Unsupported scalarization method: {self.method}")
        if self.reference_point is not None and not self.reference_point:
            raise ValueError("reference_point, when provided, must be non-empty")
        object.__setattr__(self, "method", method_norm)


def scalarize(objectives: Sequence[object], config: ScalarizationConfig) -> float:
    """Scalarize a multi-objective vector according to ``config``.

    Raises ``ValueError`` when objectives contain NaN or when configuration does not
    match the objective dimensionality.
    """

    vals = _as_float_tuple(tuple(objectives))
    dim = len(vals)
    weights = _normalize_weights(config.weights, dim=dim)
    if config.method == "weighted_sum":
        return sum(val * weight for val, weight in zip(vals, weights))

    reference = config.reference_point
    if reference is None:
        reference_vals = (0.0,) * dim
    else:
        reference_vals = _broadcast(reference, dim=dim, name="reference_point")
    deviations = [abs(val - ref) for val, ref in zip(vals, reference_vals)]
    weighted = [weight * dev for weight, dev in zip(weights, deviations)]
    chebyshev = max(weighted) if weighted else 0.0
    if weighted:
        chebyshev += config.rho * sum(weighted)
    return chebyshev


def extract_objectives(metrics: Mapping[str, object]) -> Tuple[float, ...] | None:
    """Extract objectives from evaluator metrics if present.

    Supports either an ``objectives`` iterable or ``objective_{i}`` keys.
    Returns ``None`` when objectives cannot be parsed cleanly.
    """

    raw = metrics.get("objectives")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        try:
            return _as_float_tuple(list(raw))
        except ValueError:
            return None

    collected: List[float] = []
    idx = 0
    while True:
        key = f"objective_{idx}"
        if key not in metrics:
            break
        val = metrics.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return None
        fv = float(val)
        if isnan(fv):
            return None
        collected.append(fv)
        idx += 1
    if collected:
        return tuple(collected)
    return None


def dominates(
    a: Sequence[float], b: Sequence[float], *, minimize: bool = True, atol: float = 1e-12
) -> bool:
    """Return True if ``a`` Pareto-dominates ``b`` under the given sense."""

    if len(a) != len(b):
        raise ValueError("Objective vectors must have the same length")
    if minimize:
        better_or_equal = all(x <= y + atol for x, y in zip(a, b))
        strictly_better = any(x < y - atol for x, y in zip(a, b))
    else:
        better_or_equal = all(x >= y - atol for x, y in zip(a, b))
        strictly_better = any(x > y + atol for x, y in zip(a, b))
    return better_or_equal and strictly_better


def pareto_indices(points: Sequence[Sequence[float]], *, minimize: bool = True) -> List[int]:
    """Return indices of the Pareto front for ``points``."""

    result: List[int] = []
    for i, pt in enumerate(points):
        dominated = False
        vec_i = tuple(pt)
        for j, other in enumerate(points):
            if i == j:
                continue
            if dominates(tuple(other), vec_i, minimize=minimize):
                dominated = True
                break
        if not dominated:
            result.append(i)
    return result


def pareto_front(
    records: Sequence[_T], *, key: Callable[[_T], Sequence[float]], minimize: bool = True
) -> List[_T]:
    """Filter ``records`` to the Pareto front defined by ``key``."""

    keyed = [tuple(key(rec)) for rec in records]
    front_idx = pareto_indices(keyed, minimize=minimize)
    return [records[i] for i in front_idx]


def linspace_weights(dim: int, count: int) -> List[Tuple[float, ...]]:
    """Generate simplex-aligned weights for sweeps (supports dim>=1)."""

    if dim <= 0:
        raise ValueError("dim must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if dim == 1:
        return [(1.0,)]
    if dim == 2:
        if count == 1:
            return [(1.0, 0.0)]
        return [(i / (count - 1), 1.0 - i / (count - 1)) for i in range(count)]
    # For higher dimensions, fall back to equal weights per corner and centroid
    weights: List[Tuple[float, ...]] = [
        tuple(1.0 if i == j else 0.0 for j in range(dim)) for i in range(dim)
    ]
    weights.append(tuple(1.0 / dim for _ in range(dim)))
    return weights


DEFAULT_P3_SCALARIZATION = ScalarizationConfig(
    method="weighted_chebyshev",
    weights=(0.6, 0.4),
    reference_point=(0.0, 0.0),
    rho=1e-6,
)


__all__ = [
    "ScalarizationConfig",
    "scalarize",
    "extract_objectives",
    "dominates",
    "pareto_indices",
    "pareto_front",
    "linspace_weights",
    "DEFAULT_P3_SCALARIZATION",
]
