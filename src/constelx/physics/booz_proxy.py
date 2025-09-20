"""Boozer-space proxy metrics with safe fallbacks.

These helpers expose bounded quasi-symmetry (QS) and quasi-isodynamic (QI)
residual proxies that only depend on the boundary Fourier coefficients. When
`booz_xform` (and the real evaluator stack) is available the functions attempt to
call into the actual Boozer transform, otherwise they fall back to
light-weight heuristics that are deterministic and fast. The heuristics are
constructed to stay within ``[0, 1]`` so they can be consumed directly by
constraint projections or gating logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

EPS = 1e-12


def _should_try_real(use_real: bool | None) -> bool:
    if use_real is not None:
        return use_real
    import os

    return os.getenv("CONSTELX_USE_REAL_EVAL", "0").lower() in {"1", "true", "yes"}


def _iter_coefficients(boundary: Mapping[str, Any]) -> Iterable[tuple[int, int, float]]:
    for key in ("r_cos", "z_sin"):
        rows = boundary.get(key)
        if not isinstance(rows, Iterable):
            continue
        for m_idx, row in enumerate(rows):
            if not isinstance(row, Iterable):
                continue
            for n_idx, value in enumerate(row):
                try:
                    yield m_idx, n_idx, float(value)
                except Exception:
                    yield m_idx, n_idx, 0.0


def _energy(
    boundary: Mapping[str, Any], predicate: Callable[[int, int], bool] | None = None
) -> float:
    total = 0.0
    for m_idx, n_idx, val in _iter_coefficients(boundary):
        if predicate is None or predicate(m_idx, n_idx):
            total += val * val
    return float(total)


def _fraction(numerator: float, denominator: float) -> float:
    if denominator <= EPS:
        return 0.0
    return float(np.clip(numerator / (denominator + EPS), 0.0, 1.0))


def _mirror_ratio_proxy(boundary: Mapping[str, Any]) -> float:
    r0 = 0.0
    helical_sum = 0.0
    for m_idx, n_idx, val in _iter_coefficients(boundary):
        if m_idx == 0 and n_idx == 4:
            r0 += abs(val)
        elif m_idx > 0:
            helical_sum += abs(val)
    base = max(r0, 0.05)
    spread = helical_sum
    max_r = base + spread
    min_r = max(base - spread, 1e-3)
    ratio = max(max_r / min_r, 1.0)
    # Squash to [0, 1] with a smooth logistic-like map that keeps small
    # perturbations near zero while saturating as ratio grows.
    normalised = (ratio - 1.0) / (ratio + 1.0)
    return float(np.clip(normalised, 0.0, 1.0))


@dataclass(frozen=True)
class BoozerProxy:
    """Bundle of QS/QI proxy metrics bounded in ``[0, 1]``."""

    qs_residual: float
    qi_residual: float
    helical_energy: float
    mirror_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {
            "qs_residual": self.qs_residual,
            "qi_residual": self.qi_residual,
            "helical_energy": self.helical_energy,
            "mirror_ratio": self.mirror_ratio,
        }


def _real_proxies(boundary: Mapping[str, Any]) -> BoozerProxy | None:
    try:
        from constellaration.metrics import qs
        from constellaration.physics import vmec
    except Exception:
        return None
    try:
        surface = vmec.boundary_from_dict(boundary)
        metrics = qs.compute_qs_proxies(surface)
    except Exception:
        return None
    try:
        qs_residual = float(metrics.get("qs_residual", 0.0))
        qi_residual = float(metrics.get("qi_residual", 0.0))
        helical_energy = float(metrics.get("helical_energy", 0.0))
        mirror_ratio = float(metrics.get("mirror_ratio", 0.0))
    except Exception:
        return None
    return BoozerProxy(
        qs_residual=float(np.clip(qs_residual, 0.0, 1.0)),
        qi_residual=float(np.clip(qi_residual, 0.0, 1.0)),
        helical_energy=float(np.clip(helical_energy, 0.0, 1.0)),
        mirror_ratio=float(np.clip(mirror_ratio, 0.0, 1.0)),
    )


def _weighted_energy(
    boundary: Mapping[str, Any],
    weight_fn: Callable[[int, int], float],
) -> float:
    total = 0.0
    for m_idx, n_idx, val in _iter_coefficients(boundary):
        if m_idx == 0:
            continue
        total += val * val * max(0.0, weight_fn(m_idx, n_idx))
    return float(total)


def _heuristic_proxies(boundary: Mapping[str, Any]) -> BoozerProxy:
    total_energy = _energy(boundary)
    non_axis_energy = _energy(boundary, lambda m_idx, _: m_idx > 0)
    nfp_val = 0
    try:
        nfp_val = int(boundary.get("n_field_periods", 0) or 0)
    except Exception:
        nfp_val = 0
    nfp = max(1, nfp_val)

    def qs_weight(m_idx: int, n_idx: int) -> float:
        target = m_idx * nfp
        delta = float(abs(n_idx - target))
        norm = float(abs(n_idx) + abs(target) + 1)
        return (delta / norm) ** 2

    def qi_weight(m_idx: int, n_idx: int) -> float:
        penalty_m = float(max(0, m_idx - 1))
        penalty_n = float(max(0, abs(n_idx) - nfp))
        weight = (penalty_m + penalty_n) / (penalty_m + penalty_n + 1.0)
        return weight * weight

    qs_energy = _weighted_energy(boundary, qs_weight)
    qi_energy = _weighted_energy(boundary, qi_weight)
    helical_energy = _energy(boundary, lambda m_idx, _: m_idx == 1)
    helical_fraction = (
        _fraction(helical_energy, max(non_axis_energy, EPS)) if non_axis_energy > EPS else 0.0
    )

    return BoozerProxy(
        qs_residual=_fraction(qs_energy, max(non_axis_energy, total_energy)),
        qi_residual=_fraction(qi_energy, max(non_axis_energy, total_energy)),
        helical_energy=helical_fraction,
        mirror_ratio=_mirror_ratio_proxy(boundary),
    )


def compute_proxies(
    boundary: Mapping[str, Any],
    *,
    use_real: bool | None = None,
) -> BoozerProxy:
    """Compute bounded Boozer proxies for a boundary.

    Parameters
    ----------
    boundary:
        Mapping that follows the ``SurfaceRZFourier`` schema (only the Fourier
        coefficient arrays are consulted).
    use_real:
        When ``True`` try to call into the real Boozer transform if the
        dependencies are installed. ``None`` follows the ``CONSTELX_USE_REAL_EVAL``
        environment flag. Set to ``False`` to force the heuristic fallback.

    Returns
    -------
    BoozerProxy
        Dataclass with ``qs_residual``, ``qi_residual``, ``helical_energy`` and
        ``mirror_ratio`` values in ``[0, 1]``.
    """

    if _should_try_real(use_real):
        proxy = _real_proxies(boundary)
        if proxy is not None:
            return proxy
    return _heuristic_proxies(boundary)


BOOZER_PROXY_KEYS: tuple[str, ...] = (
    "qs_residual",
    "qi_residual",
    "helical_energy",
    "mirror_ratio",
)


__all__ = ["BoozerProxy", "BOOZER_PROXY_KEYS", "compute_proxies"]
