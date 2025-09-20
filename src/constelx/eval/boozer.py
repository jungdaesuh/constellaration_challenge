"""Boozer-space quasi-symmetry and quasi-isodynamic proxy metrics.

This module offers a lightweight, dependency-free (besides NumPy) implementation of
proxy observables that correlate with quasi-symmetry (QS) and quasi-isodynamic (QI)
quality. The proxies intentionally avoid heavy Boozer transforms by operating on the
existing truncated Fourier representation of stellarator-symmetric boundaries.

The design goals are:
- Deterministic, cheap to evaluate (O(m n n_theta n_phi))
- Bounded residuals in [0, 1] so that downstream constraint handlers can rely on
  consistent scales
- Simple to integrate into guardrails or proxy-evaluation paths before the real
  evaluator is available

Example
-------
>>> from constelx.physics.constel_api import example_boundary
>>> proxies = compute_boozer_proxies(example_boundary())
>>> proxies.qs_residual <= 1.0
True
>>> proxies.to_dict(prefix="proxy_")  # doctest: +ELLIPSIS
{'proxy_b_mean': ..., 'proxy_b_std_fraction': ..., ...}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .boundary_fourier import BoundaryFourier

__all__ = ["BoozerProxies", "compute_boozer_proxies"]

_EPS = 1e-12

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _bounded_ratio(value: float) -> float:
    """Map a non-negative value to [0, 1) using ``x -> x / (1 + x)``."""
    v = max(0.0, float(value))
    return v / (1.0 + v)


@dataclass(frozen=True)
class BoozerProxies:
    """Container for Boozer-derived proxy metrics."""

    b_mean: float
    b_std_fraction: float
    mirror_ratio: float
    qs_residual: float
    qs_quality: float
    qi_residual: float
    qi_quality: float

    @staticmethod
    def zeros() -> "BoozerProxies":
        """Return a zero-initialized proxy payload."""
        return BoozerProxies(
            b_mean=0.0,
            b_std_fraction=0.0,
            mirror_ratio=0.0,
            qs_residual=0.0,
            qs_quality=1.0,
            qi_residual=0.0,
            qi_quality=1.0,
        )

    def to_dict(self, *, prefix: str | None = None) -> dict[str, float]:
        """Convert proxies to a dict with optional key prefixing."""
        data = {
            "b_mean": float(self.b_mean),
            "b_std_fraction": float(self.b_std_fraction),
            "mirror_ratio": float(self.mirror_ratio),
            "qs_residual": float(self.qs_residual),
            "qs_quality": float(self.qs_quality),
            "qi_residual": float(self.qi_residual),
            "qi_quality": float(self.qi_quality),
        }
        if prefix:
            return {f"{prefix}{k}": v for k, v in data.items()}
        return data


def _angles_for_sampling(n_theta: int, n_phi: int, nfp: int) -> tuple[FloatArray, FloatArray]:
    thetas: FloatArray = np.linspace(
        0.0, 2.0 * math.pi, num=n_theta, endpoint=False, dtype=float
    )
    # Toroidal angle spans one field period (2π / nfp)
    phis: FloatArray = np.linspace(
        0.0, 2.0 * math.pi / max(1, nfp), num=n_phi, endpoint=False, dtype=float
    )
    return thetas, phis


def _sample_surface_arrays(
    bf: BoundaryFourier, thetas: FloatArray, phis: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return (R, Z) arrays sampled on the theta/phi grid."""
    theta_grid: FloatArray
    phi_grid: FloatArray
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing="ij")
    R: FloatArray = np.zeros_like(theta_grid, dtype=float)
    Z: FloatArray = np.zeros_like(theta_grid, dtype=float)

    for m in range(bf.m_dim):
        m_theta = m * theta_grid
        for j in range(bf.n_dim):
            n = j - bf.n_offset
            phase = m_theta - n * phi_grid
            a = float(bf.r_cos[m][j])
            b = float(bf.z_sin[m][j])
            if a != 0.0:
                R += a * np.cos(phase)
            if b != 0.0:
                Z += b * np.sin(phase)
    return R, Z


def compute_boozer_proxies(
    boundary: Mapping[str, Any],
    *,
    helicity: int | None = None,
    n_theta: int = 24,
    n_phi: int = 24,
) -> BoozerProxies:
    """Compute Boozer-space proxy metrics from a boundary description.

    Parameters
    ----------
    boundary:
        Mapping compatible with ``SurfaceRZFourier`` (requires ``r_cos`` and ``z_sin`` fields).
    helicity:
        Integer Boozer helicity ``N`` for the quasi-symmetry residual. Defaults to
        ``boundary['n_field_periods']``.
    n_theta, n_phi:
        Sampling resolution for the coarse grid used to evaluate the proxies.

    Returns
    -------
    BoozerProxies
        Container with bounded residuals and quality scores.
    """

    try:
        bf = BoundaryFourier.from_surface_dict(dict(boundary))
    except Exception:
        return BoozerProxies.zeros()

    if n_theta <= 4 or n_phi <= 4:
        raise ValueError("n_theta and n_phi must be >= 5 for meaningful sampling")

    thetas, phis = _angles_for_sampling(int(n_theta), int(n_phi), bf.nfp)
    R, _ = _sample_surface_arrays(bf, thetas, phis)

    r_mean = float(np.mean(R))
    if not math.isfinite(r_mean) or abs(r_mean) <= _EPS:
        return BoozerProxies.zeros()

    B = r_mean / np.clip(np.abs(R), _EPS, None)
    B_mean = float(np.mean(B))
    B_std_fraction = _bounded_ratio(float(np.std(B) / (B_mean + _EPS)))
    B_max = float(np.max(B))
    B_min = float(np.min(B))
    mirror_ratio = _bounded_ratio((B_max - B_min) / (B_min + _EPS))

    if helicity is None:
        helicity = int(boundary.get("n_field_periods", bf.nfp))
    helicity = int(helicity)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing="ij")
    alpha: FloatArray = (theta_grid - helicity * phi_grid) % (2.0 * math.pi)
    n_bins = max(8, int(n_phi))
    bin_idx: IntArray = np.floor(alpha / (2.0 * math.pi) * n_bins).astype(int)
    bin_idx = np.mod(bin_idx, n_bins)

    flat_bins: IntArray = bin_idx.ravel()
    flat_B: FloatArray = B.ravel()

    sums: FloatArray = np.zeros(n_bins, dtype=float)
    counts: IntArray = np.zeros(n_bins, dtype=int)
    np.add.at(sums, flat_bins, flat_B)
    np.add.at(counts, flat_bins, 1)

    means: FloatArray = np.divide(
        sums, counts, out=np.zeros_like(sums), where=counts > 0
    )
    centered: FloatArray = flat_B - means[flat_bins]
    qs_rms = math.sqrt(float(np.mean(centered * centered)))
    qs_res_norm = qs_rms / (B_mean + _EPS)
    qs_residual = _bounded_ratio(qs_res_norm)
    qs_quality = 1.0 - qs_residual

    B_min_phi = np.min(B, axis=0)
    mean_min = float(np.mean(B_min_phi))
    min_spread = float(np.std(B_min_phi) / (mean_min + _EPS))
    qi_residual = _bounded_ratio(min_spread)
    qi_quality = 1.0 - qi_residual

    return BoozerProxies(
        b_mean=B_mean,
        b_std_fraction=B_std_fraction,
        mirror_ratio=mirror_ratio,
        qs_residual=qs_residual,
        qs_quality=qs_quality,
        qi_residual=qi_residual,
        qi_quality=qi_quality,
    )
