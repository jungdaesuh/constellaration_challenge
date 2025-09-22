"""Boundary Fourier helpers (R,Z in cylindrical coordinates).

This module provides a small convenience wrapper around a truncated Fourier
representation compatible with ConStelX's `SurfaceRZFourier` dictionary
format used throughout the project. It intentionally supports a configurable
`n_offset` to allow addressing negative toroidal mode numbers (n < 0) via a
single 0-based index.

Notes
- We only expose the stellarator-symmetric subset (r_cos, z_sin). The fields
  r_sin and z_cos are set to None in `to_surface_rz_fourier_dict()`.
- The phase convention used for sampling is cos(m*theta - n*phi) for r_cos and
  sin(m*theta - n*phi) for z_sin, which is standard for a stellarator-symmetric
  cylindrical surface parameterization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

Coeff2D = List[List[float]]


def _zeros(m: int, n: int) -> Coeff2D:
    return [[0.0 for _ in range(n)] for _ in range(m)]


@dataclass
class BoundaryFourier:
    """Truncated Fourier series helper for stellarator-symmetric surfaces.

    Attributes
    - nfp: number of field periods (positive int)
    - m_dim: number of poloidal indices m in [0, m_dim-1]
    - n_dim: number of toroidal indices stored; logical n indices are
             n = j - n_offset for stored column j in [0, n_dim-1]
    - n_offset: shift so that negative n can be addressed by j = n + n_offset
    - r_cos, z_sin: coefficient arrays with shape (m_dim, n_dim)
    """

    nfp: int
    m_dim: int
    n_dim: int
    n_offset: int
    r_cos: Coeff2D
    z_sin: Coeff2D

    @staticmethod
    def empty(
        nfp: int, m_dim: int, n_dim: int, *, n_offset: int | None = None
    ) -> "BoundaryFourier":
        if nfp <= 0:
            raise ValueError("nfp must be positive")
        if m_dim <= 0 or n_dim <= 0:
            raise ValueError("m_dim and n_dim must be positive")
        if n_offset is None:
            # Default: center zero at middle column when possible
            n_offset = n_dim // 2
        if not (0 <= n_offset < n_dim):
            raise ValueError("n_offset must be within [0, n_dim-1]")
        return BoundaryFourier(
            nfp=int(nfp),
            m_dim=int(m_dim),
            n_dim=int(n_dim),
            n_offset=int(n_offset),
            r_cos=_zeros(m_dim, n_dim),
            z_sin=_zeros(m_dim, n_dim),
        )

    def idx(self, m: int, n: int) -> Tuple[int, int]:
        """Return (i,j) indices for coefficient with logical (m,n).

        n is mapped to storage column j = n + n_offset. Raises ValueError if
        the indices fall outside the truncated grid.
        """
        i = int(m)
        j = int(n) + int(self.n_offset)
        if not (0 <= i < self.m_dim):
            raise ValueError("m index out of range")
        if not (0 <= j < self.n_dim):
            raise ValueError("n index out of range after offset")
        return i, j

    def to_surface_rz_fourier_dict(self) -> Dict[str, Any]:
        return {
            "r_cos": [row[:] for row in self.r_cos],
            "r_sin": None,
            "z_cos": None,
            "z_sin": [row[:] for row in self.z_sin],
            "n_field_periods": int(self.nfp),
            "is_stellarator_symmetric": True,
        }

    @staticmethod
    def from_surface_dict(
        boundary: Dict[str, Any], *, n_offset: int | None = None
    ) -> "BoundaryFourier":
        """Create a BoundaryFourier from a SurfaceRZFourier-like dict.

        If n_offset is None, choose a centered offset (n_dim//2) by default.
        """
        r_cos = boundary.get("r_cos")
        z_sin = boundary.get("z_sin")
        if not isinstance(r_cos, list) or not isinstance(z_sin, list):
            raise ValueError("boundary must contain r_cos and z_sin 2D lists")
        m_dim = len(r_cos)
        n_dim = len(r_cos[0]) if m_dim > 0 else 0
        if n_dim <= 0:
            raise ValueError("empty coefficient grid")
        if n_offset is None:
            n_offset = n_dim // 2
        bf = BoundaryFourier.empty(
            int(boundary.get("n_field_periods", 3)), m_dim=m_dim, n_dim=n_dim, n_offset=n_offset
        )
        # Deep-copy arrays
        bf.r_cos = [[float(x) for x in row] for row in r_cos]
        bf.z_sin = [[float(x) for x in row] for row in z_sin]
        return bf

    def sample_surface(
        self, n_theta: int = 16, n_phi: int = 16
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Sample (R,Z) on a coarse grid for quick inspections.

        Returns two 2D lists with shapes (n_theta, n_phi).
        """
        thetas = [2.0 * math.pi * t / n_theta for t in range(n_theta)]
        phis = [2.0 * math.pi * p / (n_phi * self.nfp) for p in range(n_phi)]
        R = [[0.0 for _ in range(n_phi)] for _ in range(n_theta)]
        Z = [[0.0 for _ in range(n_phi)] for _ in range(n_theta)]
        for it, th in enumerate(thetas):
            for ip, ph in enumerate(phis):
                rc = 0.0
                zs = 0.0
                for m in range(self.m_dim):
                    for j in range(self.n_dim):
                        n = j - self.n_offset
                        a = float(self.r_cos[m][j])
                        b = float(self.z_sin[m][j])
                        phase = m * th - n * ph
                        rc += a * math.cos(phase)
                        zs += b * math.sin(phase)
                R[it][ip] = rc
                Z[it][ip] = zs
        return R, Z
