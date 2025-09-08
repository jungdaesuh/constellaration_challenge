from __future__ import annotations

from constelx.eval.boundary_fourier import BoundaryFourier
from constelx.eval.boundary_param import validate


def test_boundary_fourier_round_trip_and_idx() -> None:
    bf = BoundaryFourier.empty(nfp=3, m_dim=3, n_dim=7, n_offset=3)
    # set base radius at conventional index used by validator (j=4)
    bf.r_cos[0][4] = 1.0
    # set a helical pair with negative n
    i1, j1 = bf.idx(1, -2)
    bf.r_cos[i1][j1] = -0.05
    bf.z_sin[i1][j1] = 0.05

    b = bf.to_surface_rz_fourier_dict()
    # validate dict shape
    validate(b)
    # reconstruct and sample a tiny grid
    bf2 = BoundaryFourier.from_surface_dict(b, n_offset=3)
    R, Z = bf2.sample_surface(n_theta=4, n_phi=4)
    assert len(R) == 4 and len(R[0]) == 4
    assert len(Z) == 4 and len(Z[0]) == 4
