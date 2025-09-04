"""
Evaluation helpers for ConStellaration.

This module provides a thin adapter around the `constellaration` package to:
- validate/convert boundary JSON into a VMEC-compatible boundary object
- run a forward evaluation that returns a metrics dict
- aggregate a scalar score from metrics with simple, deterministic rules

Notes:
- The current implementation delegates to `constelx.physics.constel_api` which
  contains lightweight placeholders. Swap to direct `constellaration` calls as
  the external API stabilizes.
"""

from __future__ import annotations

from math import inf, isnan
from typing import Any, Dict, Mapping, TypeAlias

from constellaration.geometry import surface_rz_fourier

from ..physics.constel_api import evaluate_boundary

# A simple local type alias; keep as Any to avoid leaking third-party types.
VmecBoundary: TypeAlias = Any


def boundary_to_vmec(boundary: Mapping[str, Any]) -> VmecBoundary:
    """Validate and convert a boundary JSON dict to a VMEC boundary object.

    Parameters
    - boundary: Mapping with keys matching `SurfaceRZFourier` fields

    Returns
    - A validated `SurfaceRZFourier` instance, usable as VMEC boundary input.
    """

    return surface_rz_fourier.SurfaceRZFourier.model_validate(boundary)


def forward(boundary: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the forward evaluator for a single boundary.

    Parameters
    - boundary: Boundary specification (JSON-like dict) using `SurfaceRZFourier` fields.

    Returns
    - Dict of metric names to values (numeric or informative non-numeric entries).

    This starter delegates to `constelx.physics.constel_api.evaluate_boundary` which
    provides lightweight, deterministic metrics. Replace with direct evaluator calls
    to compute physical figures of merit once available.
    """

    # Validate inputs to provide clear errors early; convert to pydantic model if needed.
    _ = boundary_to_vmec(boundary)
    # evaluate_boundary expects a plain dict
    return evaluate_boundary(dict(boundary))


def score(metrics: Mapping[str, Any]) -> float:
    """Aggregate a scalar score from a metrics dict.

    Rules (deterministic and simple by design):
    - Consider only numeric (int/float) values.
    - If any considered value is NaN, return +inf (treat as invalid/bad).
    - Otherwise return the sum of numeric values (lower is better).

    This is a placeholder aggregation compatible with the starter's toy metrics.
    Swap in evaluator-default aggregation when integrating the real metrics.
    """

    total = 0.0
    for v in metrics.values():
        # Exclude booleans explicitly (bool is a subclass of int in Python).
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
            if isnan(fv):
                return inf
            total += fv
    return float(total)
