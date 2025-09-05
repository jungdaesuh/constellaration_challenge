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

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import inf, isnan
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, TypeAlias, cast

from ..physics.constel_api import evaluate_boundary
from .cache import CacheBackend, get_cache_backend

# A simple local type alias; keep as Any to avoid leaking third-party types.
VmecBoundary: TypeAlias = Any


def boundary_to_vmec(boundary: Mapping[str, Any]) -> VmecBoundary:
    """Validate and convert a boundary JSON dict to a VMEC boundary object.

    Parameters
    - boundary: Mapping with keys matching `SurfaceRZFourier` fields

    Returns
    - A validated `SurfaceRZFourier` instance, usable as VMEC boundary input.
    """

    try:
        from constellaration.geometry import surface_rz_fourier

        return surface_rz_fourier.SurfaceRZFourier.model_validate(boundary)
    except Exception as e:
        msg = (
            "constellaration is not installed; install '.[physics]' and system NetCDF "
            "to enable VMEC boundary validation"
        )
        raise RuntimeError(msg) from e


def _normalize(obj: Any) -> Any:
    # Optional numpy support for normalizing arrays; keep typing mypy-clean.
    _np: Any | None
    try:
        import numpy as _np_mod

        _np = cast(Any, _np_mod)
    except Exception:  # pragma: no cover - numpy always available in deps
        _np = None

    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: kv[0])}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if _np is not None:
        try:
            np_mod = _np
            if isinstance(obj, np_mod.ndarray):
                return obj.tolist()
            if isinstance(obj, np_mod.generic):
                return obj.item()
        except Exception:
            pass
    return obj


def _hash_boundary(boundary: Mapping[str, Any]) -> str:
    norm = _normalize(boundary)
    s = json.dumps(norm, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


def _cache_backend(cache_dir: Optional[Path]) -> Optional[CacheBackend]:
    if cache_dir is None:
        return None
    try:
        return get_cache_backend(cache_dir)
    except Exception:
        return None


def forward(
    boundary: Mapping[str, Any],
    *,
    cache_dir: Optional[Path] = None,
    prefer_vmec: bool = False,
    use_real: Optional[bool] = None,
    problem: str = "p1",
) -> Dict[str, Any]:
    """Run the forward evaluator for a single boundary.

    Parameters
    - boundary: Boundary specification (JSON-like dict) using `SurfaceRZFourier` fields.

    Returns
    - Dict of metric names to values (numeric or informative non-numeric entries).

    This starter delegates to `constelx.physics.constel_api.evaluate_boundary` which
    provides lightweight, deterministic metrics. Replace with direct evaluator calls
    to compute physical figures of merit once available.
    """

    # Optional VMEC validation (best-effort) when requested.
    # When prefer_vmec=False, skip validation to avoid unnecessary import/time.
    if prefer_vmec:
        try:
            _ = boundary_to_vmec(boundary)
        except Exception:
            # Prefer VMEC validation, but fall back if unavailable
            pass
    # Optional cache lookup
    cache = _cache_backend(cache_dir)
    cache_key = _hash_boundary(boundary)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    # evaluate using real physics (if requested) or placeholder path
    if use_real is None:
        # Allow env toggle without changing call sites
        import os

        use_real = os.getenv("CONSTELX_USE_REAL_EVAL", "0").lower() in {"1", "true", "yes"}
    if use_real:
        try:
            from ..physics.proxima_eval import forward_metrics as px_forward

            result, _info = px_forward(dict(boundary), problem=problem)
        except Exception:
            result = evaluate_boundary(dict(boundary), use_real=False)
    else:
        result = evaluate_boundary(dict(boundary), use_real=False)
    if cache is not None:
        cache.set(cache_key, result)
    return result


def forward_many(
    boundaries: Iterable[Mapping[str, Any]],
    *,
    max_workers: int = 1,
    cache_dir: Optional[Path] = None,
    prefer_vmec: bool = False,
    use_real: Optional[bool] = None,
    problem: str = "p1",
) -> List[Dict[str, Any]]:
    items = list(boundaries)
    n = len(items)
    out: List[Optional[Dict[str, Any]]] = [None] * n

    keys: List[Optional[str]] = [None] * n
    to_compute: list[tuple[int, Mapping[str, Any]]] = []

    cache = _cache_backend(cache_dir)

    # Try cache
    for i, b in enumerate(items):
        # Optional VMEC validation (best-effort)
        if prefer_vmec:
            try:
                _ = boundary_to_vmec(b)
            except Exception:
                pass
        if cache is None:
            to_compute.append((i, b))
            continue
        k = _hash_boundary(b)
        keys[i] = k
        got = cache.get(k)
        if got is not None:
            out[i] = got
            continue
        to_compute.append((i, b))

    # Compute missing
    if to_compute:
        if use_real:
            try:
                from ..physics.proxima_eval import forward_metrics as px_forward

                for i, b in to_compute:
                    out[i] = px_forward(dict(b), problem=problem)[0]
            except Exception:
                for i, b in to_compute:
                    out[i] = evaluate_boundary(dict(b), use_real=False)
        elif max_workers <= 1:
            for i, b in to_compute:
                out[i] = evaluate_boundary(dict(b), use_real=False)
        else:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(evaluate_boundary, dict(b), False): i for i, b in to_compute}
                    for fut in as_completed(futs):
                        i = futs[fut]
                        out[i] = fut.result()
            except Exception:
                # Fallback to sequential if process pool is unavailable (e.g., sandboxed env)
                for i, b in to_compute:
                    out[i] = evaluate_boundary(dict(b), use_real=False)

    # Persist new caches
    if cache is not None:
        for i, r in enumerate(out):
            assert r is not None
            k = keys[i] or _hash_boundary(items[i])
            cache.set(k, r)

    # type narrowing
    return [v for v in out if v is not None]


def score(metrics: Mapping[str, Any], problem: Optional[str] = None) -> float:
    """Aggregate a scalar score from a metrics dict.

    Rules (deterministic and simple by design):
    - Consider only numeric (int/float) values.
    - If any considered value is NaN, return +inf (treat as invalid/bad).
    - Otherwise return the sum of numeric values (lower is better).

    This is a placeholder aggregation compatible with the starter's toy metrics.
    Swap in evaluator-default aggregation when integrating the real metrics.
    """

    # Use official scorer when available and a problem is provided
    if problem is not None:
        try:
            from ..physics.proxima_eval import score as px_score

            return float(px_score(problem, metrics))
        except Exception:
            pass

    # Prefer a single combined placeholder metric if available to avoid double-counting
    if "placeholder_metric" in metrics and isinstance(metrics["placeholder_metric"], (int, float)):
        v = float(metrics["placeholder_metric"])  # may be NaN
        return inf if isnan(v) else v

    total = 0.0
    for v in metrics.values():
        # Exclude booleans explicitly (bool is a subclass of int in Python).
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
            if isnan(fv):
                return inf
            total += fv
    return float(total)
