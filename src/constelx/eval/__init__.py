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


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def forward(boundary: Mapping[str, Any], *, cache_dir: Optional[Path] = None) -> Dict[str, Any]:
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
    # Validate with VMEC model if available; otherwise fall back to dict-based evaluation
    try:
        _ = boundary_to_vmec(boundary)
    except Exception:
        pass
    # Optional cache lookup
    cache_key = None
    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _hash_boundary(boundary)
        cache_file = _cache_path(cache_dir, cache_key)
        if cache_file.exists():
            try:
                return cast(Dict[str, Any], json.loads(cache_file.read_text()))
            except Exception:
                pass
    # evaluate_boundary expects a plain dict
    result = evaluate_boundary(dict(boundary))
    if cache_file is not None:
        try:
            cache_file.write_text(json.dumps(result))
        except Exception:
            pass
    return result


def forward_many(
    boundaries: Iterable[Mapping[str, Any]],
    *,
    max_workers: int = 1,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    items = list(boundaries)
    n = len(items)
    out: List[Optional[Dict[str, Any]]] = [None] * n

    keys: List[Optional[str]] = [None] * n
    paths: List[Optional[Path]] = [None] * n
    to_compute: list[tuple[int, Mapping[str, Any]]] = []

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Try cache
    for i, b in enumerate(items):
        if cache_dir is None:
            to_compute.append((i, b))
            continue
        k = _hash_boundary(b)
        p = _cache_path(cache_dir, k)
        keys[i] = k
        paths[i] = p
        if p.exists():
            try:
                out[i] = json.loads(p.read_text())
                continue
            except Exception:
                pass
        to_compute.append((i, b))

    # Compute missing
    if to_compute:
        if max_workers <= 1:
            for i, b in to_compute:
                out[i] = evaluate_boundary(dict(b))
        else:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(evaluate_boundary, dict(b)): i for i, b in to_compute}
                    for fut in as_completed(futs):
                        i = futs[fut]
                        out[i] = fut.result()
            except Exception:
                # Fallback to sequential if process pool is unavailable (e.g., sandboxed env)
                for i, b in to_compute:
                    out[i] = evaluate_boundary(dict(b))

    # Persist new caches
    if cache_dir is not None:
        for i, r in enumerate(out):
            assert r is not None
            p_existing = paths[i]
            if p_existing is not None:
                p_final = p_existing
            else:
                k = _hash_boundary(items[i])
                p_final = _cache_path(cache_dir, k)
            if not p_final.exists():
                try:
                    p_final.write_text(json.dumps(r))
                except Exception:
                    pass

    # type narrowing
    return [v for v in out if v is not None]


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
