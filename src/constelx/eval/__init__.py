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

# Load environment variables early if python-dotenv is available so that
# evaluator settings (e.g., timeouts) can be configured via a local .env.
try:  # optional dependency; safe no-op if missing
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    pass

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
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


def _timeout_config() -> tuple[float, int, float]:
    """Return (timeout_seconds, retries, backoff) from env with defaults.

    Env vars:
    - CONSTELX_REAL_TIMEOUT_MS: per-call timeout in ms (default: 20000)
    - CONSTELX_REAL_RETRIES: number of retries on failure/timeout (default: 1)
    - CONSTELX_REAL_BACKOFF: multiplicative backoff factor (default: 1.5)
    """
    try:
        t_ms = float(os.getenv("CONSTELX_REAL_TIMEOUT_MS", "20000"))
    except Exception:
        t_ms = 20000.0
    try:
        retries = int(os.getenv("CONSTELX_REAL_RETRIES", "1"))
    except Exception:
        retries = 1
    try:
        backoff = float(os.getenv("CONSTELX_REAL_BACKOFF", "1.5"))
    except Exception:
        backoff = 1.5
    return max(0.0, t_ms / 1000.0), max(0, retries), max(1.0, backoff)


def _scoring_version() -> str:
    try:
        import importlib.metadata as _im

        return _im.version("constellaration")
    except Exception:
        return ""


def _real_eval_with_timeout(boundary: Mapping[str, Any], problem: str) -> Dict[str, Any]:
    """Call the real evaluator in a separate process with timeout/retries.

    Returns a metrics dict annotated with elapsed_ms/feasible/fail_reason and provenance.
    """
    timeout_s, retries, backoff = _timeout_config()
    attempt = 0
    last_err: str = ""
    t0_all = time.perf_counter()
    while attempt <= retries:
        attempt += 1
        deadline = timeout_s * (backoff ** (attempt - 1))
        try:
            with ProcessPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_real_eval_task, (dict(boundary), problem))
                metrics = fut.result(timeout=deadline)
                # Provenance and versioning
                try:
                    metrics.setdefault("source", "real")
                    sv = _scoring_version()
                    if sv:
                        metrics.setdefault("scoring_version", sv)
                except Exception:
                    pass
                return metrics
        except TimeoutError:
            last_err = f"timeout_after_{int(deadline*1000)}ms"
        except Exception as e:  # pragma: no cover - depends on external evaluator
            last_err = f"error:{type(e).__name__}"
        # retry loop continues
    # Failure payload
    t1_all = time.perf_counter()
    return {
        "feasible": False,
        "fail_reason": last_err or "timeout",
        "elapsed_ms": (t1_all - t0_all) * 1000.0,
        "source": "real",
        "scoring_version": _scoring_version() or "",
    }


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
    t0 = time.perf_counter()
    if use_real is None:
        # Allow env toggle without changing call sites
        import os

        use_real = os.getenv("CONSTELX_USE_REAL_EVAL", "0").lower() in {"1", "true", "yes"}
    if use_real:
        try:
            result = _real_eval_with_timeout(boundary, problem)
        except Exception:
            result = evaluate_boundary(dict(boundary), use_real=False)
    else:
        result = evaluate_boundary(dict(boundary), use_real=False)
    t1 = time.perf_counter()
    # annotate timing and defaults
    try:
        result.setdefault("elapsed_ms", (t1 - t0) * 1000.0)
        result.setdefault("feasible", True)
        result.setdefault("fail_reason", "")
        result.setdefault("source", "real" if use_real else "placeholder")
        if use_real:
            sv = _scoring_version()
            if sv:
                result.setdefault("scoring_version", sv)
    except Exception:
        pass
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
        allow_parallel = os.getenv("CONSTELX_ALLOW_PARALLEL_REAL", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if use_real and max_workers > 1 and allow_parallel:
            # Opt-in parallel execution for real evaluator. Configure BLAS/OpenMP threads
            # per worker to avoid oversubscription on many-core Macs.
            try:
                from ..physics.proxima_eval import forward_metrics as px_forward

                # Compute a conservative OMP thread count per worker
                try:
                    cpu_total = multiprocessing.cpu_count()
                except Exception:
                    cpu_total = 8
                threads_per = max(1, min(5, cpu_total // max(1, max_workers)))
                for var in (
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    os.environ[var] = str(threads_per)
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    start_times: dict[int, float] = {}
                    futs = {}
                    timeout_s, _retries, _backoff = _timeout_config()
                    for i, b in to_compute:
                        fut = ex.submit(_real_eval_task, (dict(b), problem))
                        futs[fut] = i
                        start_times[i] = time.perf_counter()
                    for fut in as_completed(futs, timeout=None):
                        i = futs[fut]
                        try:
                            out[i] = fut.result(timeout=0)
                        except Exception:
                            out[i] = {
                                "feasible": False,
                                "fail_reason": "worker_error",
                                "elapsed_ms": (
                                    time.perf_counter() - start_times.get(i, time.perf_counter())
                                )
                                * 1000.0,
                                "source": "real",
                                "scoring_version": _scoring_version() or "",
                            }
                    # Mark timed-out futures
                    for fut, i in futs.items():
                        if out[i] is None:
                            elapsed = time.perf_counter() - start_times.get(i, time.perf_counter())
                            if elapsed > timeout_s:
                                out[i] = {
                                    "feasible": False,
                                    "fail_reason": f"timeout_after_{int(timeout_s*1000)}ms",
                                    "elapsed_ms": elapsed * 1000.0,
                                    "source": "real",
                                    "scoring_version": _scoring_version() or "",
                                }
                    # Fill any remaining by best-effort result() with small timeout to avoid hanging
                    for fut, i in futs.items():
                        if out[i] is None:
                            try:
                                out[i] = fut.result(timeout=0.01)
                            except Exception:
                                out[i] = {
                                    "feasible": False,
                                    "fail_reason": f"timeout_after_{int(timeout_s*1000)}ms",
                                    "elapsed_ms": (
                                        time.perf_counter()
                                        - start_times.get(i, time.perf_counter())
                                    )
                                    * 1000.0,
                                    "source": "real",
                                    "scoring_version": _scoring_version() or "",
                                }
            except Exception:
                # Fallback to sequential real-eval
                try:
                    from ..physics.proxima_eval import forward_metrics as px_forward

                    for i, b in to_compute:
                        _t0 = time.perf_counter()
                        metrics, info = px_forward(dict(b), problem=problem)
                        _t1 = time.perf_counter()
                        metrics.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
                        if isinstance(info, dict):
                            feasible = bool(info.get("feasible", True))
                            metrics.setdefault("feasible", feasible)
                            if not feasible and "fail_reason" not in metrics:
                                fr = (
                                    info.get("reason")
                                    if isinstance(info.get("reason"), str)
                                    else ""
                                )
                                metrics["fail_reason"] = fr
                        out[i] = metrics
                except Exception:
                    for i, b in to_compute:
                        out[i] = evaluate_boundary(dict(b), use_real=False)
        elif use_real:
            try:
                from ..physics.proxima_eval import forward_metrics as px_forward

                for i, b in to_compute:
                    _t0 = time.perf_counter()
                    metrics, info = px_forward(dict(b), problem=problem)
                    _t1 = time.perf_counter()
                    metrics.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
                    if isinstance(info, dict):
                        feasible = bool(info.get("feasible", True))
                        metrics.setdefault("feasible", feasible)
                        if not feasible and "fail_reason" not in metrics:
                            fr = info.get("reason") if isinstance(info.get("reason"), str) else ""
                            metrics["fail_reason"] = fr
                    out[i] = metrics
            except Exception:
                for i, b in to_compute:
                    out[i] = evaluate_boundary(dict(b), use_real=False)
        elif max_workers <= 1:
            for i, b in to_compute:
                _t0 = time.perf_counter()
                metrics = evaluate_boundary(dict(b), use_real=False)
                _t1 = time.perf_counter()
                try:
                    metrics.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
                    metrics.setdefault("feasible", True)
                    metrics.setdefault("fail_reason", "")
                except Exception:
                    pass
                out[i] = metrics
        else:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(_placeholder_eval_task, dict(b)): i for i, b in to_compute}
                    for fut in as_completed(futs):
                        i = futs[fut]
                        out[i] = fut.result()
            except Exception:
                # Fallback to sequential if process pool is unavailable (e.g., sandboxed env)
                for i, b in to_compute:
                    _t0 = time.perf_counter()
                    metrics = evaluate_boundary(dict(b), use_real=False)
                    _t1 = time.perf_counter()
                    try:
                        metrics.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
                        metrics.setdefault("feasible", True)
                        metrics.setdefault("fail_reason", "")
                    except Exception:
                        pass
                    out[i] = metrics

    # Strip non-deterministic timing before caching/returning to keep cache equality stable
    if cache is not None:
        for i, r in enumerate(out):
            assert r is not None
            r.pop("elapsed_ms", None)
            k = keys[i] or _hash_boundary(items[i])
            cache.set(k, r)

    # type narrowing
    # Remove elapsed_ms to make results deterministic for equality checks
    return [{k: v.get(k) for k in v.keys() if k != "elapsed_ms"} for v in out if v is not None]


def score(metrics: Mapping[str, Any], problem: Optional[str] = None) -> float:
    """Aggregate a scalar score from a metrics dict.

    Rules (deterministic and simple by design):
    - Consider only numeric (int/float) values.
    - If any considered value is NaN, return +inf (treat as invalid/bad).
    - Otherwise return the sum of numeric values (lower is better).

    This is a placeholder aggregation compatible with the starter's toy metrics.
    Swap in evaluator-default aggregation when integrating the real metrics.
    """

    # If the metrics already contain an authoritative 'score' (e.g., from the
    # official evaluator), return it directly to avoid recomputation/mismatch.
    if "score" in metrics and isinstance(metrics["score"], (int, float)):
        sv = float(metrics["score"])  # may be NaN or inf depending on evaluator
        return inf if isnan(sv) else sv

    # Use official scorer when available and a problem is provided
    if problem is not None:
        try:
            from ..physics.proxima_eval import score as px_score

            return float(px_score(problem, metrics))
        except Exception:
            pass
    # If the metrics already contain an authoritative 'score', use it.
    if "score" in metrics and isinstance(metrics["score"], (int, float)):
        sv = float(metrics["score"])  # may be NaN or inf depending on evaluator
        return inf if isnan(sv) else sv

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


def _real_eval_task(args: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Helper for parallel real-evaluator calls.

    Accepts (boundary, problem) and returns metrics dict.
    """
    b, prob = args
    try:
        from ..physics.proxima_eval import forward_metrics as px_forward

        _t0 = time.perf_counter()
        metrics, info = px_forward(b, problem=prob)
        _t1 = time.perf_counter()
        metrics.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
        if isinstance(info, dict):
            feasible = bool(info.get("feasible", True))
            metrics.setdefault("feasible", feasible)
            if not feasible and "fail_reason" not in metrics:
                fr = info.get("reason") if isinstance(info.get("reason"), str) else ""
                metrics["fail_reason"] = fr
        return metrics
    except Exception:
        # Fallback to placeholder if real path unavailable in worker
        _t0 = time.perf_counter()
        metrics = evaluate_boundary(b, use_real=False)
        _t1 = time.perf_counter()
        try:
            metrics.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
            metrics.setdefault("feasible", True)
            metrics.setdefault("fail_reason", "")
        except Exception:
            pass
        return metrics


def _placeholder_eval_task(b: dict[str, Any]) -> dict[str, Any]:
    """Helper for parallel placeholder evaluator calls with timing."""
    _t0 = time.perf_counter()
    metrics = evaluate_boundary(b, use_real=False)
    _t1 = time.perf_counter()
    try:
        metrics.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
        metrics.setdefault("feasible", True)
        metrics.setdefault("fail_reason", "")
    except Exception:
        pass
    return metrics
