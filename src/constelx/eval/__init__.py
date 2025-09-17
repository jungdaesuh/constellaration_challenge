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

from ..physics.booz_proxy import compute_proxies
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
        # Allow TTL via environment without changing public signatures.
        ttl_env = os.getenv("CONSTELX_CACHE_TTL_SECONDS")
        ttl_val = int(ttl_env) if isinstance(ttl_env, str) and ttl_env.isdigit() else None
        return get_cache_backend(cache_dir, ttl_seconds=ttl_val)
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
            last_err = f"timeout_after_{int(deadline * 1000)}ms"
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
        try:
            to_store = dict(result)
            to_store.pop("elapsed_ms", None)
            cache.set(cache_key, to_store)
            # Return a result consistent with cached (strip non-deterministic timing)
            result = to_store
        except Exception:
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
    # Multi-fidelity gating (proxy -> select -> real)
    mf_proxy: bool = False,
    mf_threshold: float | None = None,
    mf_quantile: float | None = None,
    mf_max_high: int | None = None,
    mf_metric: str = "score",
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

    # Optional multi-fidelity proxy pass setup
    proxy_scores: dict[int, float] = {}
    proxy_metrics: dict[int, Dict[str, Any]] = {}

    proxy_value_keys = {"qs_residual", "qi_residual", "helical_energy", "mirror_ratio"}

    def _attach_proxy_fields(
        idx: int, metrics: Dict[str, Any], phase_override: str | None = None
    ) -> None:
        if not mf_proxy:
            return
        payload = proxy_metrics.get(idx)
        if payload:
            proxy_metric_val = payload.get("proxy_metric")
            if isinstance(proxy_metric_val, str) and "proxy_metric" not in metrics:
                metrics["proxy_metric"] = proxy_metric_val
            proxy_score_val = payload.get("proxy_score")
            if isinstance(proxy_score_val, (int, float)) and "proxy_score" not in metrics:
                metrics["proxy_score"] = float(proxy_score_val)
            for key in proxy_value_keys:
                if key in payload and key not in metrics:
                    metrics[key] = payload[key]
        if phase_override is not None:
            metrics.setdefault("phase", phase_override)

    # Compute missing
    if to_compute:
        # If requested, run a cheap proxy evaluation on all to_compute items to select survivors
        if mf_proxy:
            metric_key_norm = (mf_metric or "score").strip().lower()

            def _proxy_score_value(metrics: Mapping[str, Any]) -> float:
                if metric_key_norm == "score":
                    try:
                        return float(score(metrics, problem=None))
                    except Exception:
                        return float("inf")
                val = metrics.get(metric_key_norm)
                if isinstance(val, (int, float)):
                    return float(val)
                return float("inf")

            for i, b in to_compute:
                k = keys[i] or _hash_boundary(b)
                pm_key = f"{k}:proxy"
                cached_pm = cache.get(pm_key) if cache is not None else None
                if isinstance(cached_pm, dict):
                    m = dict(cached_pm)
                    m["phase"] = "proxy"
                else:
                    _t0p = time.perf_counter()
                    m = _placeholder_eval_task(dict(b))
                    _t1p = time.perf_counter()
                    try:
                        m.setdefault("elapsed_ms", (_t1p - _t0p) * 1000.0)
                    except Exception:
                        pass
                    m["phase"] = "proxy"
                    if cache is not None:
                        try:
                            to_store = dict(m)
                            to_store.pop("elapsed_ms", None)
                            cache.set(pm_key, to_store)
                        except Exception:
                            pass
                try:
                    proxies = compute_proxies(dict(b))
                    for key_proxy, val_proxy in proxies.as_dict().items():
                        m.setdefault(key_proxy, float(val_proxy))
                except Exception:
                    pass
                score_val = _proxy_score_value(m)
                m.setdefault("proxy_metric", metric_key_norm)
                m.setdefault("proxy_score", score_val)
                proxy_metrics[i] = m
                proxy_scores[i] = score_val

            idxs = list(proxy_scores.keys())
            survivors: set[int] = set()
            if mf_threshold is not None:
                thr = float(mf_threshold)
                for i in idxs:
                    if proxy_scores[i] <= thr:
                        survivors.add(i)
            else:
                q = 0.5 if mf_quantile is None else float(mf_quantile)
                q = min(max(q, 0.0), 1.0)
                sorted_idx = sorted(idxs, key=lambda i: proxy_scores[i])
                k_keep = int(round(q * len(sorted_idx)))
                if len(sorted_idx) > 0 and k_keep < 1:
                    k_keep = 1
                for i in sorted_idx[:k_keep]:
                    survivors.add(i)
            if mf_max_high is not None and len(survivors) > int(mf_max_high):
                best = sorted(list(survivors), key=lambda i: proxy_scores[i])[: int(mf_max_high)]
                survivors = set(best)

            non_survivors = {i for (i, _b) in to_compute} - survivors
            for i in non_survivors:
                if out[i] is None and i in proxy_metrics:
                    out[i] = proxy_metrics[i]
            to_compute = [(i, b) for (i, b) in to_compute if i in survivors]

        allow_parallel = os.getenv("CONSTELX_ALLOW_PARALLEL_REAL", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if use_real and max_workers > 1 and allow_parallel and to_compute:
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
                            r = fut.result(timeout=0)
                            if isinstance(r, dict):
                                # Trust worker provenance; only add scoring_version when real.
                                if r.get("source") == "real":
                                    sv = _scoring_version()
                                    if sv:
                                        r.setdefault("scoring_version", sv)
                            if isinstance(r, dict):
                                r.setdefault("phase", "real")
                                if mf_proxy:
                                    _attach_proxy_fields(i, r, phase_override=None)
                            out[i] = r
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
                                "phase": "real",
                            }
                    # Mark timed-out futures
                    for fut, i in futs.items():
                        if out[i] is None:
                            elapsed = time.perf_counter() - start_times.get(i, time.perf_counter())
                            if elapsed > timeout_s:
                                out[i] = {
                                    "feasible": False,
                                    "fail_reason": f"timeout_after_{int(timeout_s * 1000)}ms",
                                    "elapsed_ms": elapsed * 1000.0,
                                    "source": "real",
                                    "scoring_version": _scoring_version() or "",
                                }
                    # Fill any remaining by best-effort result() with small timeout to avoid hanging
                    for fut, i in futs.items():
                        if out[i] is None:
                            try:
                                r = fut.result(timeout=0.01)
                                if isinstance(r, dict) and r.get("source") == "real":
                                    sv = _scoring_version()
                                    if sv:
                                        r.setdefault("scoring_version", sv)
                                    r.setdefault("phase", "real")
                                out[i] = r
                            except Exception:
                                out[i] = {
                                    "feasible": False,
                                    "fail_reason": f"timeout_after_{int(timeout_s * 1000)}ms",
                                    "elapsed_ms": (
                                        time.perf_counter()
                                        - start_times.get(i, time.perf_counter())
                                    )
                                    * 1000.0,
                                    "source": "real",
                                    "scoring_version": _scoring_version() or "",
                                    "phase": "real",
                                }
            except Exception:
                # Fallback to sequential real-eval
                try:
                    from ..physics.proxima_eval import forward_metrics as px_forward

                    for i, b in to_compute:
                        _t0 = time.perf_counter()
                        _metrics_raw, info = px_forward(dict(b), problem=problem)
                        # Widen type to allow non-float annotations
                        metrics_any1: Dict[str, Any] = dict(_metrics_raw)
                        _t1 = time.perf_counter()
                        metrics_any1.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
                        if isinstance(info, dict):
                            feasible = bool(info.get("feasible", True))
                            metrics_any1.setdefault("feasible", feasible)
                            if not feasible and "fail_reason" not in metrics_any1:
                                fr = (
                                    info.get("reason")
                                    if isinstance(info.get("reason"), str)
                                    else ""
                                )
                                metrics_any1["fail_reason"] = fr
                        metrics_any1.setdefault("source", "real")
                        sv = _scoring_version()
                        if sv:
                            metrics_any1.setdefault("scoring_version", sv)
                        metrics_any1.setdefault("phase", "real")
                        if mf_proxy:
                            _attach_proxy_fields(i, metrics_any1, phase_override=None)
                        out[i] = metrics_any1
                except Exception:
                    for i, b in to_compute:
                        out[i] = evaluate_boundary(dict(b), use_real=False)
        elif use_real and to_compute:
            try:
                from ..physics.proxima_eval import forward_metrics as px_forward

                for i, b in to_compute:
                    _t0 = time.perf_counter()
                    _metrics_raw, info = px_forward(dict(b), problem=problem)
                    metrics_any2: Dict[str, Any] = dict(_metrics_raw)
                    _t1 = time.perf_counter()
                    metrics_any2.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
                    if isinstance(info, dict):
                        feasible = bool(info.get("feasible", True))
                        metrics_any2.setdefault("feasible", feasible)
                        if not feasible and "fail_reason" not in metrics_any2:
                            fr = info.get("reason") if isinstance(info.get("reason"), str) else ""
                            metrics_any2["fail_reason"] = fr
                    metrics_any2.setdefault("source", "real")
                    sv = _scoring_version()
                    if sv:
                        metrics_any2.setdefault("scoring_version", sv)
                    metrics_any2.setdefault("phase", "real")
                    if mf_proxy:
                        _attach_proxy_fields(i, metrics_any2, phase_override=None)
                    out[i] = metrics_any2
            except Exception:
                for i, b in to_compute:
                    out[i] = evaluate_boundary(dict(b), use_real=False)
        elif max_workers <= 1 and to_compute:
            for i, b in to_compute:
                _t0 = time.perf_counter()
                metrics = evaluate_boundary(dict(b), use_real=False)
                _t1 = time.perf_counter()
                try:
                    metrics.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
                    metrics.setdefault("feasible", True)
                    metrics.setdefault("fail_reason", "")
                    metrics.setdefault("source", "placeholder")
                except Exception:
                    pass
                if mf_proxy:
                    _attach_proxy_fields(i, metrics, phase_override="proxy")
                out[i] = metrics
        elif to_compute:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(_placeholder_eval_task, dict(b)): i for i, b in to_compute}
                    for fut in as_completed(futs):
                        i = futs[fut]
                        r = fut.result()
                        if isinstance(r, dict):
                            r.setdefault("source", "placeholder")
                            if mf_proxy:
                                _attach_proxy_fields(futs[fut], r, phase_override="proxy")
                        out[i] = r
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
                        metrics.setdefault("source", "placeholder")
                    except Exception:
                        pass
                    if mf_proxy:
                        _attach_proxy_fields(i, metrics, phase_override="proxy")
                    out[i] = metrics

    # Strip non-deterministic timing before caching/returning to keep cache equality stable
    if cache is not None:
        for i, rec in enumerate(out):
            assert rec is not None
            row = rec
            row.pop("elapsed_ms", None)
            k = keys[i] or _hash_boundary(items[i])
            # Separate namespaces for proxy vs real results when MF is enabled
            if row.get("phase") == "proxy":
                cache.set(f"{k}:proxy", row)
            else:
                cache.set(k, row)

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

    # If evaluator flagged failure, return +inf deterministically
    try:
        feas = metrics.get("feasible")
        if isinstance(feas, bool) and feas is False:
            return inf
        fr = metrics.get("fail_reason")
        if isinstance(fr, str) and fr:
            return inf
    except Exception:
        pass

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
        _metrics_raw, info = px_forward(b, problem=prob)
        metrics: Dict[str, Any] = dict(_metrics_raw)
        _t1 = time.perf_counter()
        metrics.setdefault("elapsed_ms", (_t1 - _t0) * 1000.0)
        metrics["source"] = "real"
        if isinstance(info, dict):
            feasible = bool(info.get("feasible", True))
            metrics.setdefault("feasible", feasible)
            if not feasible and "fail_reason" not in metrics:
                fr = info.get("reason") if isinstance(info.get("reason"), str) else ""
                metrics["fail_reason"] = fr
        sv = _scoring_version()
        if sv:
            metrics.setdefault("scoring_version", sv)
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
            metrics["source"] = "placeholder"
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
        metrics.setdefault("source", "placeholder")
    except Exception:
        pass
    return metrics
