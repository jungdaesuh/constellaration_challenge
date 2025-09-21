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
import re
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from math import inf, isinf, isnan
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, TypeAlias, cast

from ..optim.pareto import DEFAULT_P3_SCALARIZATION, extract_objectives, scalarize
from ..physics.booz_proxy import BOOZER_PROXY_KEYS, compute_proxies
from ..physics.constel_api import evaluate_boundary
from .cache import CacheBackend, get_cache_backend

# A simple local type alias; keep as Any to avoid leaking third-party types.
VmecBoundary: TypeAlias = Any


_VMEC_LEVEL_ALIASES: Dict[str, str] = {
    "low": "low",
    "lo": "low",
    "fast": "low",
    "medium": "medium",
    "med": "medium",
    "mid": "medium",
    "default": "auto",
    "auto": "auto",
    "high": "high",
    "hi": "high",
    "full": "high",
}


MF_PROXY_METRICS: tuple[str, ...] = (
    "score",
    "placeholder_metric",
    *BOOZER_PROXY_KEYS,
)


def _normalize_vmec_level(value: Optional[str]) -> str:
    if value is None:
        return "auto"
    normalized = value.strip().lower()
    if not normalized:
        return "auto"
    return _VMEC_LEVEL_ALIASES.get(
        normalized, normalized if normalized in {"low", "medium", "high", "auto"} else "auto"
    )


def _env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return None


def _resolve_vmec_opts(
    vmec_level: Optional[str],
    vmec_hot_restart: Optional[bool],
    vmec_restart_key: Optional[str],
) -> Dict[str, Any]:
    level = _normalize_vmec_level(vmec_level or os.getenv("CONSTELX_VMEC_LEVEL"))
    hot_restart = vmec_hot_restart
    if hot_restart is None:
        env_val = _env_bool("CONSTELX_VMEC_HOT_RESTART")
        if env_val is not None:
            hot_restart = env_val
    if hot_restart is None:
        hot_restart = False
    restart_key = vmec_restart_key or os.getenv("CONSTELX_VMEC_RESTART_KEY")
    restart_key = (
        restart_key.strip() if isinstance(restart_key, str) and restart_key.strip() else None
    )
    return {
        "level": level,
        "hot_restart": bool(hot_restart),
        "restart_key": restart_key,
    }


def _combine_cache_key(base: str, vmec_opts: Mapping[str, Any]) -> str:
    parts: List[str] = []
    level = vmec_opts.get("level")
    if isinstance(level, str) and level:
        parts.append(f"lvl={level}")
    if vmec_opts.get("hot_restart"):
        parts.append("hot=1")
    restart_key = vmec_opts.get("restart_key")
    if isinstance(restart_key, str) and restart_key:
        parts.append(f"rk={restart_key}")
    if not parts:
        return base
    return f"{base}|{'|'.join(parts)}"


def _annotate_vmec_metadata(
    metrics: Dict[str, Any], vmec_opts: Mapping[str, Any]
) -> Dict[str, Any]:
    try:
        metrics.setdefault("vmec_level", vmec_opts.get("level", "auto"))
        metrics.setdefault("vmec_hot_restart", bool(vmec_opts.get("hot_restart", False)))
        restart_key = vmec_opts.get("restart_key")
        if isinstance(restart_key, str) and restart_key:
            metrics.setdefault("vmec_restart_key", restart_key)
    except Exception:
        pass
    return metrics


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


_LABEL_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def _json_ready(obj: Any) -> Any:
    """Return a JSON-serializable representation with stable ordering.

    Converts numpy scalars/arrays via `_normalize` and replaces NaN/inf floats
    with string sentinels to keep log files valid JSON.
    """

    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize(v) for v in value]
        if isinstance(value, float):
            if isnan(value):
                return "NaN"
            if isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
        return value

    return _sanitize(_normalize(obj))


def _sanitize_label(value: Optional[str], fallback: str = "") -> str:
    """Return a filesystem-safe label consisting of lowercase [a-z0-9-]."""

    if not isinstance(value, str):
        return fallback
    lowered = value.strip().lower()
    if not lowered:
        return fallback
    cleaned = _LABEL_SANITIZE_RE.sub("-", lowered).strip("-")
    return cleaned or fallback


def _log_eval_event(
    boundary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    problem: Optional[str],
    vmec_opts: Mapping[str, Any],
    cache_hit: bool,
) -> None:
    """Persist a JSON log of evaluator inputs/outputs when enabled."""

    log_dir_raw = os.getenv("CONSTELX_EVAL_LOG_DIR")
    if not log_dir_raw or cache_hit:
        return

    try:
        log_dir = Path(log_dir_raw).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        problem_label = _sanitize_label(problem, fallback="unknown")
        phase_value = metrics.get("phase") if isinstance(metrics, Mapping) else None
        phase_label_raw = str(phase_value) if isinstance(phase_value, str) and phase_value else None
        phase_label = _sanitize_label(phase_label_raw, fallback="")
        boundary_fingerprint = _hash_boundary(boundary)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        suffix_parts: List[str] = []
        if phase_label:
            suffix_parts.append(phase_label)
        suffix_parts.append("eval")
        suffix = "_".join(suffix_parts)
        filename = (
            f"{problem_label}_{suffix}_{timestamp}_{boundary_fingerprint[:12]}_"
            f"{uuid.uuid4().hex[:8]}.json"
        )
        target_path = log_dir / filename
        resolved_dir = log_dir.resolve()
        resolved_target = target_path.resolve()
        try:
            if not resolved_target.is_relative_to(resolved_dir):
                return
        except AttributeError:  # Python <3.9 fallback (not expected but defensive)
            target_str = str(resolved_target)
            if not target_str.startswith(str(resolved_dir)):
                return
        payload = {
            "timestamp": time.time(),
            "problem": problem,
            "cache_hit": cache_hit,
            "phase": phase_label,
            "vmec_opts": _json_ready(dict(vmec_opts)),
            "boundary_hash": boundary_fingerprint,
            "boundary": _json_ready(dict(boundary)),
            "metrics": _json_ready(dict(metrics)),
        }
        path = log_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        # Logging should never interrupt evaluation; swallow best-effort failures.
        pass


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


def _real_eval_with_timeout(
    boundary: Mapping[str, Any], problem: str, vmec_opts: Mapping[str, Any]
) -> Dict[str, Any]:
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
                fut = ex.submit(_real_eval_task, (dict(boundary), problem, dict(vmec_opts)))
                metrics = fut.result(timeout=deadline)
                # Provenance and versioning
                try:
                    metrics.setdefault("source", "real")
                    sv = _scoring_version()
                    if sv:
                        metrics.setdefault("scoring_version", sv)
                except Exception:
                    pass
                return _annotate_vmec_metadata(metrics, vmec_opts)
        except TimeoutError:
            last_err = f"timeout_after_{int(deadline * 1000)}ms"
        except Exception as e:  # pragma: no cover - depends on external evaluator
            last_err = f"error:{type(e).__name__}"
        # retry loop continues
    # Failure payload
    t1_all = time.perf_counter()
    failure = {
        "feasible": False,
        "fail_reason": last_err or "timeout",
        "elapsed_ms": (t1_all - t0_all) * 1000.0,
        "source": "real",
        "scoring_version": _scoring_version() or "",
    }
    return _annotate_vmec_metadata(failure, vmec_opts)


def forward(
    boundary: Mapping[str, Any],
    *,
    cache_dir: Optional[Path] = None,
    prefer_vmec: bool = False,
    use_real: Optional[bool] = None,
    problem: str = "p1",
    vmec_level: Optional[str] = None,
    vmec_hot_restart: Optional[bool] = None,
    vmec_restart_key: Optional[str] = None,
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
    vmec_opts = _resolve_vmec_opts(vmec_level, vmec_hot_restart, vmec_restart_key)

    cache = _cache_backend(cache_dir)
    cache_key = _combine_cache_key(_hash_boundary(boundary), vmec_opts)
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            reason_cached = str(cached.get("fail_reason") or "").strip()
            feasible_cached = cached.get("feasible")
            if reason_cached or feasible_cached is False:
                cached = None
        if cached is not None:
            if isinstance(cached, dict):
                return _annotate_vmec_metadata(dict(cached), vmec_opts)
            return cached
    # evaluate using real physics (if requested) or placeholder path
    t0 = time.perf_counter()
    if use_real is None:
        # Allow env toggle without changing call sites
        import os

        use_real = os.getenv("CONSTELX_USE_REAL_EVAL", "0").lower() in {"1", "true", "yes"}
    if use_real:
        try:
            result = _real_eval_with_timeout(boundary, problem, vmec_opts)
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
    # Enrich with canonical metrics (non-destructive; adds proxies/geometry if missing)
    try:
        from ..physics.metrics import enrich as _enrich_metrics

        result = _enrich_metrics(result, boundary)
    except Exception:
        pass

    result = _annotate_vmec_metadata(result, vmec_opts)

    _log_eval_event(boundary, result, problem=problem, vmec_opts=vmec_opts, cache_hit=False)

    if cache is not None:
        reason_val = str(result.get("fail_reason") or "").strip()
        feasible_val = result.get("feasible")
        should_cache = not reason_val and feasible_val is not False
        if should_cache:
            try:
                to_store = dict(result)
                to_store.pop("elapsed_ms", None)
                cache.set(cache_key, to_store)
                # Return a result consistent with cached (strip non-deterministic timing)
                result = dict(to_store)
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
    vmec_level: Optional[str] = None,
    vmec_hot_restart: Optional[bool] = None,
    vmec_restart_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    items = list(boundaries)
    n = len(items)
    out: List[Optional[Dict[str, Any]]] = [None] * n
    cache_hits: List[bool] = [False] * n

    keys: List[Optional[str]] = [None] * n
    to_compute: list[tuple[int, Mapping[str, Any]]] = []

    vmec_opts = _resolve_vmec_opts(vmec_level, vmec_hot_restart, vmec_restart_key)

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
        k = _combine_cache_key(_hash_boundary(b), vmec_opts)
        keys[i] = k
        got = cache.get(k)
        if isinstance(got, dict):
            reason_cached = str(got.get("fail_reason") or "").strip()
            feasible_cached = got.get("feasible")
            if reason_cached or feasible_cached is False:
                got = None
        if got is not None:
            out[i] = _annotate_vmec_metadata(
                dict(got) if isinstance(got, dict) else got,
                vmec_opts,
            )
            cache_hits[i] = True
            continue
        to_compute.append((i, b))

    # Optional multi-fidelity proxy pass setup
    proxy_scores: dict[int, float] = {}
    proxy_metrics: dict[int, Dict[str, Any]] = {}
    proxy_cache_hits: set[int] = set()

    proxy_value_keys = set(BOOZER_PROXY_KEYS)

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
            allowed_metrics = {m.strip().lower() for m in MF_PROXY_METRICS}
            if metric_key_norm not in allowed_metrics:
                supported = ", ".join(sorted(allowed_metrics))
                raise ValueError("mf_proxy_metric must be one of: " + supported)

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
                k = keys[i] or _combine_cache_key(_hash_boundary(b), vmec_opts)
                pm_key = f"{k}:proxy"
                cached_pm = cache.get(pm_key) if cache is not None else None
                if isinstance(cached_pm, dict):
                    m_raw: Mapping[str, Any] | None = dict(cached_pm)
                    proxy_cache_hits.add(i)
                else:
                    _t0p = time.perf_counter()
                    try:
                        from ..physics import metrics as _metrics_mod

                        m_raw = _metrics_mod.compute(
                            dict(b),
                            use_real=False,
                            attach_proxies=True,
                        )
                    except Exception:
                        m_raw = _placeholder_eval_task(dict(b), vmec_opts)
                        try:
                            from ..physics.metrics import enrich as _enrich_metrics

                            m_raw = _enrich_metrics(m_raw, b)
                        except Exception:
                            pass
                    _t1p = time.perf_counter()
                    if isinstance(m_raw, dict) and "elapsed_ms" not in m_raw:
                        try:
                            m_raw = dict(m_raw)
                            m_raw.setdefault("elapsed_ms", (_t1p - _t0p) * 1000.0)
                        except Exception:
                            pass
                m = dict(m_raw) if isinstance(m_raw, Mapping) else {}
                m.setdefault("feasible", True)
                m.setdefault("fail_reason", "")
                m.setdefault("source", "placeholder")
                m = _annotate_vmec_metadata(m, vmec_opts)
                m.setdefault("phase", "proxy")
                if not all(key in m for key in BOOZER_PROXY_KEYS):
                    try:
                        proxies = compute_proxies(dict(b))
                        for key_proxy, val_proxy in proxies.as_dict().items():
                            m.setdefault(key_proxy, float(val_proxy))
                    except Exception:
                        pass
                score_val = _proxy_score_value(m)
                m["proxy_metric"] = metric_key_norm
                m["proxy_score"] = score_val
                proxy_metrics[i] = dict(m)
                proxy_scores[i] = score_val
                if cache is not None:
                    try:
                        to_store = dict(m)
                        to_store.pop("elapsed_ms", None)
                        cache.set(pm_key, to_store)
                    except Exception:
                        pass

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
                    if i in proxy_cache_hits:
                        cache_hits[i] = True
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
                        fut = ex.submit(_real_eval_task, (dict(b), problem, vmec_opts))
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
                                r.setdefault("phase", "real")
                                if mf_proxy:
                                    _attach_proxy_fields(i, r, phase_override=None)
                                r = _annotate_vmec_metadata(r, vmec_opts)
                            if isinstance(r, dict):
                                # Enrich before storing
                                try:
                                    from ..physics.metrics import enrich as _enrich_metrics

                                    r = _enrich_metrics(r, items[i])
                                except Exception:
                                    pass
                            out[i] = r
                        except Exception:
                            out[i] = _annotate_vmec_metadata(
                                {
                                    "feasible": False,
                                    "fail_reason": "worker_error",
                                    "elapsed_ms": (
                                        time.perf_counter()
                                        - start_times.get(i, time.perf_counter())
                                    )
                                    * 1000.0,
                                    "source": "real",
                                    "scoring_version": _scoring_version() or "",
                                    "phase": "real",
                                },
                                vmec_opts,
                            )
                    # Mark timed-out futures
                    for fut, i in futs.items():
                        if out[i] is None:
                            elapsed = time.perf_counter() - start_times.get(i, time.perf_counter())
                            if elapsed > timeout_s:
                                fail_payload = {
                                    "feasible": False,
                                    "fail_reason": f"timeout_after_{int(timeout_s * 1000)}ms",
                                    "elapsed_ms": elapsed * 1000.0,
                                    "source": "real",
                                    "scoring_version": _scoring_version() or "",
                                }
                                out[i] = _annotate_vmec_metadata(fail_payload, vmec_opts)
                    # Fill any remaining by best-effort result() with small timeout to avoid hanging
                    for fut, i in futs.items():
                        if out[i] is None:
                            try:
                                r = fut.result(timeout=0.01)
                                if isinstance(r, dict) and r.get("source") == "real":
                                    sv = _scoring_version()
                                    if sv:
                                        r.setdefault("scoring_version", sv)
                                if isinstance(r, dict):
                                    r.setdefault("phase", "real")
                                    if mf_proxy:
                                        _attach_proxy_fields(i, r, phase_override=None)
                                    r = _annotate_vmec_metadata(r, vmec_opts)
                                if isinstance(r, dict):
                                    try:
                                        from ..physics.metrics import enrich as _enrich_metrics

                                        r = _enrich_metrics(r, items[i])
                                    except Exception:
                                        pass
                                out[i] = r
                            except Exception:
                                elapsed = time.perf_counter() - start_times.get(
                                    i, time.perf_counter()
                                )
                                fail_payload = {
                                    "feasible": False,
                                    "fail_reason": f"timeout_after_{int(timeout_s * 1000)}ms",
                                    "elapsed_ms": elapsed * 1000.0,
                                    "source": "real",
                                    "scoring_version": _scoring_version() or "",
                                    "phase": "real",
                                }
                                out[i] = _annotate_vmec_metadata(fail_payload, vmec_opts)
            except Exception:
                # Fallback to sequential real-eval
                try:
                    from ..physics.proxima_eval import forward_metrics as px_forward

                    for i, b in to_compute:
                        _t0 = time.perf_counter()
                        _metrics_raw, info = px_forward(
                            dict(b), problem=problem, vmec_opts=vmec_opts
                        )
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
                        metrics_any1 = _annotate_vmec_metadata(metrics_any1, vmec_opts)
                        try:
                            from ..physics.metrics import enrich as _enrich_metrics

                            metrics_any1 = _enrich_metrics(metrics_any1, items[i])
                        except Exception:
                            pass
                        out[i] = metrics_any1
                except Exception:
                    for i, b in to_compute:
                        placeholder = evaluate_boundary(dict(b), use_real=False)
                        if isinstance(placeholder, dict):
                            placeholder = _annotate_vmec_metadata(placeholder, vmec_opts)
                        out[i] = placeholder
        elif use_real and to_compute:
            try:
                from ..physics.proxima_eval import forward_metrics as px_forward

                for i, b in to_compute:
                    _t0 = time.perf_counter()
                    _metrics_raw, info = px_forward(dict(b), problem=problem, vmec_opts=vmec_opts)
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
                    metrics_any2 = _annotate_vmec_metadata(metrics_any2, vmec_opts)
                    try:
                        from ..physics.metrics import enrich as _enrich_metrics

                        metrics_any2 = _enrich_metrics(metrics_any2, items[i])
                    except Exception:
                        pass
                    out[i] = metrics_any2
            except Exception:
                for i, b in to_compute:
                    r = evaluate_boundary(dict(b), use_real=False)
                    if isinstance(r, dict):
                        try:
                            from ..physics.metrics import enrich as _enrich_metrics

                            r = _enrich_metrics(r, items[i])
                        except Exception:
                            pass
                        r = _annotate_vmec_metadata(r, vmec_opts)
                    out[i] = r
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
                metrics = _annotate_vmec_metadata(metrics, vmec_opts)
                # Enrich results before storing
                try:
                    from ..physics.metrics import enrich as _enrich_metrics

                    metrics = _enrich_metrics(metrics, items[i])
                except Exception:
                    pass
                out[i] = metrics
        elif to_compute:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futs = {
                        ex.submit(_placeholder_eval_task, dict(b), vmec_opts): i
                        for i, b in to_compute
                    }
                    for fut in as_completed(futs):
                        i = futs[fut]
                        r = fut.result()
                        if isinstance(r, dict):
                            r.setdefault("source", "placeholder")
                            if mf_proxy:
                                _attach_proxy_fields(futs[fut], r, phase_override="proxy")
                            r = _annotate_vmec_metadata(r, vmec_opts)
                        if isinstance(r, dict):
                            try:
                                from ..physics.metrics import enrich as _enrich_metrics

                                r = _enrich_metrics(r, items[i])
                            except Exception:
                                pass
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
                    metrics = _annotate_vmec_metadata(metrics, vmec_opts)
                    # Enrich results before storing
                    try:
                        from ..physics.metrics import enrich as _enrich_metrics

                        metrics = _enrich_metrics(metrics, items[i])
                    except Exception:
                        pass
                    out[i] = metrics

    # Persist logs (best-effort) before mutating metrics for cache storage
    for i, rec in enumerate(out):
        if rec is None:
            continue
        try:
            boundary_payload = items[i]
        except IndexError:
            boundary_payload = {}
        _log_eval_event(
            boundary_payload,
            dict(rec),
            problem=problem,
            vmec_opts=vmec_opts,
            cache_hit=cache_hits[i],
        )

    # Strip non-deterministic timing before caching/returning to keep cache equality stable
    if cache is not None:
        for i, rec in enumerate(out):
            assert rec is not None
            row = rec
            row.pop("elapsed_ms", None)
            k = keys[i] or _combine_cache_key(_hash_boundary(items[i]), vmec_opts)
            # Separate namespaces for proxy vs real results when MF is enabled
            if row.get("phase") == "proxy":
                cache.set(f"{k}:proxy", row)
                continue
            reason_val = str(row.get("fail_reason") or "").strip()
            feasible_val = row.get("feasible")
            if reason_val or feasible_val is False:
                continue
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
    prob_key = problem.lower().strip() if isinstance(problem, str) else ""
    if prob_key in {"p3", "multi", "mhd", "qi_stable"} or "objectives" in metrics:
        objs = extract_objectives(metrics)
        if objs is not None:
            try:
                return float(scalarize(objs, DEFAULT_P3_SCALARIZATION))
            except ValueError:
                pass

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


def _real_eval_task(args: tuple[dict[str, Any], str, Mapping[str, Any]]) -> dict[str, Any]:
    """Helper for parallel real-evaluator calls.

    Accepts (boundary, problem) and returns metrics dict.
    """
    b, prob, vmec_opts_in = args
    vmec_opts = dict(vmec_opts_in)
    try:
        from ..physics.proxima_eval import forward_metrics as px_forward

        _t0 = time.perf_counter()
        _metrics_raw, info = px_forward(b, problem=prob, vmec_opts=vmec_opts)
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
        return _annotate_vmec_metadata(metrics, vmec_opts)
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
        if isinstance(metrics, dict):
            metrics = _annotate_vmec_metadata(metrics, vmec_opts)
        return metrics


def _placeholder_eval_task(
    b: dict[str, Any], vmec_opts: Mapping[str, Any] | None = None
) -> dict[str, Any]:
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
    if vmec_opts is not None and isinstance(metrics, dict):
        metrics = _annotate_vmec_metadata(metrics, vmec_opts)
    return metrics


__all__ = [
    "boundary_to_vmec",
    "forward",
    "forward_many",
    "score",
    "MF_PROXY_METRICS",
]
