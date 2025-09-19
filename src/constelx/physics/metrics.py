"""Unified metrics/constraints facade.

This module is the single source of truth for computing and enriching
metrics used across constelx. It centralizes:

- Placeholder vs real evaluator metrics
- Boozer-space proxy metrics (bounded, cheap fallbacks)
- Light geometry-derived helpers where cheap to compute

Design notes
- Keep dependencies optional: real physics paths are attempted only when
  explicitly requested via flags or environment variables.
- Do not override authoritative fields already present in an input metrics
  mapping (e.g., score/objective from the evaluator). Enrichment only adds
  missing values using setdefault.
- Names are stable across paths so downstream components (agent, gating,
  scoring) do not drift.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _attach_geometry_defaults(boundary: Mapping[str, Any], out: MutableMapping[str, Any]) -> None:
    # Cheap, representation-level helpers that are invariant across paths.
    # nfp
    try:
        nfp = int(boundary.get("n_field_periods", 0) or 0)
    except Exception:
        nfp = 0
    out.setdefault("nfp", nfp)
    # symmetry (best-effort)
    try:
        ss = bool(boundary.get("is_stellarator_symmetric", True))
    except Exception:
        ss = True
    out.setdefault("stellarator_symmetric", ss)


def _attach_boozer_proxies(boundary: Mapping[str, Any], out: MutableMapping[str, Any]) -> None:
    try:
        from .booz_proxy import compute_proxies

        proxies = compute_proxies(boundary)
        proxy_dict = proxies.as_dict()
        for k, v in proxy_dict.items():
            # Do not override if evaluator already provided a value for the same key.
            if k not in out:
                out[k] = _as_float(v)
    except Exception:
        # Proxies are optional — ignore failures silently to keep callers robust.
        pass


def compute(
    boundary: Mapping[str, Any],
    *,
    use_real: Optional[bool] = None,
    problem: Optional[str] = None,
    attach_proxies: bool = True,
) -> dict[str, Any]:
    """Compute metrics for a boundary with optional real-physics path.

    - When use_real is truthy, attempt to call the official evaluator via
      constellaration adapters; otherwise, use placeholder metrics.
    - Always attach cheap geometry defaults; attach Boozer proxies when requested.
    - Never overwrite evaluator-provided fields.
    """

    metrics: dict[str, Any]

    # Resolve feature flag lazily
    if use_real is None:
        import os

        use_real = os.getenv("CONSTELX_USE_REAL_EVAL", "0").lower() in {"1", "true", "yes"}

    if use_real:
        try:
            from .proxima_eval import forward_metrics as px_forward

            m, _info = px_forward(boundary, problem=problem or "p1")
            metrics = dict(m)
        except Exception:
            from .constel_api import evaluate_boundary

            metrics = evaluate_boundary(dict(boundary), use_real=False)
    else:
        from .constel_api import evaluate_boundary

        metrics = evaluate_boundary(dict(boundary), use_real=False)

    # Enrichment (non-destructive)
    _attach_geometry_defaults(boundary, metrics)
    if attach_proxies:
        _attach_boozer_proxies(boundary, metrics)

    return metrics


def enrich(
    metrics: Mapping[str, Any],
    boundary: Mapping[str, Any],
    *,
    attach_proxies: bool = True,
) -> dict[str, Any]:
    """Return a copy of metrics enriched with canonical fields.

    Safe to call on results produced by either the real or placeholder path.
    """
    out = dict(metrics)
    _attach_geometry_defaults(boundary, out)
    if attach_proxies:
        _attach_boozer_proxies(boundary, out)
    return out


__all__ = ["compute", "enrich"]
