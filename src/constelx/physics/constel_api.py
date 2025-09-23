from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional, Tuple, cast

from ..dev import is_dev_mode, require_dev_for_placeholder


def example_boundary() -> dict[str, Any]:
    """Return a tiny example stellarator-symmetric boundary as a plain dict."""

    r_cos = [[0.0] * 9 for _ in range(5)]
    z_sin = [[0.0] * 9 for _ in range(5)]
    r_cos[0][4] = 1.0
    r_cos[1][5] = -0.05
    z_sin[1][5] = 0.05
    return {
        "r_cos": r_cos,
        "r_sin": None,
        "z_cos": None,
        "z_sin": z_sin,
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }


def _ensure_placeholder_failure(
    metrics: MutableMapping[str, Any],
    *,
    reason: str,
    placeholder_reason: str | None = None,
) -> None:
    """Normalize placeholder results after a failed real evaluation."""

    if "source" not in metrics or str(metrics.get("source")) == "":
        metrics["source"] = "placeholder"
    if "feasible" not in metrics:
        metrics["feasible"] = False
    existing_reason = str(metrics.get("fail_reason") or "").strip()
    if reason and (not existing_reason or existing_reason in {"placeholder_metrics", "placeholder_eval"}):
        metrics["fail_reason"] = reason
    if placeholder_reason and "placeholder_reason" not in metrics:
        metrics["placeholder_reason"] = placeholder_reason


def evaluate_boundary(
    boundary_json: dict[str, Any], use_real: Optional[bool] = None
) -> dict[str, Any]:
    """Compute metrics for a boundary, preferring the real evaluator path."""

    import os

    if use_real is None:
        use_real = os.getenv("CONSTELX_USE_REAL_EVAL", "0").lower() in {"1", "true", "yes"}

    if use_real:
        try:
            metrics = _evaluate_with_real_physics(boundary_json)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            if is_dev_mode():
                placeholder = _compute_placeholder_metrics(boundary_json)
                if isinstance(placeholder, dict):
                    reason = f"real_eval_error: {exc}"
                    _ensure_placeholder_failure(
                        placeholder,
                        reason=reason,
                        placeholder_reason="real_evaluator_failure",
                    )
                return placeholder
            raise RuntimeError(
                "Real physics evaluation failed. Install extras via "
                "pip install -e '.[physics]' or set CONSTELX_DEV=1 to allow "
                "placeholder metrics explicitly."
            ) from exc
        _add_proxy_metrics(boundary_json, metrics)
        return metrics

    return _compute_placeholder_metrics(boundary_json)


def _evaluate_with_real_physics(boundary_json: dict[str, Any]) -> dict[str, Any]:
    """Evaluate metrics using the official constellaration adapter."""

    boundary_payload, problem = _strip_problem(boundary_json)
    from .proxima_eval import forward_metrics  # Lazy import to avoid cycles

    metrics_raw, info = forward_metrics(boundary_payload, problem=problem)
    metrics = dict(metrics_raw)
    if isinstance(info, dict):
        source = info.get("source")
        if isinstance(source, str):
            metrics.setdefault("source", source)
        feasible = info.get("feasible")
        if isinstance(feasible, bool):
            metrics.setdefault("feasible", feasible)
        reason = info.get("reason") or info.get("fail_reason")
        if isinstance(reason, str) and reason:
            metrics.setdefault("fail_reason", reason)
        placeholder_reason = info.get("placeholder_reason")
        if isinstance(placeholder_reason, str) and placeholder_reason:
            metrics.setdefault("placeholder_reason", placeholder_reason)
        phase = info.get("phase")
        if isinstance(phase, str) and phase:
            metrics.setdefault("phase", phase)
    metrics.setdefault("source", "real")
    return metrics


def _compute_placeholder_metrics(boundary_json: dict[str, Any]) -> dict[str, Any]:
    """Return lightweight placeholder metrics (dev flows and tests only)."""

    require_dev_for_placeholder("Placeholder evaluation (constel_api.evaluate_boundary)")

    try:
        import numpy as _np_mod

        np = cast(Any, _np_mod)
    except Exception:  # pragma: no cover - numpy missing in constrained envs
        np = None

    r_cos = boundary_json.get("r_cos")
    z_sin = boundary_json.get("z_sin")
    if np is not None:
        r_cos_norm = float((np.asarray(r_cos) ** 2).sum()) if r_cos is not None else 0.0
        z_sin_norm = float((np.asarray(z_sin) ** 2).sum()) if z_sin is not None else 0.0
    else:

        def _norm2(a: Any) -> float:
            return float(sum(sum((float(x) ** 2 for x in row)) for row in a))

        r_cos_norm = _norm2(r_cos) if r_cos is not None else 0.0
        z_sin_norm = _norm2(z_sin) if z_sin is not None else 0.0

    objectives = [r_cos_norm, z_sin_norm]
    metrics = {
        "r_cos_norm": r_cos_norm,
        "z_sin_norm": z_sin_norm,
        "nfp": int(boundary_json.get("n_field_periods", 0)),
        "stellarator_symmetric": bool(boundary_json.get("is_stellarator_symmetric", True)),
        "placeholder_metric": r_cos_norm + z_sin_norm,
        "objectives": objectives,
        "source": "placeholder",
    }
    _ensure_placeholder_failure(metrics, reason="placeholder_metrics")
    _add_proxy_metrics(boundary_json, metrics)
    return metrics


def _strip_problem(boundary_json: dict[str, Any]) -> Tuple[dict[str, Any], str]:
    """Return a boundary dict suitable for VMEC validation and the problem key."""

    clean = dict(boundary_json)
    problem = clean.pop("problem", None) or clean.pop("_problem", None) or "p1"
    return clean, str(problem)


def _add_proxy_metrics(boundary_json: dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Attach Boozer proxy metrics when computable without raising."""

    try:
        from ..eval.boozer import compute_boozer_proxies

        proxies = compute_boozer_proxies(boundary_json)
    except Exception:  # pragma: no cover - optional physics dependency
        return

    proxy_values = proxies.to_dict()
    mapping = {
        "proxy_boozer_b_mean": proxy_values["b_mean"],
        "proxy_boozer_b_std_fraction": proxy_values["b_std_fraction"],
        "proxy_boozer_mirror_ratio": proxy_values["mirror_ratio"],
        "proxy_qs_residual": proxy_values["qs_residual"],
        "proxy_qs_quality": proxy_values["qs_quality"],
        "proxy_qi_residual": proxy_values["qi_residual"],
        "proxy_qi_quality": proxy_values["qi_quality"],
    }
    for key, value in mapping.items():
        metrics.setdefault(key, value)
