from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from math import inf, isnan
from typing import Any, Callable, Generator, Mapping, MutableMapping, Tuple

from ..dev import is_dev_mode, require_dev_for_placeholder

logger = logging.getLogger(__name__)


def boundary_to_vmec(boundary: Mapping[str, Any]) -> Any:
    """Validate and convert a boundary dict to a constellaration VMEC boundary model.

    Returns a SurfaceRZFourier pydantic model when constellaration is available.
    Raises RuntimeError if the library is missing or validation fails.
    """
    try:
        from constellaration.geometry import surface_rz_fourier as srf

        return srf.SurfaceRZFourier.model_validate(dict(boundary))
    except Exception as e:  # pragma: no cover - optional dependency branch
        raise RuntimeError(
            "constellaration is required for VMEC boundary validation. "
            "Install extras: pip install -e '.[physics]'"
        ) from e


def _fallback_metrics(boundary: Mapping[str, Any]) -> dict[str, Any]:
    # Lightweight placeholder using the existing development path
    from .constel_api import evaluate_boundary as _eval

    return _eval(dict(boundary), use_real=False)


def _apply_fallbacks(
    payload: MutableMapping[str, Any],
    *,
    feasible: bool | None,
    fail_reason: str,
    placeholder_reason: str | None = None,
) -> None:
    """Fill failure metadata without overriding authoritative evaluator data."""

    if feasible is not None and "feasible" not in payload:
        payload["feasible"] = bool(feasible)
    existing_reason = str(payload.get("fail_reason") or "").strip()
    if fail_reason and (
        not existing_reason or existing_reason in {"placeholder_metrics", "placeholder_eval"}
    ):
        payload["fail_reason"] = fail_reason
    if placeholder_reason and "placeholder_reason" not in payload:
        payload["placeholder_reason"] = placeholder_reason


@contextmanager
def _vmec_verbose_context(enable: bool) -> Generator[None, None, None]:
    if not enable:
        yield
        return
    try:
        from constellaration.forward_model import ConstellarationSettings

        orig_high = ConstellarationSettings.default_high_fidelity
        orig_skip = ConstellarationSettings.default_high_fidelity_skip_qi

        def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                settings = fn(*args, **kwargs)
                try:
                    settings.vmec_preset_settings.verbose = True
                except Exception:
                    pass
                return settings

            return wrapper

        ConstellarationSettings.default_high_fidelity = staticmethod(_wrap(orig_high))
        ConstellarationSettings.default_high_fidelity_skip_qi = staticmethod(_wrap(orig_skip))
        try:
            yield
        finally:
            ConstellarationSettings.default_high_fidelity = staticmethod(orig_high)
            ConstellarationSettings.default_high_fidelity_skip_qi = staticmethod(orig_skip)
    except Exception:
        yield


def _apply_vmec_hot_settings(settings: Any, hot_restart: bool, restart_key: str | None) -> Any:
    if settings is None:
        return settings
    try:
        preset = getattr(settings, "vmec_preset_settings", None)
        if preset is not None:
            if hot_restart:
                try:
                    if hasattr(preset, "enable_hot_restart"):
                        setattr(preset, "enable_hot_restart", True)
                except Exception:
                    pass
                hr = getattr(preset, "hot_restart", None)
                if hr is not None:
                    try:
                        setattr(hr, "enabled", True)
                    except Exception:
                        pass
                    if restart_key is not None:
                        try:
                            setattr(hr, "restart_key", restart_key)
                        except Exception:
                            pass
            else:
                hr = getattr(preset, "hot_restart", None)
                if hr is not None:
                    try:
                        setattr(hr, "enabled", False)
                    except Exception:
                        pass
    except Exception:
        pass
    if restart_key is not None and hasattr(settings, "restart_key"):
        try:
            setattr(settings, "restart_key", restart_key)
        except Exception:
            pass
    return settings


@contextmanager
def _vmec_settings_context(
    level: str,
    hot_restart: bool,
    restart_key: str | None,
    verbose_flag: bool,
    log_flag: bool,
) -> Generator[None, None, None]:
    with _vmec_verbose_context(verbose_flag or log_flag):
        try:
            from constellaration.forward_model import ConstellarationSettings

            level_map = {
                "low": "default_low_fidelity",
                "medium": "default_medium_fidelity",
                "high": "default_high_fidelity",
            }
            attr = level_map.get(level)
            base_factory: Callable[..., Any]
            if attr and hasattr(ConstellarationSettings, attr):
                base_factory = getattr(ConstellarationSettings, attr)
            else:
                base_factory = ConstellarationSettings.default_high_fidelity

            orig_high = ConstellarationSettings.default_high_fidelity
            orig_skip = ConstellarationSettings.default_high_fidelity_skip_qi

            def _wrap(factory: Callable[..., Any]) -> Callable[..., Any]:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    settings = factory(*args, **kwargs)
                    return _apply_vmec_hot_settings(settings, hot_restart, restart_key)

                return wrapper

            wrapped = _wrap(base_factory)
            ConstellarationSettings.default_high_fidelity = staticmethod(wrapped)
            ConstellarationSettings.default_high_fidelity_skip_qi = staticmethod(wrapped)
        except Exception:
            yield
            return

        try:
            yield
        finally:
            ConstellarationSettings.default_high_fidelity = staticmethod(orig_high)
            ConstellarationSettings.default_high_fidelity_skip_qi = staticmethod(orig_skip)


def forward_metrics(
    boundary: Mapping[str, Any], *, problem: str, vmec_opts: dict[str, Any] | None = None
) -> Tuple[dict[str, Any], dict[str, Any]]:
    """Compute metrics via the official constellaration evaluator.

    Returns (metrics, info). Falls back to lightweight placeholder metrics if the
    physics stack is unavailable so that callers can degrade gracefully.
    """
    opts = vmec_opts or {}
    level_raw = str(opts.get("level") or "auto").lower()
    level = level_raw if level_raw in {"low", "medium", "high"} else "auto"
    hot_restart = bool(opts.get("hot_restart", False))
    restart_key = opts.get("restart_key")
    if isinstance(restart_key, str) and not restart_key.strip():
        restart_key = None

    prob_key = problem.lower().strip()

    try:
        # Prefer problem-driven evaluation path compatible with constellaration>=0.2.x
        from constellaration.geometry import surface_rz_fourier as srf
        from constellaration.problems import (
            GeometricalProblem,
            MHDStableQIStellarator,
            SimpleToBuildQIStellarator,
        )

        verbose_flag = os.getenv("CONSTELX_VMEC_VERBOSE", "").lower() in {
            "1",
            "true",
            "yes",
        }
        log_flag = bool(os.getenv("CONSTELX_EVAL_LOG_DIR"))

        with _vmec_settings_context(level, hot_restart, restart_key, verbose_flag, log_flag):
            b = srf.SurfaceRZFourier.model_validate(dict(boundary))
            if prob_key in {"p1", "geom", "geometric", "geometrical"}:
                ev = GeometricalProblem().evaluate(b)
            elif prob_key in {"p2", "simple", "qi_simple", "simple_qi"}:
                ev = SimpleToBuildQIStellarator().evaluate(b)
            else:
                # Multi-objective problem expects a list of boundaries
                ev = MHDStableQIStellarator().evaluate([b])

        # Convert evaluation to metrics dict
        metrics: dict[str, Any] = {}
        try:
            d = ev.model_dump()  # pydantic model
        except Exception:
            d = getattr(ev, "__dict__", {})
        # Common fields: objective (float) for single-objective; score; feasibility
        if isinstance(d.get("objective"), (int, float)):
            metrics["objective"] = float(d["objective"])
        if isinstance(d.get("score"), (int, float)):
            metrics["score"] = float(d["score"])
        if isinstance(d.get("feasibility"), (int, float)):
            metrics["feasibility"] = float(d["feasibility"])
        # Multi-objective: capture objectives if present
        objs = d.get("objectives")
        if isinstance(objs, (list, tuple)):
            # Keep a list for callers that expect multi-objective shape
            metrics["objectives"] = [float(v) for v in objs if isinstance(v, (int, float))]
            for k, val in enumerate(objs):
                if isinstance(val, (int, float)):
                    metrics[f"objective_{k}"] = float(val)
        # Extract authoritative feasibility context from the evaluation object.
        feasible_flag: bool | None = None
        reason_text: str | None = None
        violation_payload: Any = None

        def _maybe_bool(val: Any) -> bool | None:
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                try:
                    return bool(int(val)) if val in (0, 1) else bool(val <= 0.0)
                except Exception:
                    return None
            return None

        # Candidate boolean fields exposed by constellaration models
        for key in ("feasible", "is_feasible", "success"):
            candidate = d.get(key, None)
            if candidate is None:
                candidate = getattr(ev, key, None)
            flag = _maybe_bool(candidate)
            if flag is not None:
                feasible_flag = flag
                break

        if feasible_flag is None:
            feas_metric = d.get("feasibility", None)
            if feas_metric is None:
                feas_metric = getattr(ev, "feasibility", None)
            if isinstance(feas_metric, (int, float)):
                try:
                    feasible_flag = feas_metric <= 0.0
                except Exception:
                    feasible_flag = None

        for key in ("fail_reason", "reason", "failure_reason", "status_message"):
            val = d.get(key)
            if not val:
                val = getattr(ev, key, None)
            if isinstance(val, str) and val.strip():
                reason_text = val.strip()
                break

        if reason_text is None:
            violations_attr = None
            for key in ("violation_details", "violations", "violation_context"):
                violations_attr = d.get(key)
                if violations_attr is None:
                    violations_attr = getattr(ev, key, None)
                if violations_attr:
                    violation_payload = violations_attr
                    # Derive a readable string for the reason field.
                    reason_text = str(violations_attr)
                    break

        info = {
            "problem": problem,
            "source": "real",
            "vmec_level": level,
            "vmec_hot_restart": hot_restart,
        }
        if restart_key is not None:
            info["vmec_restart_key"] = restart_key
        if feasible_flag is not None:
            info["feasible"] = bool(feasible_flag)
        if reason_text:
            info["reason"] = reason_text
        if violation_payload is not None:
            info["violations"] = violation_payload
        return metrics, info
    except Exception as exc:
        if not is_dev_mode():
            raise RuntimeError(
                "Real physics evaluation failed. Install extras via pip install -e '.[physics]' "
                "or set CONSTELX_DEV=1 to allow placeholder metrics explicitly."
            ) from exc
        require_dev_for_placeholder("Placeholder evaluation (proxima_eval.forward_metrics)")
        m = _fallback_metrics(boundary)
        metrics_f: dict[str, Any] = {
            k: float(v) for k, v in m.items() if isinstance(v, (int, float))
        }
        metrics_f["source"] = "placeholder"
        try:
            pm = float(m.get("placeholder_metric", 0.0))
        except Exception:
            pm = 0.0
        metrics_f["score"] = float(1.0 / (1.0 + max(0.0, pm)))
        prob_key = problem.lower().strip()
        if prob_key in {"p3", "multi", "mhd", "qi_stable"}:
            r = float(m.get("r_cos_norm", 0.0))
            z = float(m.get("z_sin_norm", 0.0))
            metrics_f["objectives"] = [r, z]
        reason_msg = f"real_eval_error: {exc}"
        placeholder_reason = "real_evaluator_failure"
        _apply_fallbacks(
            metrics_f,
            feasible=False,
            fail_reason=reason_msg,
            placeholder_reason=placeholder_reason,
        )
        metrics_f.setdefault("phase", "real")
        info_placeholder = {
            "problem": problem,
            "source": "placeholder",
            "vmec_level": level,
            "vmec_hot_restart": hot_restart,
            "feasible": False,
            "phase": "real",
            "reason": reason_msg,
            "fail_reason": reason_msg,
            "placeholder_reason": placeholder_reason,
        }
        if restart_key is not None:
            info_placeholder["vmec_restart_key"] = restart_key
        return metrics_f, info_placeholder


def score(problem: str, metrics: Mapping[str, Any]) -> float:
    """Aggregate a scalar score using the official scorer when available.

    Falls back to summing numeric metrics (NaN -> +inf) if the physics scorer
    is unavailable.
    """
    try:
        from constellaration.metrics import scoring as px_scoring

        # Try a few likely entrypoints to avoid hard-coding an exact name.
        if hasattr(px_scoring, "score"):
            return float(getattr(px_scoring, "score")(problem, dict(metrics)))
        if hasattr(px_scoring, "aggregate_score"):
            return float(getattr(px_scoring, "aggregate_score")(problem, dict(metrics)))
    except Exception as exc:
        logger.warning(
            "constellaration scorer unavailable; falling back to placeholder aggregation",
            exc_info=exc,
        )
        if isinstance(metrics, dict):
            metrics.setdefault("scorer_fallback", True)
            warning_msg = f"scorer_import_failed: {exc}" if exc else "scorer_import_failed"
            metrics.setdefault("scorer_warning", warning_msg)

    total = 0.0
    for v in metrics.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
            if isnan(fv):
                return inf
            total += fv
    return float(total)


__all__ = ["boundary_to_vmec", "forward_metrics", "score"]
