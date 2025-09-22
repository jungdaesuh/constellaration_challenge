from __future__ import annotations

import os
from contextlib import contextmanager
from math import inf, isnan
from typing import Any, Callable, Generator, Mapping, Tuple


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
        info = {
            "problem": problem,
            "feasible": True,
            "source": "constellaration",
            "vmec_level": level,
            "vmec_hot_restart": hot_restart,
        }
        if restart_key is not None:
            info["vmec_restart_key"] = restart_key
        return metrics, info
    except Exception:
        m = _fallback_metrics(boundary)
        # Best-effort cast to float values for compatibility
        metrics_f: dict[str, Any] = {
            k: float(v) for k, v in m.items() if isinstance(v, (int, float))
        }
        # Derive a bounded aggregate score in (0, 1]
        try:
            pm = float(m.get("placeholder_metric", 0.0))
        except Exception:
            pm = 0.0
        metrics_f["score"] = float(1.0 / (1.0 + max(0.0, pm)))
        # For multi-objective (p3), expose an objectives list to satisfy shape expectations
        prob_key = problem.lower().strip()
        if prob_key in {"p3", "multi", "mhd", "qi_stable"}:
            # Two synthetic objectives derived from simple placeholder norms
            r = float(m.get("r_cos_norm", 0.0))
            z = float(m.get("z_sin_norm", 0.0))
            metrics_f["objectives"] = [r, z]
        # Mark provenance correctly as placeholder
        info_placeholder = {
            "problem": problem,
            "feasible": True,
            "source": "placeholder",
            "vmec_level": level,
            "vmec_hot_restart": hot_restart,
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
    except Exception:
        pass

    total = 0.0
    for v in metrics.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
            if isnan(fv):
                return inf
            total += fv
    return float(total)


__all__ = ["boundary_to_vmec", "forward_metrics", "score"]
