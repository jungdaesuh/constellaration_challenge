from __future__ import annotations

# ruff: noqa: I001  (import-sorting suppressed to match local style)

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .simple_agent import AgentConfig, run as run_agent


@dataclass(frozen=True)
class AblationResult:
    name: str
    best_score: float
    run_dir: Path


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_best_score(run_dir: Path) -> float:
    best_path = run_dir / "best.json"
    try:
        payload = json.loads(best_path.read_text())
        if isinstance(payload, dict):
            if isinstance(payload.get("agg_score"), (int, float)):
                return float(payload["agg_score"])
            if isinstance(payload.get("score"), (int, float)):
                return float(payload["score"])
    except Exception:
        pass
    return float("inf")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_ablation(
    *,
    base: AgentConfig,
    components: Iterable[str],
    out_root: Path,
) -> List[AblationResult]:
    """Run a simple ablation study over selected pipeline components.

    For each component name in `components`, toggles that single component on
    (relative to the provided `base` config), runs the agent with a small
    budget, and records the best aggregate score. A baseline run is always
    executed first using the unmodified base config.

    Results are written under `out_root/<timestamp>/` with subfolders
    `baseline/` and one folder per component name.
    """

    ts_root = out_root / _now_ts()
    _ensure_dir(ts_root)

    results: List[AblationResult] = []

    # Baseline
    base_dir = ts_root / "baseline"
    _ensure_dir(base_dir)
    # Force writing into our folder by using resume path
    run_agent(
        AgentConfig(
            nfp=base.nfp,
            seed=base.seed,
            out_dir=ts_root,
            algo=base.algo,
            budget=base.budget,
            resume=base_dir,
            max_workers=base.max_workers,
            cache_dir=base.cache_dir,
            correction=base.correction,
            constraints=base.constraints,
            use_physics=base.use_physics,
            problem=base.problem,
            init_seeds=base.init_seeds,
            guard_simple=base.guard_simple,
            guard_geo=base.guard_geo,
            guard_geom_validate=base.guard_geom_validate,
            guard_geom_r0_min=base.guard_geom_r0_min,
            guard_geom_r0_max=base.guard_geom_r0_max,
            guard_geom_helical_ratio_max=base.guard_geom_helical_ratio_max,
            mf_proxy=base.mf_proxy,
            mf_threshold=base.mf_threshold,
            mf_quantile=base.mf_quantile,
            mf_max_high=base.mf_max_high,
            pcfm_gn_iters=base.pcfm_gn_iters,
            pcfm_damping=base.pcfm_damping,
            pcfm_tol=base.pcfm_tol,
        )
    )
    results.append(
        AblationResult(name="baseline", best_score=_read_best_score(base_dir), run_dir=base_dir)
    )

    # Component-specific toggles
    for comp in components:
        name = str(comp).strip().lower()
        if not name:
            continue
        comp_dir = ts_root / name
        _ensure_dir(comp_dir)
        cfg = AgentConfig(
            nfp=base.nfp,
            seed=base.seed,
            out_dir=ts_root,
            algo=base.algo,
            budget=base.budget,
            resume=comp_dir,
            max_workers=base.max_workers,
            cache_dir=base.cache_dir,
            correction=base.correction,
            constraints=base.constraints,
            use_physics=base.use_physics,
            problem=base.problem,
            init_seeds=base.init_seeds,
            guard_simple=base.guard_simple,
            guard_geo=base.guard_geo,
            guard_geom_validate=base.guard_geom_validate,
            guard_geom_r0_min=base.guard_geom_r0_min,
            guard_geom_r0_max=base.guard_geom_r0_max,
            guard_geom_helical_ratio_max=base.guard_geom_helical_ratio_max,
            mf_proxy=base.mf_proxy,
            mf_threshold=base.mf_threshold,
            mf_quantile=base.mf_quantile,
            mf_max_high=base.mf_max_high,
            pcfm_gn_iters=base.pcfm_gn_iters,
            pcfm_damping=base.pcfm_damping,
            pcfm_tol=base.pcfm_tol,
        )

        # Apply a single-component toggle
        if name in {"guard_simple", "guard-simple"}:
            cfg = AgentConfig(**{**cfg.__dict__, "guard_simple": True})
        elif name in {"guard_geo", "guard-geo"}:
            cfg = AgentConfig(**{**cfg.__dict__, "guard_geo": True})
        elif name in {"guard_geom_validate", "guard-geom-validate"}:
            cfg = AgentConfig(**{**cfg.__dict__, "guard_geom_validate": True})
        elif name.startswith("mf_proxy") or name.startswith("mf-proxy"):
            # Enable proxy with a reasonable keep quantile for quick runs
            # Support forms: "mf_proxy", "mf_proxy=q:0.5", "mf_proxy=t:0.1"
            mf_threshold: Optional[float] = None
            mf_quantile: Optional[float] = 0.5
            if "=" in name and ":" in name:
                try:
                    mode, val = name.split("=", 1)[1].split(":", 1)
                    if mode in {"q", "quantile"}:
                        mf_quantile = float(val)
                        mf_threshold = None
                    elif mode in {"t", "thresh", "threshold"}:
                        mf_threshold = float(val)
                        mf_quantile = None
                except Exception:
                    pass
            cfg = AgentConfig(
                **{
                    **cfg.__dict__,
                    "mf_proxy": True,
                    "mf_threshold": mf_threshold,
                    "mf_quantile": mf_quantile,
                }
            )
        elif name in {"algo=cmaes", "algo:cmaes", "cmaes"}:
            cfg = AgentConfig(**{**cfg.__dict__, "algo": "cmaes"})
        elif name.startswith("algo=") or name.startswith("algo:"):
            # Allow generic algo toggles for future additions
            try:
                algo_val = name.split("=", 1)[1] if "=" in name else name.split(":", 1)[1]
                cfg = AgentConfig(**{**cfg.__dict__, "algo": str(algo_val)})
            except Exception:
                pass
        elif name in {"correction=eci_linear", "correction:eci_linear", "eci_linear"}:
            # Default simple equality: r_cos[1][5] + z_sin[1][5] = 0
            cfg = AgentConfig(
                **{
                    **cfg.__dict__,
                    "correction": "eci_linear",
                    "constraints": [
                        {
                            "rhs": 0.0,
                            "coeffs": [
                                {"field": "r_cos", "i": 1, "j": 5, "c": 1.0},
                                {"field": "z_sin", "i": 1, "j": 5, "c": 1.0},
                            ],
                        }
                    ],
                }
            )
        elif name in {"correction=pcfm", "correction:pcfm", "pcfm"}:
            # Default norm equality on (r_cos[1][5], z_sin[1][5]) radius=0.06
            cfg = AgentConfig(
                **{
                    **cfg.__dict__,
                    "correction": "pcfm",
                    "constraints": [
                        {
                            "type": "norm_eq",
                            "radius": 0.06,
                            "terms": [
                                {"field": "r_cos", "i": 1, "j": 5, "w": 1.0},
                                {"field": "z_sin", "i": 1, "j": 5, "w": 1.0},
                            ],
                        }
                    ],
                }
            )
        else:
            # Unknown component: run baseline settings but record name
            pass

        run_agent(cfg)
        results.append(
            AblationResult(name=name, best_score=_read_best_score(comp_dir), run_dir=comp_dir)
        )

    # Write a compact summary at the root timestamp folder
    summary_rows = [
        {"name": r.name, "best_score": r.best_score, "run_dir": str(r.run_dir)} for r in results
    ]
    (ts_root / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    # CSV is optional; keep columns minimal and stable
    lines = ["name,best_score,run_dir"]
    for r in results:
        lines.append(f"{r.name},{r.best_score},{r.run_dir}")
    (ts_root / "summary.csv").write_text("\n".join(lines) + "\n")

    return results


# -------------------- Spec-driven multi-seed ablation --------------------


@dataclass(frozen=True)
class PlanVariant:
    name: str
    overrides: Mapping[str, Any]


@dataclass(frozen=True)
class AblationPlan:
    base: Mapping[str, Any]
    seeds: Iterable[int]
    variants: Iterable[PlanVariant]


def _read_best_json(run_dir: Path) -> float:
    best_path = run_dir / "best.json"
    try:
        payload = json.loads(best_path.read_text())
        if isinstance(payload, dict):
            if isinstance(payload.get("agg_score"), (int, float)):
                return float(payload["agg_score"])
            if isinstance(payload.get("score"), (int, float)):
                return float(payload["score"])
    except Exception:
        pass
    return float("inf")


def run_ablation_plan(*, plan: AblationPlan, out_root: Path) -> Path:
    """Run an explicit multi-seed, multi-variant ablation plan.

    Plan schema (JSON-friendly):
      {"base": {...}, "seeds": [0,1], "variants": [{"name": "v1", "overrides": {...}}, ...]}

    Behavior:
      - Creates a timestamped root under out_root
      - For each variant and seed, runs the agent in a dedicated subfolder
        root/variant/seed_<k>/ using resume to force placement.
      - Writes details.csv (variant,seed,best_agg_score,run_dir) and
        summary.csv (variant,best_agg_score,mean_agg_score) at the root.
    """
    ts_root = out_root / _now_ts()
    _ensure_dir(ts_root)

    base_map: Dict[str, Any] = dict(plan.base)
    # Normalize required fields; minimal base fallback
    if "nfp" not in base_map:
        base_map["nfp"] = 3
    if "budget" not in base_map:
        base_map["budget"] = 4
    # We always control physical location via resume below
    base_map.setdefault("algo", "random")

    # Persist plan for reproducibility
    plan_json = {
        "base": base_map,
        "seeds": list(int(s) for s in plan.seeds),
        "variants": [{"name": v.name, "overrides": dict(v.overrides)} for v in plan.variants],
    }
    (ts_root / "plan.json").write_text(json.dumps(plan_json, indent=2))

    # Collect details
    details_rows: List[Mapping[str, Any]] = []
    for variant in plan.variants:
        vdir = ts_root / str(variant.name)
        _ensure_dir(vdir)
        for sd in plan.seeds:
            sdir = vdir / f"seed_{int(sd)}"
            _ensure_dir(sdir)
            cfg_map: Dict[str, Any] = dict(base_map)
            cfg_map.update({"seed": int(sd)})
            cfg_map.update({k: v for k, v in variant.overrides.items()})

            cfg = AgentConfig(
                nfp=int(cfg_map.get("nfp", 3)),
                seed=int(cfg_map.get("seed", 0)),
                out_dir=ts_root,  # ignored when resume set
                algo=str(cfg_map.get("algo", "random")),
                budget=int(cfg_map.get("budget", 4)),
                resume=sdir,
                max_workers=int(cfg_map.get("max_workers", 1)),
                cache_dir=(Path(cfg_map["cache_dir"]) if cfg_map.get("cache_dir") else None),
                correction=(str(cfg_map["correction"]) if cfg_map.get("correction") else None),
                constraints=(
                    list(cfg_map["constraints"])
                    if isinstance(cfg_map.get("constraints"), list)
                    else None
                ),
                use_physics=bool(cfg_map.get("use_physics", False)),
                pcfm_gn_iters=(
                    int(cfg_map["pcfm_gn_iters"])
                    if cfg_map.get("pcfm_gn_iters") is not None
                    else None
                ),
                pcfm_damping=(
                    float(cfg_map["pcfm_damping"])
                    if cfg_map.get("pcfm_damping") is not None
                    else None
                ),
                pcfm_tol=(
                    float(cfg_map["pcfm_tol"]) if cfg_map.get("pcfm_tol") is not None else None
                ),
                problem=(str(cfg_map["problem"]) if cfg_map.get("problem") else None),
                init_seeds=(Path(cfg_map["init_seeds"]) if cfg_map.get("init_seeds") else None),
                guard_simple=bool(cfg_map.get("guard_simple", False)),
                guard_geo=bool(cfg_map.get("guard_geo", False)),
                guard_geom_validate=bool(cfg_map.get("guard_geom_validate", False)),
                guard_geom_r0_min=float(cfg_map.get("guard_geom_r0_min", 0.05)),
                guard_geom_r0_max=float(cfg_map.get("guard_geom_r0_max", 5.0)),
                guard_geom_helical_ratio_max=float(
                    cfg_map.get("guard_geom_helical_ratio_max", 0.5)
                ),
                mf_proxy=bool(cfg_map.get("mf_proxy", False)),
                mf_threshold=(
                    float(cfg_map["mf_threshold"])
                    if cfg_map.get("mf_threshold") is not None
                    else None
                ),
                mf_quantile=(
                    float(cfg_map["mf_quantile"])
                    if cfg_map.get("mf_quantile") is not None
                    else None
                ),
                mf_max_high=(
                    int(cfg_map["mf_max_high"]) if cfg_map.get("mf_max_high") is not None else None
                ),
            )

            run_agent(cfg)
            best = _read_best_json(sdir)
            details_rows.append(
                {
                    "variant": variant.name,
                    "seed": int(sd),
                    "best_agg_score": best,
                    "run_dir": str(sdir),
                }
            )

    # Write details.csv and summary.csv/json
    import csv

    with (ts_root / "details.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "seed", "best_agg_score", "run_dir"])
        w.writeheader()
        for r in details_rows:
            w.writerow(r)

    # Aggregate per variant
    from math import inf

    by_variant: Dict[str, List[float]] = {}
    for r in details_rows:
        by_variant.setdefault(str(r["variant"]), []).append(float(r["best_agg_score"]))
    summary_rows: List[Mapping[str, Any]] = []
    for name, vals in by_variant.items():
        if vals:
            best = min(vals)
            mean = sum(vals) / len(vals)
        else:
            best, mean = inf, inf
        summary_rows.append({"variant": name, "best_agg_score": best, "mean_agg_score": mean})
    summary_rows.sort(key=lambda x: x["variant"])

    with (ts_root / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "best_agg_score", "mean_agg_score"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    (ts_root / "summary.json").write_text(json.dumps(summary_rows, indent=2))

    return ts_root
