"""Minimal agent loop utilities."""

# ruff: noqa: I001  (import-sorting suppressed for local grouping)

from __future__ import annotations

import csv
import json
import time
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Callable

from ..eval import forward as eval_forward, score as eval_score
from ..eval.boundary_param import (
    sample_random,
    sample_near_axis_qs,
    validate as validate_boundary,
)


@dataclass(frozen=True)
class AgentConfig:
    nfp: int
    seed: int
    out_dir: Path
    # Optional: explore multiple NFP values in one run (round-robin allocation)
    nfp_list: list[int] | None = None
    algo: str = "random"  # one of: random, cmaes
    budget: int = 50  # total evaluations allowed
    resume: Path | None = None
    max_workers: int = 1
    cache_dir: Path | None = None
    correction: str | None = None  # e.g., "eci_linear"
    # simple list of constraints when using eci_linear
    constraints: list[dict[str, Any]] | None = None
    use_physics: bool = False
    # Optional PCFM tuning (overrides file-provided values if set)
    pcfm_gn_iters: int | None = None
    pcfm_damping: float | None = None
    pcfm_tol: float | None = None
    # Optional problem id for real physics ('p1'|'p2'|'p3')
    problem: str | None = None
    # Optional initial seeds JSONL path containing boundaries
    init_seeds: Path | None = None
    # Optional simple guard to clamp base radius and helical amplitudes
    guard_simple: bool = False
    # Optional geometric nudge: tighten helical amps and align ratio toward 1
    guard_geo: bool = False
    # Optional strict geometry validation (skip invalid and log fail_reason)
    guard_geom_validate: bool = False
    # Geometry guard thresholds (apply when guard_geom_validate=True)
    guard_geom_r0_min: float = 0.05
    guard_geom_r0_max: float = 5.0
    guard_geom_helical_ratio_max: float = 0.5
    # Multi-fidelity gating
    mf_proxy: bool = False
    mf_threshold: float | None = None
    mf_quantile: float | None = None
    mf_max_high: int | None = None
    # Seed generator: "random" or "near-axis"
    seed_mode: str = "random"
    # Novelty gating
    novelty_skip: bool = False
    novelty_metric: str = "l2"  # l2|cosine|allclose
    novelty_eps: float = 1e-6
    novelty_window: int = 128
    novelty_db: Path | None = None
    # Surrogate screening
    surrogate_screen: bool = False
    surrogate_model: Path | None = None
    surrogate_metadata: Path | None = None
    surrogate_threshold: float | None = None
    surrogate_quantile: float | None = None
    surrogate_keep_max: int | None = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        # create empty file with no rows
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _gather_env_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split(" ")[0],
        "platform": sys.platform,
        "cwd": os.getcwd(),
    }


def _git_sha() -> str | None:
    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return None


def _pkg_versions(pkgs: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pkgs:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "unknown"
        except Exception:
            out[p] = "unknown"
    return out


_CorrectionHook = Callable[[Mapping[str, Any]], Dict[str, Any]]


def _build_eci_linear_hook(constraints: List[Dict[str, Any]]) -> Optional[_CorrectionHook]:
    try:
        from .corrections.eci_linear import (
            EciLinearSpec,
            LinearConstraint,
            Variable,
            make_hook,
        )
    except Exception:
        return None

    # Collect variables encountered in constraints in order of first appearance
    var_index: Dict[Tuple[str, int, int], int] = {}
    variables: List[Variable] = []

    def _get_var(field: str, i: int, j: int) -> Variable:
        key = (field, int(i), int(j))
        if key not in var_index:
            var_index[key] = len(variables)
            variables.append(Variable(field=field, i=int(i), j=int(j)))
        return variables[var_index[key]]

    lin_cons: List[LinearConstraint] = []
    for con in constraints:
        rhs = float(con.get("rhs", 0.0))
        coeffs_in = con.get("coeffs", [])
        coeffs: List[Tuple[Variable, float]] = []
        for c in coeffs_in:
            field = str(c["field"])  # required
            i = int(c["i"])  # required
            j = int(c["j"])  # required
            val = float(c.get("c", c.get("coeff", 0.0)))
            coeffs.append((_get_var(field, i, j), val))
        lin_cons.append(LinearConstraint(coeffs=coeffs, rhs=rhs))

    spec = EciLinearSpec(variables=variables, constraints=lin_cons)
    return make_hook(spec)


def _build_pcfm_hook(
    constraints: List[Dict[str, Any]],
    *,
    gn_iters: int | None = None,
    damping: float | None = None,
    tol: float | None = None,
) -> Optional[_CorrectionHook]:
    try:
        from .corrections.pcfm import PcfmSpec, build_spec_from_json, make_hook
    except Exception:
        return None
    try:
        spec: PcfmSpec = build_spec_from_json(constraints)
    except Exception:
        return None
    # Apply overrides if provided
    if gn_iters is not None:
        spec = PcfmSpec(
            variables=spec.variables,
            constraints=spec.constraints,
            coeff_abs_max=spec.coeff_abs_max,
            gn_iters=int(gn_iters),
            damping=spec.damping,
            tol=spec.tol,
        )
    if damping is not None:
        spec = PcfmSpec(
            variables=spec.variables,
            constraints=spec.constraints,
            coeff_abs_max=spec.coeff_abs_max,
            gn_iters=spec.gn_iters,
            damping=float(damping),
            tol=spec.tol,
        )
    if tol is not None:
        spec = PcfmSpec(
            variables=spec.variables,
            constraints=spec.constraints,
            coeff_abs_max=spec.coeff_abs_max,
            gn_iters=spec.gn_iters,
            damping=spec.damping,
            tol=float(tol),
        )
    return make_hook(spec)


def run(config: AgentConfig) -> Path:
    """Run a tiny random-search agent loop and write artifacts.

    This is a minimal, deterministic implementation that samples random
    boundaries, evaluates metrics, aggregates a score, and tracks the best.
    It avoids optional deps (e.g., CMA-ES) to keep tests lightweight.

    Returns the output directory path.
    """

    # Decide output directory (new or resume)
    if config.resume is not None:
        out_dir = Path(config.resume)
        _ensure_dir(out_dir)
    else:
        out_dir = config.out_dir / _timestamp()
        _ensure_dir(out_dir)

    # config.yaml with env info, git SHA, and package versions
    cfg = asdict(config)
    cfg["out_dir"] = str(cfg["out_dir"])  # Path -> str for YAML/JSON compatibility
    # Only write config on fresh runs (avoid clobber on resume)
    if config.resume is None:
        # Ensure JSON-serializable
        if cfg.get("resume") is not None:
            cfg["resume"] = str(cfg["resume"])
        if cfg.get("init_seeds") is not None:
            cfg["init_seeds"] = str(cfg["init_seeds"])  # Path -> str
        if cfg.get("cache_dir") is not None:
            cfg["cache_dir"] = str(cfg["cache_dir"])
        if cfg.get("surrogate_model") is not None:
            cfg["surrogate_model"] = str(cfg["surrogate_model"])
        if cfg.get("surrogate_metadata") is not None:
            cfg["surrogate_metadata"] = str(cfg["surrogate_metadata"])
        conf = {
            "run": cfg,
            "env": _gather_env_info(),
            "git": {"sha": _git_sha()},
            "versions": _pkg_versions(["constelx", "numpy", "pandas", "constellaration"]),
        }
        # YAML supports JSON subset; write JSON content for portability
        (out_dir / "config.yaml").write_text(json.dumps(conf, indent=2))

    proposals_path = out_dir / "proposals.jsonl"
    metrics_csv_path = out_dir / "metrics.csv"
    best_json_path = out_dir / "best.json"
    readme_path = out_dir / "README.md"
    # Novelty DB path (optional)
    novelty_db_path = (
        (config.novelty_db if config.novelty_db is not None else out_dir / "novelty.jsonl")
        if config.novelty_skip
        else None
    )

    # When surrogate screening is enabled, make sure metrics.csv always exposes the
    # surrogate-specific bookkeeping columns so resume logic can distinguish proxy
    # rows even if filtering starts after the first evaluation.
    forced_metric_columns: tuple[str, ...] = ()
    if config.surrogate_screen:
        forced_metric_columns = ("phase", "surrogate_score")

    # Resume bookkeeping
    completed = 0
    best_score = float("inf")
    best_payload: Dict[str, Any] = {}
    metrics_counted = False
    if metrics_csv_path.exists():
        try:
            with metrics_csv_path.open("r", newline="") as mf:
                dict_reader = csv.DictReader(mf)
                seen_row = False
                for row in dict_reader:
                    seen_row = True
                    phase_val = (row.get("phase") or "").strip().lower()
                    fail_reason_val = (row.get("fail_reason") or "").strip().lower()
                    if phase_val == "surrogate" or fail_reason_val == "filtered_surrogate":
                        continue
                    completed += 1
                metrics_counted = seen_row
        except Exception:
            completed = 0
            metrics_counted = False
    if proposals_path.exists():
        if not metrics_counted:
            completed = sum(1 for _ in proposals_path.open())
    if best_json_path.exists():
        try:
            best_payload = json.loads(best_json_path.read_text())
            if isinstance(best_payload.get("score"), (int, float)):
                best_score = float(best_payload["score"])
        except Exception:
            best_payload = {}
            best_score = float("inf")

    def _ensure_metrics_columns(
        path: Path, required: tuple[str, ...]
    ) -> list[str] | None:
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            with path.open("r", newline="") as rf:
                dict_reader = csv.DictReader(rf)
                rows = list(dict_reader)
                header = list(dict_reader.fieldnames or [])
        except Exception:
            return None
        missing = [col for col in required if col not in header]
        if not missing:
            return header
        header.extend(missing)
        with path.open("w", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=header)
            writer.writeheader()
            for row in rows:
                for col in missing:
                    row.setdefault(col, "")
                writer.writerow(row)
        return header

    existing_header: list[str] | None = None
    if metrics_csv_path.exists() and metrics_csv_path.stat().st_size > 0:
        existing_header = _ensure_metrics_columns(metrics_csv_path, forced_metric_columns)

    # Open for append
    proposals_f = proposals_path.open("a")
    metrics_f = metrics_csv_path.open("a", newline="")
    metrics_writer: csv.DictWriter[str] | None = None
    if existing_header:
        try:
            metrics_writer = csv.DictWriter(
                metrics_f, fieldnames=existing_header, extrasaction="ignore"
            )
        except Exception:
            metrics_writer = None
    elif metrics_csv_path.exists() and metrics_csv_path.stat().st_size > 0:
        try:
            with metrics_csv_path.open("r", newline="") as rf:
                csv_reader = csv.reader(rf)
                header = next(csv_reader, None)
            if header:
                metrics_writer = csv.DictWriter(metrics_f, fieldnames=header, extrasaction="ignore")
        except Exception:
            metrics_writer = None

    rng_seed = int(config.seed)
    budget = int(config.budget)
    problem = (config.problem or "p1") if config.use_physics else "p1"

    # ---------------- Novelty helpers ----------------
    from collections import deque
    import numpy as _np

    def _flatten_map(boundary: Mapping[str, Any]) -> Mapping[str, float]:
        # Include nfp to avoid collisions across NFPs
        out: Dict[str, float] = {"nfp": float(boundary.get("n_field_periods", 0) or 0)}
        try:
            r_cos = boundary.get("r_cos") or []
            z_sin = boundary.get("z_sin") or []
            for i, row in enumerate(r_cos):
                for j, v in enumerate(row):
                    out[f"r_cos_{i}_{j}"] = float(v)
            for i, row in enumerate(z_sin):
                for j, v in enumerate(row):
                    out[f"z_sin_{i}_{j}"] = float(v)
        except Exception:
            pass
        return out

    def _flatten_vec(boundary: Mapping[str, Any]) -> tuple[int, _np.ndarray]:
        nfp_val = int(boundary.get("n_field_periods", 0) or 0)
        rc = boundary.get("r_cos") or []
        zs = boundary.get("z_sin") or []
        vals: list[float] = [float(nfp_val)]
        try:
            for row in rc:
                for v in row:
                    vals.append(float(v))
            for row in zs:
                for v in row:
                    vals.append(float(v))
        except Exception:
            pass
        return nfp_val, _np.asarray(vals, dtype=float)

    def _is_novel_local(boundary: Mapping[str, Any], window: dict[int, deque[_np.ndarray]]) -> bool:
        nfp_val, v = _flatten_vec(boundary)
        hist = window.get(nfp_val)
        if not hist:
            return True
        metric = (config.novelty_metric or "l2").lower()
        eps = float(config.novelty_eps)
        if metric == "cosine":

            def cdist(a: _np.ndarray, b: _np.ndarray) -> float:
                na = _np.linalg.norm(a)
                nb = _np.linalg.norm(b)
                if na == 0 or nb == 0:
                    return float("inf")
                cos = float(a.dot(b) / (na * nb))
                return 1.0 - cos
        elif metric == "allclose":

            def cdist(a: _np.ndarray, b: _np.ndarray) -> float:
                return 0.0 if _np.allclose(a, b, atol=eps, rtol=0.0) else float("inf")
        else:  # l2

            def cdist(a: _np.ndarray, b: _np.ndarray) -> float:
                d = a - b
                return float(_np.sqrt(float(d.dot(d))))

        for u in hist:
            if cdist(v, u) <= eps:
                return False
        return True

    novelty_window: dict[int, deque[_np.ndarray]] = {}
    novelty_db = None
    if config.novelty_skip and novelty_db_path is not None:
        try:
            from ..data.results_db import ResultsDB

            novelty_db = ResultsDB(novelty_db_path)
        except Exception:
            novelty_db = None

    def log_entry(
        it: int,
        idx: int,
        seed_val: int,
        boundary: Dict[str, Any],
        metrics: Dict[str, Any],
        agg_s: float,
        *,
        elapsed_ms: float | None = None,
    ) -> None:
        nonlocal best_score, best_payload, metrics_writer
        # Derive nfp from boundary when present
        try:
            nfp_raw = boundary.get("n_field_periods")
            nfp_val = int(nfp_raw) if isinstance(nfp_raw, (int, float)) else None
        except Exception:
            nfp_val = None
        prop = {
            "iteration": it,
            "index": idx,
            "seed": seed_val,
            "nfp": nfp_val,
            "boundary": boundary,
        }
        proposals_f.write(json.dumps(prop) + "\n")
        # Separate evaluator-provided score from aggregated score to avoid confusion
        evaluator_score: float | None = None
        val_score = metrics.get("score")
        if isinstance(val_score, (int, float)):
            try:
                evaluator_score = float(val_score)
            except Exception:
                evaluator_score = None
        # Build CSV row with distinct columns
        metrics_no_collision = dict(metrics)
        if "score" in metrics_no_collision:
            # Remove original "score" to prevent ambiguity in CSV header
            metrics_no_collision.pop("score", None)
        row = {
            "iteration": it,
            "index": idx,
            "nfp": nfp_val,
            **metrics_no_collision,
            "evaluator_score": evaluator_score,
            "agg_score": agg_s,
            "elapsed_ms": elapsed_ms,
        }
        if metrics_writer is None:
            # Initialize writer with current row's keys and create header
            fieldnames = list(row.keys())
            for col in forced_metric_columns:
                if col not in fieldnames:
                    fieldnames.append(col)
            metrics_writer = csv.DictWriter(
                metrics_f, fieldnames=fieldnames, extrasaction="ignore"
            )
            if not metrics_csv_path.exists() or metrics_csv_path.stat().st_size == 0:
                metrics_writer.writeheader()
        else:
            # Avoid ValueError if row contains fields not in writer.fieldnames
            row = {k: row.get(k) for k in metrics_writer.fieldnames}
        metrics_writer.writerow(row)
        if agg_s < best_score:
            best_score = agg_s
            best_payload = {
                "agg_score": agg_s,
                "score": agg_s,
                "evaluator_score": evaluator_score,
                "metrics": metrics_no_collision,
                "boundary": boundary,
                "nfp": nfp_val,
            }
        # Persist novelty record when requested
        if config.novelty_skip and novelty_db is not None and novelty_db_path is not None:
            try:
                novelty_db.add(_flatten_map(boundary), {"agg_score": float(agg_s)})
                novelty_db.save()
            except Exception:
                pass

    # Optional correction hook (eci_linear)
    hook: Optional[_CorrectionHook] = None
    if config.correction == "eci_linear" and config.constraints:
        hook = _build_eci_linear_hook(config.constraints)
    elif config.correction == "pcfm" and config.constraints:
        hook = _build_pcfm_hook(
            config.constraints,
            gn_iters=config.pcfm_gn_iters,
            damping=config.pcfm_damping,
            tol=config.pcfm_tol,
        )

    def maybe_correct(bnd: Dict[str, Any]) -> Dict[str, Any]:
        if hook is None:
            return bnd
        try:
            return hook(bnd)
        except Exception:
            return bnd

    def maybe_guard(bnd: Dict[str, Any]) -> Dict[str, Any]:
        if not config.guard_simple:
            return bnd
        try:
            b = dict(bnd)
            # Ensure required fields exist
            if not isinstance(b.get("r_cos"), list) or not isinstance(b.get("z_sin"), list):
                return bnd
            r_cos = [[float(x) for x in row] for row in b["r_cos"]]
            z_sin = [[float(x) for x in row] for row in b["z_sin"]]
            # Clamp base radius term (r_cos[0][4]) if available
            i0, j0 = 0, 4
            if len(r_cos) > i0 and len(r_cos[i0]) > j0:
                base = r_cos[i0][j0]
                base = max(0.3, min(2.5, base))
                r_cos[i0][j0] = base
            # Clamp helical amplitudes around (1,5) if present
            ih, jh = 1, 5
            cap = 0.08
            if len(r_cos) > ih and len(r_cos[ih]) > jh:
                r_cos[ih][jh] = max(-cap, min(cap, r_cos[ih][jh]))
            if len(z_sin) > ih and len(z_sin[ih]) > jh:
                z_sin[ih][jh] = max(-cap, min(cap, z_sin[ih][jh]))
            # Write back
            b["r_cos"] = r_cos
            b["z_sin"] = z_sin
            b["r_sin"] = None
            b["z_cos"] = None
            b["is_stellarator_symmetric"] = True
            return b
        except Exception:
            return bnd

    def maybe_guard_geo(bnd: Dict[str, Any]) -> Dict[str, Any]:
        if not config.guard_geo:
            return bnd
        try:
            b = dict(bnd)
            r_cos = [[float(x) for x in row] for row in b.get("r_cos", [])]
            z_sin = [[float(x) for x in row] for row in b.get("z_sin", [])]
            if not r_cos or not z_sin:
                return bnd
            ih, jh = 1, 5
            # Target: modest helical amps with ratio ~1 (magnitude)
            cap = 0.03
            if len(r_cos) > ih and len(r_cos[ih]) > jh and len(z_sin) > ih and len(z_sin[ih]) > jh:
                xr = r_cos[ih][jh]
                yz = z_sin[ih][jh]
                # Bring magnitudes closer and within cap
                mag = 0.5 * (abs(xr) + abs(yz))
                mag = min(mag, cap)
                # Preserve signs but align magnitudes
                r_cos[ih][jh] = -mag if xr <= 0 else mag
                z_sin[ih][jh] = mag if yz >= 0 else -mag
            b["r_cos"] = r_cos
            b["z_sin"] = z_sin
            b["r_sin"] = None
            b["z_cos"] = None
            b["is_stellarator_symmetric"] = True
            return b
        except Exception:
            return bnd

    surrogate_scorer = None
    if config.surrogate_screen:
        if config.surrogate_model is None:
            raise ValueError("surrogate_screen requires surrogate_model path")
        try:
            from ..surrogate.screen import SurrogateScreenError, load_scorer
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("surrogate screening unavailable") from exc
        try:
            surrogate_scorer = load_scorer(
                Path(config.surrogate_model),
                metadata_path=(
                    Path(config.surrogate_metadata)
                    if config.surrogate_metadata is not None
                    else None
                ),
            )
        except SurrogateScreenError as exc:  # pragma: no cover - import guard
            raise RuntimeError(str(exc)) from exc

    def _select_survivors(scores: List[float]) -> List[int]:
        if not scores:
            return []
        idxs = list(range(len(scores)))
        sorted_by_score = sorted(idxs, key=lambda i: scores[i])
        survivors: set[int]
        if config.surrogate_threshold is not None:
            thr = float(config.surrogate_threshold)
            survivors = {i for i in idxs if scores[i] <= thr}
        else:
            q = 0.5 if config.surrogate_quantile is None else float(config.surrogate_quantile)
            q = max(0.0, min(1.0, q))
            keep = max(1, int(round(q * len(sorted_by_score))))
            survivors = set(sorted_by_score[:keep])
        if config.surrogate_keep_max is not None:
            kmax = int(config.surrogate_keep_max)
            if kmax > 0:
                survivors &= set(sorted_by_score[:kmax])
            elif kmax == 0:
                survivors = set()
        if not survivors:
            survivors = {sorted_by_score[0]}
        return [i for i in idxs if i in survivors]

    def apply_surrogate(
        *,
        batch: List[Dict[str, Any]],
        seeds: List[int],
        iteration: int,
        indices: List[int],
    ) -> tuple[List[Dict[str, Any]], List[int], List[int], Dict[int, float]]:
        if surrogate_scorer is None or not batch:
            return batch, seeds, list(range(len(batch))), {}
        preds = surrogate_scorer.score_many(batch)
        survivors_idx = _select_survivors(preds)
        survivors_set = set(survivors_idx)
        filtered_scores = {i: preds[i] for i in range(len(preds)) if i not in survivors_set}
        for i, score in filtered_scores.items():
            metrics = {
                "feasible": False,
                "fail_reason": "filtered_surrogate",
                "source": "placeholder",
                "phase": "surrogate",
                "surrogate_score": float(score),
            }
            log_entry(iteration, indices[i], seeds[i], batch[i], metrics, float("inf"))
        survivors_batch = [batch[i] for i in survivors_idx]
        survivors_seeds = [seeds[i] for i in survivors_idx]
        return survivors_batch, survivors_seeds, survivors_idx, filtered_scores

    # Load initial seed boundaries if provided
    seed_boundaries: List[Dict[str, Any]] = []
    if config.init_seeds is not None and Path(config.init_seeds).exists():
        try:
            with Path(config.init_seeds).open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if (
                        isinstance(obj, dict)
                        and "boundary" in obj
                        and isinstance(obj["boundary"], dict)
                    ):
                        seed_boundaries.append(dict(obj["boundary"]))
                    elif isinstance(obj, dict):
                        seed_boundaries.append(dict(obj))
        except Exception:
            seed_boundaries = []

    # Prepare NFP selection (round-robin over provided list if any)
    nfp_values: list[int] = []
    try:
        if config.nfp_list:
            nfp_values = [int(x) for x in config.nfp_list if int(x) > 0]
    except Exception:
        nfp_values = []
    if not nfp_values:
        nfp_values = [int(config.nfp)]
    _nfp_rr_idx = 0

    def _next_nfp() -> int:
        nonlocal _nfp_rr_idx
        val = nfp_values[_nfp_rr_idx % len(nfp_values)]
        _nfp_rr_idx += 1
        return val

    # Random search
    def run_random() -> None:
        nonlocal completed
        it = 0
        idx = 0
        batch_size = 8 if config.max_workers > 1 else 1
        while completed < budget:
            # Prepare a batch of proposals
            seeds: List[int] = []
            batch: List[Dict[str, Any]] = []
            for _ in range(min(batch_size, budget - completed)):
                if seed_boundaries:
                    b = seed_boundaries.pop(0)
                    seed_val = (rng_seed + it * 10007 + idx * 7919) % (2**31 - 1)
                    try:
                        b = maybe_guard_geo(b)
                        b = maybe_guard(b)
                        if config.guard_geom_validate:
                            from ..eval.geometry import quick_geometry_validate

                            ok, reason = quick_geometry_validate(
                                b,
                                r0_min=float(config.guard_geom_r0_min),
                                r0_max=float(config.guard_geom_r0_max),
                                helical_ratio_max=float(config.guard_geom_helical_ratio_max),
                            )
                            if not ok:
                                log_entry(
                                    it,
                                    idx,
                                    seed_val,
                                    b,
                                    {
                                        "feasible": False,
                                        "fail_reason": "invalid_geometry",
                                        "source": "placeholder",
                                    },
                                    float("inf"),
                                )
                                completed += 1
                                idx += 1
                                continue
                        # Novelty skip (on seeds)
                        if config.novelty_skip:
                            # Check persisted DB first
                            if novelty_db is not None:
                                try:
                                    if not novelty_db.is_novel(
                                        _flatten_map(b), atol=float(config.novelty_eps), rtol=0.0
                                    ):
                                        log_entry(
                                            it,
                                            idx,
                                            seed_val,
                                            b,
                                            {
                                                "feasible": False,
                                                "fail_reason": "duplicate_novelty",
                                                "source": "placeholder",
                                            },
                                            float("inf"),
                                        )
                                        idx += 1
                                        continue
                                except Exception:
                                    pass
                            if not _is_novel_local(b, novelty_window):
                                log_entry(
                                    it,
                                    idx,
                                    seed_val,
                                    b,
                                    {
                                        "feasible": False,
                                        "fail_reason": "duplicate_novelty",
                                        "source": "placeholder",
                                    },
                                    float("inf"),
                                )
                                idx += 1
                                continue
                        validate_boundary(b)
                        b = maybe_correct(b)
                    except Exception:
                        pass
                else:
                    seed_val = (rng_seed + it * 10007 + idx * 7919) % (2**31 - 1)
                    if (config.seed_mode or "random").lower() == "near-axis":
                        b = sample_near_axis_qs(nfp=_next_nfp(), seed=seed_val)
                    else:
                        b = sample_random(nfp=_next_nfp(), seed=seed_val)
                    b = maybe_guard_geo(b)
                    b = maybe_guard(b)
                    if config.guard_geom_validate:
                        from ..eval.geometry import quick_geometry_validate

                        ok, reason = quick_geometry_validate(
                            b,
                            r0_min=float(config.guard_geom_r0_min),
                            r0_max=float(config.guard_geom_r0_max),
                            helical_ratio_max=float(config.guard_geom_helical_ratio_max),
                        )
                        if not ok:
                            log_entry(
                                it,
                                idx,
                                seed_val,
                                b,
                                {
                                    "feasible": False,
                                    "fail_reason": "invalid_geometry",
                                    "source": "placeholder",
                                },
                                float("inf"),
                            )
                            completed += 1
                            idx += 1
                            continue
                    # Novelty skip (on generated)
                    if config.novelty_skip:
                        if novelty_db is not None:
                            try:
                                if not novelty_db.is_novel(
                                    _flatten_map(b), atol=float(config.novelty_eps), rtol=0.0
                                ):
                                    log_entry(
                                        it,
                                        idx,
                                        seed_val,
                                        b,
                                        {
                                            "feasible": False,
                                            "fail_reason": "duplicate_novelty",
                                            "source": "placeholder",
                                        },
                                        float("inf"),
                                    )
                                    idx += 1
                                    continue
                            except Exception:
                                pass
                        if not _is_novel_local(b, novelty_window):
                            log_entry(
                                it,
                                idx,
                                seed_val,
                                b,
                                {
                                    "feasible": False,
                                    "fail_reason": "duplicate_novelty",
                                    "source": "placeholder",
                                },
                                float("inf"),
                            )
                            idx += 1
                            continue
                    validate_boundary(b)
                    b = maybe_correct(b)
                seeds.append(seed_val)
                batch.append(b)
                idx += 1
            if surrogate_scorer is not None and batch:
                base_idx = idx - len(batch)
                idxs = [base_idx + k for k in range(len(batch))]
                batch, seeds, _, _ = apply_surrogate(
                    batch=batch,
                    seeds=seeds,
                    iteration=it,
                    indices=idxs,
                )
                if completed >= budget:
                    break
                if not batch:
                    continue
            # Evaluate
            if not batch:
                break
            if config.max_workers > 1 or config.mf_proxy:
                from ..eval import forward_many

                results = forward_many(
                    batch,
                    max_workers=config.max_workers,
                    cache_dir=config.cache_dir,
                    prefer_vmec=config.use_physics,
                    use_real=config.use_physics,
                    problem=problem,
                    mf_proxy=bool(config.mf_proxy),
                    mf_threshold=(
                        float(config.mf_threshold) if config.mf_threshold is not None else None
                    ),
                    mf_quantile=(
                        float(config.mf_quantile) if config.mf_quantile is not None else None
                    ),
                    mf_max_high=(
                        int(config.mf_max_high) if config.mf_max_high is not None else None
                    ),
                )
                for j, (b, m) in enumerate(zip(batch, results)):
                    try:
                        # Prefer score from metrics when present (official evaluator)
                        agg_s = (
                            float(m["score"])
                            if "score" in m
                            else eval_score(m, problem=problem if config.use_physics else None)
                        )
                    except Exception:
                        continue
                    # Use elapsed_ms from metrics if provided by eval.forward_many
                    ems = None
                    try:
                        val_ms = m.get("elapsed_ms")
                        if isinstance(val_ms, (int, float)):
                            ems = float(val_ms)
                    except Exception:
                        ems = None
                    log_entry(it, j, seeds[j], b, m, agg_s, elapsed_ms=ems)
                    completed += 1
            else:
                for j, b in enumerate(batch):
                    try:
                        _t0 = time.perf_counter()
                        m = eval_forward(
                            b,
                            cache_dir=config.cache_dir,
                            prefer_vmec=config.use_physics,
                            use_real=config.use_physics,
                            problem=problem,
                        )
                        _t1 = time.perf_counter()
                        agg_s = (
                            float(m["score"])
                            if "score" in m
                            else eval_score(m, problem=problem if config.use_physics else None)
                        )
                    except Exception:
                        continue
                    log_entry(it, j, seeds[j], b, m, agg_s, elapsed_ms=(_t1 - _t0) * 1000.0)
                    completed += 1
            # Update novelty memory after each batch to include evaluated items
            if config.novelty_skip:
                from collections import deque as _deque  # shadow safe

                for bb in batch:
                    nfpv, vv = _flatten_vec(bb)
                    dq = novelty_window.setdefault(nfpv, _deque(maxlen=int(config.novelty_window)))
                    dq.append(vv)
            if idx % 100 == 0:
                it += 1

    # CMA-ES search (2D helical coefficients)
    def run_cmaes() -> None:
        nonlocal completed
        try:
            import cma
        except Exception:
            # Fallback to random if CMA-ES not available
            run_random()
            return

        x0 = [0.05, 0.05]
        sigma0 = 0.05
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"bounds": [-0.2, 0.2], "seed": rng_seed})
        it = 0
        while completed < budget and not es.stop():
            xs = es.ask()
            xs_eval: List[List[float]] = []
            scores: List[float] = []
            proposals: List[Dict[str, Any]] = []
            seeds_batch: List[int] = []
            xs_params: List[List[float]] = []
            idxs: List[int] = []
            for j, x in enumerate(xs):
                if completed >= budget:
                    break
                seed_val = (rng_seed + it * 10007 + j * 7919) % (2**31 - 1)
                if (config.seed_mode or "random").lower() == "near-axis":
                    b = sample_near_axis_qs(nfp=_next_nfp(), seed=seed_val)
                else:
                    b = sample_random(nfp=_next_nfp(), seed=seed_val)
                try:
                    b["r_cos"][1][5] = float(-abs(x[0]))
                    b["z_sin"][1][5] = float(abs(x[1]))
                    b = maybe_guard(b)
                    if config.guard_geom_validate:
                        from ..eval.geometry import quick_geometry_validate

                        ok, reason = quick_geometry_validate(
                            b,
                            r0_min=float(config.guard_geom_r0_min),
                            r0_max=float(config.guard_geom_r0_max),
                            helical_ratio_max=float(config.guard_geom_helical_ratio_max),
                        )
                        if not ok:
                            metrics = {
                                "feasible": False,
                                "fail_reason": "invalid_geometry",
                                "source": "placeholder",
                            }
                            log_entry(it, j, seed_val, b, metrics, float("inf"))
                            xs_eval.append(list(x))
                            scores.append(float("inf"))
                            completed += 1
                            continue
                    validate_boundary(b)
                    b = maybe_correct(b)
                except Exception:
                    metrics = {}
                    log_entry(it, j, seed_val, b, metrics, float("inf"))
                    xs_eval.append(list(x))
                    scores.append(float("inf"))
                    completed += 1
                    continue
                proposals.append(b)
                seeds_batch.append(seed_val)
                xs_params.append(list(x))
                idxs.append(j)

            if surrogate_scorer is not None and proposals and completed < budget:
                xs_params_all = list(xs_params)
                idxs_all = list(idxs)
                proposals, seeds_batch, survivors_idx, filtered_scores = apply_surrogate(
                    batch=proposals,
                    seeds=seeds_batch,
                    iteration=it,
                    indices=idxs,
                )
                for i in sorted(filtered_scores):
                    xs_eval.append(xs_params_all[i])
                    scores.append(float("inf"))
                xs_params = [xs_params_all[i] for i in survivors_idx]
                idxs = [idxs_all[i] for i in survivors_idx]
            if completed >= budget:
                if xs_eval:
                    es.tell(xs_eval, scores)
                break

            for local_idx, b in enumerate(proposals):
                if completed >= budget:
                    break
                seed_val = seeds_batch[local_idx]
                x_vec = xs_params[local_idx]
                idx_val = idxs[local_idx]
                _t0 = time.perf_counter()
                try:
                    metrics = eval_forward(
                        b,
                        cache_dir=config.cache_dir,
                        prefer_vmec=config.use_physics,
                        use_real=config.use_physics,
                        problem=problem,
                    )
                    raw_score = metrics.get("score")
                    if isinstance(raw_score, (int, float)):
                        s = float(raw_score)
                    else:
                        s = eval_score(metrics, problem=problem if config.use_physics else None)
                except Exception:
                    metrics = {}
                    s = float("inf")
                _t1 = time.perf_counter()
                log_entry(
                    it,
                    idx_val,
                    seed_val,
                    b,
                    metrics,
                    s,
                    elapsed_ms=(_t1 - _t0) * 1000.0,
                )
                xs_eval.append(list(x_vec))
                scores.append(s)
                completed += 1
            if xs_eval:
                es.tell(xs_eval, scores)
            it += 1

    if config.algo.lower() == "cmaes":
        run_cmaes()
    else:
        run_random()

    # Finalize artifacts
    proposals_f.close()
    metrics_f.close()
    best_json_path.write_text(json.dumps(best_payload, indent=2))

    # README
    readme_lines = [
        "# ConStelX Agent Run",
        "",
        "CLI:",
        (
            f"constelx agent run --nfp {config.nfp} "
            f"--budget {config.budget} "
            f"--algo {config.algo} "
            f"--seed {config.seed}"
        ),
        (f"Problem: {problem}" if config.use_physics else "Placeholder evaluator"),
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Environment",
        json.dumps(_gather_env_info(), indent=2),
    ]
    readme_path.write_text("\n".join(readme_lines))

    return out_dir
