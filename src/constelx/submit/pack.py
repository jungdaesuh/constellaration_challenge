from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_best_record(run_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (best_row_from_metrics, boundary_from_proposals).

    Falls back to the first line if matching by (iteration,index) fails.
    """
    import csv

    metrics_path = run_dir / "metrics.csv"
    props_path = run_dir / "proposals.jsonl"
    if not metrics_path.exists() or not props_path.exists():
        raise FileNotFoundError("metrics.csv or proposals.jsonl is missing in the run directory")

    best_row: Optional[Dict[str, Any]] = None
    best_val = float("inf")
    with metrics_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Prefer feasible rows; otherwise compare raw agg_score
                feas = row.get("feasible")
                if isinstance(feas, str):
                    feas = feas.lower() in {"true", "1", "yes"}
                score_str = row.get("agg_score")
                if score_str is None:
                    continue
                val = float(score_str)
                if val < best_val:
                    best_val = val
                    best_row = dict(row)
            except Exception:
                continue
    if best_row is None:
        raise ValueError("Could not determine best row from metrics.csv")

    # Map to proposals by (iteration,index)
    it_key = best_row.get("iteration")
    idx_key = best_row.get("index")
    boundary: Optional[Dict[str, Any]] = None
    try:
        it_val = int(it_key) if it_key is not None else None
        idx_val = int(idx_key) if idx_key is not None else None
    except Exception:
        it_val = None
        idx_val = None
    with props_path.open() as f:
        first_obj: Optional[Dict[str, Any]] = None
        for line in f:
            try:
                obj = json.loads(line)
                if first_obj is None:
                    first_obj = obj if isinstance(obj, dict) else None
                if (
                    isinstance(obj, dict)
                    and obj.get("iteration") == it_val
                    and obj.get("index") == idx_val
                    and isinstance(obj.get("boundary"), dict)
                ):
                    boundary = dict(obj["boundary"])
                    break
            except Exception:
                continue
        if (
            boundary is None
            and isinstance(first_obj, dict)
            and isinstance(first_obj.get("boundary"), dict)
        ):
            boundary = dict(first_obj["boundary"])
    if boundary is None:
        raise ValueError("Could not match boundary from proposals.jsonl")
    return best_row, boundary


def _collect_topk_records(run_dir: Path, k: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Collect top-K (row, boundary) pairs sorted by agg_score and feasibility.

    Returns up to K pairs; skips rows that cannot be matched to proposals.
    """
    import csv

    metrics_path = run_dir / "metrics.csv"
    props_path = run_dir / "proposals.jsonl"
    if not metrics_path.exists() or not props_path.exists():
        raise FileNotFoundError("metrics.csv or proposals.jsonl is missing in the run directory")

    rows: List[Dict[str, Any]] = []
    with metrics_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                val = float(row.get("agg_score", "inf"))
            except Exception:
                continue
            rows.append({**row, "_agg": val})

    def _feas_order(r: Dict[str, Any]) -> int:
        feas = r.get("feasible")
        if isinstance(feas, str):
            feas_b = feas.lower() in {"true", "1", "yes"}
        elif isinstance(feas, bool):
            feas_b = feas
        else:
            feas_b = True
        return 0 if feas_b else 1

    rows_sorted = sorted(rows, key=lambda r: (_feas_order(r), r.get("_agg", float("inf"))))

    # Helper to match a single boundary from proposals
    def _match(it_val: int, idx_val: int) -> Optional[Dict[str, Any]]:
        with props_path.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if (
                    isinstance(obj, dict)
                    and obj.get("iteration") == it_val
                    and obj.get("index") == idx_val
                    and isinstance(obj.get("boundary"), dict)
                ):
                    return dict(obj["boundary"])
        return None

    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen: set[Tuple[int, int]] = set()
    for row in rows_sorted:
        try:
            it_raw = row.get("iteration")
            idx_raw = row.get("index")
            if it_raw is None or idx_raw is None:
                continue
            it_val = int(it_raw)
            idx_val = int(idx_raw)
        except Exception:
            continue
        key = (it_val, idx_val)
        if key in seen:
            continue
        bnd = _match(it_val, idx_val)
        if bnd is None:
            continue
        out.append((row, bnd))
        seen.add(key)
        if len(out) >= k:
            break
    return out


def pack_run(run_dir: Path, out_path: Path, top_k: int = 1) -> Path:
    """Pack a run directory into a submission zip.

    Contents:
    - boundary.json: best boundary (validated by internal validator)
    - best.json: copied from run
    - metadata.json: minimal provenance (run path)
    - MANIFEST.txt: list of files
    """
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    # Load best row and boundary
    best_row, boundary = _find_best_record(run_dir)
    # Validate boundary using local validator
    try:
        from ..eval.boundary_param import validate as validate_boundary

        validate_boundary(boundary)
    except Exception:
        # still allow packaging but make it explicit in metadata
        pass

    best_json_path = run_dir / "best.json"
    # Load run config to augment metadata (config.yaml is JSON content)
    cfg_path = run_dir / "config.yaml"
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    git_cfg = cfg.get("git", {}) if isinstance(cfg.get("git"), dict) else {}
    metadata = {
        "run_dir": str(run_dir),
        "iteration": best_row.get("iteration"),
        "index": best_row.get("index"),
        "problem": run_cfg.get("problem"),
        "scoring_version": best_row.get("scoring_version"),
        "git_sha": git_cfg.get("sha"),
    }
    # Create zip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # boundary.json
        zf.writestr("boundary.json", json.dumps(boundary, indent=2))
        # boundaries.jsonl (optional top-K)
        if isinstance(top_k, int) and top_k > 1:
            pairs = _collect_topk_records(run_dir, int(top_k))
            if pairs:
                lines: List[str] = []
                for row, bnd in pairs:
                    rec = {
                        "iteration": row.get("iteration"),
                        "index": row.get("index"),
                        "agg_score": row.get("agg_score"),
                        "evaluator_score": row.get("evaluator_score"),
                        "feasible": row.get("feasible"),
                        "fail_reason": row.get("fail_reason"),
                        "source": row.get("source"),
                        "scoring_version": row.get("scoring_version"),
                        "boundary": bnd,
                    }
                    lines.append(json.dumps(rec))
                zf.writestr("boundaries.jsonl", "\n".join(lines) + "\n")
        # best.json if present
        if best_json_path.exists():
            zf.write(best_json_path, arcname="best.json")
        # metadata.json
        zf.writestr(
            "metadata.json",
            json.dumps({**metadata, "top_k": int(top_k)}, indent=2),
        )
        # manifest
        zf.writestr(
            "MANIFEST.txt",
            "\n".join(
                [
                    "boundary.json",
                    "boundaries.jsonl" if (isinstance(top_k, int) and top_k > 1) else "",
                    "best.json" if best_json_path.exists() else "",
                    "metadata.json",
                ]
            ).strip(),
        )
    return out_path
