"""Minimal agent loop utilities."""

# ruff: noqa: I001  (import-sorting suppressed for local grouping)

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml

from ..eval import forward as eval_forward, score as eval_score
from ..eval.boundary_param import sample_random, validate as validate_boundary


@dataclass(frozen=True)
class AgentConfig:
    nfp: int
    iterations: int
    population: int
    seed: int
    out_dir: Path


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


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


def run(config: AgentConfig) -> Path:
    """Run a tiny random-search agent loop and write artifacts.

    This is a minimal, deterministic implementation that samples random
    boundaries, evaluates metrics, aggregates a score, and tracks the best.
    It avoids optional deps (e.g., CMA-ES) to keep tests lightweight.

    Returns the output directory path.
    """

    out_dir = config.out_dir / _timestamp()
    _ensure_dir(out_dir)

    # config.yaml with env info, git SHA, and package versions
    cfg = asdict(config)
    cfg["out_dir"] = str(cfg["out_dir"])  # Path -> str for YAML/JSON compatibility
    conf = {
        "run": cfg,
        "env": _gather_env_info(),
        "git": {"sha": _git_sha()},
        "versions": _pkg_versions(["constelx", "numpy", "pandas", "constellaration"]),
    }
    (out_dir / "config.yaml").write_text(yaml.safe_dump(conf, sort_keys=False))

    proposals_path = out_dir / "proposals.jsonl"
    metrics_csv_path = out_dir / "metrics.csv"
    best_json_path = out_dir / "best.json"
    readme_path = out_dir / "README.md"

    proposals: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []

    rng_seed = int(config.seed)
    best_score = float("inf")
    best_payload: Dict[str, Any] = {}

    for it in range(config.iterations):
        for j in range(config.population):
            seed = (rng_seed + it * 10007 + j * 7919) % (2**31 - 1)
            boundary = sample_random(nfp=config.nfp, seed=seed)
            validate_boundary(boundary)
            try:
                metrics = eval_forward(boundary)
                s = eval_score(metrics)
            except Exception:
                # Skip invalid evaluations; keep stub minimal
                continue

            prop = {"iteration": it, "index": j, "seed": seed, "boundary": boundary}
            proposals.append(prop)
            row = {"iteration": it, "index": j, **metrics, "score": s}
            metrics_rows.append(row)

            if s < best_score:
                best_score = s
                best_payload = {"score": s, "metrics": metrics, "boundary": boundary}

    # Write artifacts
    _write_jsonl(proposals_path, proposals)
    _write_csv(metrics_csv_path, metrics_rows)
    best_json_path.write_text(json.dumps(best_payload, indent=2))

    # README
    readme_lines = [
        "# ConStelX Agent Run",
        "",
        "CLI:",
        (
            f"constelx agent run --nfp {config.nfp} "
            f"--iterations {config.iterations} "
            f"--population {config.population} "
            f"--seed {config.seed}"
        ),
        f"Created: {datetime.now(UTC).isoformat()}",
        "",
        "## Environment",
        json.dumps(_gather_env_info(), indent=2),
    ]
    readme_path.write_text("\n".join(readme_lines))

    return out_dir
