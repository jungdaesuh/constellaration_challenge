"""Minimal agent loop utilities."""

# ruff: noqa: I001  (import-sorting suppressed for local grouping)

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Callable

from ..eval import forward as eval_forward, score as eval_score
from ..eval.boundary_param import sample_random, validate as validate_boundary


@dataclass(frozen=True)
class AgentConfig:
    nfp: int
    seed: int
    out_dir: Path
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

    # Resume bookkeeping
    completed = 0
    best_score = float("inf")
    best_payload: Dict[str, Any] = {}
    if proposals_path.exists():
        # Count existing proposals
        completed = sum(1 for _ in proposals_path.open())
    if best_json_path.exists():
        try:
            best_payload = json.loads(best_json_path.read_text())
            if isinstance(best_payload.get("score"), (int, float)):
                best_score = float(best_payload["score"])
        except Exception:
            best_payload = {}
            best_score = float("inf")

    # Open for append
    proposals_f = proposals_path.open("a")
    metrics_f = metrics_csv_path.open("a", newline="")
    metrics_writer: csv.DictWriter[str] | None = None
    if metrics_csv_path.stat().st_size == 0:
        metrics_writer = None
    else:
        # We don't know columns; write new rows with full keys but rely on initial header
        # For simplicity, when appending we re-create writer after first row produced.
        metrics_writer = None

    rng_seed = int(config.seed)
    budget = int(config.budget)

    def log_entry(
        it: int,
        idx: int,
        seed_val: int,
        boundary: Dict[str, Any],
        metrics: Dict[str, Any],
        s: float,
    ) -> None:
        nonlocal best_score, best_payload, metrics_writer
        prop = {"iteration": it, "index": idx, "seed": seed_val, "boundary": boundary}
        proposals_f.write(json.dumps(prop) + "\n")
        row = {"iteration": it, "index": idx, **metrics, "score": s}
        if metrics_writer is None:
            # Initialize writer with current row's keys
            metrics_writer = csv.DictWriter(metrics_f, fieldnames=list(row.keys()))
            if metrics_csv_path.stat().st_size == 0:
                metrics_writer.writeheader()
        metrics_writer.writerow(row)
        if s < best_score:
            best_score = s
            best_payload = {"score": s, "metrics": metrics, "boundary": boundary}

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
                seed_val = (rng_seed + it * 10007 + idx * 7919) % (2**31 - 1)
                b = sample_random(nfp=config.nfp, seed=seed_val)
                validate_boundary(b)
                b = maybe_correct(b)
                seeds.append(seed_val)
                batch.append(b)
                idx += 1
            # Evaluate
            if not batch:
                break
            if config.max_workers > 1:
                from ..eval import forward_many

                results = forward_many(
                    batch,
                    max_workers=config.max_workers,
                    cache_dir=config.cache_dir,
                    prefer_vmec=config.use_physics,
                )
                for j, (b, m) in enumerate(zip(batch, results)):
                    try:
                        s = eval_score(m)
                    except Exception:
                        continue
                    log_entry(it, j, seeds[j], b, m, s)
                    completed += 1
            else:
                for j, b in enumerate(batch):
                    try:
                        m = eval_forward(
                            b,
                            cache_dir=config.cache_dir,
                            prefer_vmec=config.use_physics,
                        )
                        s = eval_score(m)
                    except Exception:
                        continue
                    log_entry(it, j, seeds[j], b, m, s)
                    completed += 1
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
            for j, x in enumerate(xs):
                if completed >= budget:
                    break
                seed_val = (rng_seed + it * 10007 + j * 7919) % (2**31 - 1)
                b = sample_random(nfp=config.nfp, seed=seed_val)
                # Override two helical coeffs deterministically from CMA-ES params
                try:
                    b["r_cos"][1][5] = float(-abs(x[0]))
                    b["z_sin"][1][5] = float(abs(x[1]))
                    validate_boundary(b)
                    b = maybe_correct(b)
                    metrics = eval_forward(
                        b, cache_dir=config.cache_dir, prefer_vmec=config.use_physics
                    )
                    s = eval_score(metrics)
                except Exception:
                    # Skip invalid points; penalize in CMA-ES
                    s = float("inf")
                    metrics = {}
                log_entry(it, j, seed_val, b, metrics, s)
                xs_eval.append(list(x))
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
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Environment",
        json.dumps(_gather_env_info(), indent=2),
    ]
    readme_path.write_text("\n".join(readme_lines))

    return out_dir
