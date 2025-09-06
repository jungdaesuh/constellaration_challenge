from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

"""CLI entrypoints for constelx.

Imports of internal subpackages are deferred to command bodies to avoid
import-time coupling and keep simple commands lightweight.
"""

app = typer.Typer(help="ConstelX CLI — ConStellaration starter tools")
console = Console()


@app.command()
def version() -> None:
    """Print version."""
    from . import __version__

    console.print(f"constelx {__version__}")


@app.command()
def hello(name: str = "world") -> None:
    """Sanity check."""
    console.print(f"Hello, {name} 👋")


# -------------------- DATA --------------------
data_app = typer.Typer(help="Data access and preprocessing")
app.add_typer(data_app, name="data")


@data_app.command("fetch")
def data_fetch(
    cache_dir: Path = typer.Option(Path("data/cache"), help="Where to store HF cache/parquets."),
    nfp: Optional[int] = typer.Option(
        None, help="Filter by number of field periods (boundary.n_field_periods)."
    ),
    limit: Optional[int] = typer.Option(1000, help="Take first N examples for quick experiments."),
    source: str = typer.Option("synthetic", help="Data source: synthetic|hf"),
) -> None:
    if source == "hf":
        try:
            from .data.constellaration import (
                filter_nfp as hf_filter,
            )
            from .data.constellaration import (
                load as hf_load,
            )
            from .data.constellaration import (
                to_parquet as hf_to_parquet,
            )
        except Exception as e:
            raise typer.BadParameter(f"HF dataset path unavailable: {e}")
        ds = hf_load()
        if nfp is not None:
            ds = hf_filter(ds, int(nfp))
        if limit is not None:
            ds = ds.select(range(min(int(limit), len(ds))))
        out = hf_to_parquet(ds, cache_dir / "subset.parquet")
        console.print(f"Saved HF subset to: [bold]{out}[/bold]")
        return

    # synthetic path
    from .data.dataset import fetch_dataset, save_subset

    ds = fetch_dataset()
    if nfp is not None:
        # Use single-process filter to avoid fork issues in constrained environments/tests
        ds = ds.filter(lambda x: x == nfp, input_columns=["boundary.n_field_periods"], num_proc=1)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    out = save_subset(ds, cache_dir)
    console.print(f"Saved subset to: [bold]{out}[/bold]")


@data_app.command("seeds")
def data_seeds(
    out_path: Path = typer.Option(
        Path("data/seeds.jsonl"),
        "--out",
        "--out-path",
        help="Output seeds JSONL path",
    ),
    nfp: int = typer.Option(3, help="Filter boundaries by NFP"),
    k: int = typer.Option(64, help="Number of seeds to write"),
) -> None:
    """Create a seeds.jsonl from the HF dataset with {boundary: {...}} records."""
    try:
        from .data.constellaration import (
            filter_nfp as hf_filter,
        )
        from .data.constellaration import (
            load as hf_load,
        )
        from .data.constellaration import (
            make_seeds_jsonl,
        )
    except Exception as e:
        raise typer.BadParameter(f"HF dataset unavailable: {e}")
    ds = hf_filter(hf_load(), int(nfp))
    out = make_seeds_jsonl(ds, out_path, k=int(k))
    console.print(f"Wrote seeds to: [bold]{out}[/bold]")


# -------------------- EVAL --------------------
eval_app = typer.Typer(help="Run physics/evaluator metrics via constellaration")
app.add_typer(eval_app, name="eval")


@eval_app.command("forward")
def eval_forward(
    boundary_json: Optional[Path] = typer.Option(
        None, help="Path to a JSON boundary (SurfaceRZFourier)."
    ),
    example: bool = typer.Option(False, "--example", help="Use a synthetic example."),
    random_boundary: bool = typer.Option(
        False, "--random", help="Use a random sampled boundary (deterministic with --seed)."
    ),
    nfp: int = typer.Option(3, help="NFP used with --random."),
    seed: int = typer.Option(0, help="Seed used with --random."),
    cache_dir: Optional[Path] = typer.Option(None, help="Optional cache directory for metrics."),
    use_physics: bool = typer.Option(
        False,
        "--use-physics",
        "--use-real",
        help="Use real evaluator if available.",
    ),
    problem: str = typer.Option("p1", help="Problem id for scoring/metrics (e.g., p1/p2/p3)."),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON metrics."),
) -> None:
    if sum([bool(example), boundary_json is not None, bool(random_boundary)]) != 1:
        raise typer.BadParameter("Choose exactly one of --example, --boundary-json, or --random")

    if example:
        from .physics.constel_api import example_boundary

        b = example_boundary()
    elif random_boundary:
        from .eval.boundary_param import sample_random
        from .eval.boundary_param import validate as validate_boundary

        b = sample_random(nfp=nfp, seed=seed)
        validate_boundary(b)
    else:
        assert boundary_json is not None
        b = json.loads(boundary_json.read_text())
        from .eval.boundary_param import validate as validate_boundary

        validate_boundary(b)

    from .eval import forward as eval_forward_metrics

    result = eval_forward_metrics(b, cache_dir=cache_dir, use_real=use_physics, problem=problem)
    if json_out:
        console.print_json(data=result)
        return
    table = Table(title="Forward metrics")
    table.add_column("metric")
    table.add_column("value")
    for k, v in result.items():
        table.add_row(k, f"{v:.6g}" if isinstance(v, (int, float)) else str(v))
    console.print(table)


@eval_app.command("score")
def eval_score(
    metrics_json: Optional[Path] = typer.Option(
        None, "--metrics-json", help="Path to a JSON file containing a metrics dict."
    ),
    metrics_file: Optional[Path] = typer.Option(
        None, "--metrics-file", help="Path to a CSV file with metric columns."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Optional output CSV path when using --metrics-file."
    ),
    problem: str = typer.Option("p1", help="Problem id for scoring (e.g., p1/p2/p3)."),
) -> None:
    """Aggregate a scalar score from a metrics JSON file.

    The JSON must contain a flat dict of metric name to numeric value. Non-numeric
    entries are ignored. If any numeric value is NaN, the score is +inf.
    """
    if (metrics_json is None) == (metrics_file is None):
        raise typer.BadParameter("Provide exactly one of --metrics-json or --metrics-file")

    if metrics_json is not None:
        from .eval import score as eval_score_agg

        metrics = json.loads(Path(metrics_json).read_text())
        # Prefer official score passthrough if present
        has_score = (
            isinstance(metrics, dict)
            and "score" in metrics
            and isinstance(metrics["score"], (int, float))
        )
        if has_score:
            value = float(metrics["score"])
        else:
            value = eval_score_agg(metrics, problem=problem)
        console.print(f"score = {value}")
        return

    # CSV mode
    import pandas as pd

    df = pd.read_csv(metrics_file)

    def row_score(row: Any) -> float:
        # Convert row (Series) to plain dict for aggregator
        from .eval import score as eval_score_agg

        # Avoid double-counting an existing 'score' column on re-runs
        cols = [c for c in df.columns if c != "score"]
        return eval_score_agg({k: row[k] for k in cols}, problem=problem)

    df["score"] = df.apply(row_score, axis=1)
    if output is None:
        console.print(df.to_string(index=False))
    else:
        df.to_csv(output, index=False)
        console.print(f"Wrote: {output}")


# -------------------- OPTIMIZATION --------------------
opt_app = typer.Typer(help="Optimization baselines")
app.add_typer(opt_app, name="opt")


@opt_app.command("baseline")
def opt_baseline(
    steps: int = 50,
    seed: int = 0,
    algo: str = typer.Option("cma-es", help="One of: cma-es"),
) -> None:
    random.seed(seed)
    if algo == "cma-es":
        from .optim.evolution import run_cma_es_baseline

        best = run_cma_es_baseline(steps=steps)
        console.print(f"Best score: {best}")
    else:
        raise typer.BadParameter(f"Unknown algo: {algo}")


@opt_app.command("cmaes")
def opt_cmaes(
    nfp: int = typer.Option(3, help="Boundary NFP for boundary-mode optimization."),
    budget: int = typer.Option(50, help="Number of CMA-ES iterations."),
    seed: int = typer.Option(0, help="Random seed."),
    toy: bool = typer.Option(False, help="Use toy sphere objective instead of boundary."),
) -> None:
    """Run a CMA-ES optimization on a toy sphere objective for a quick smoke test."""
    try:
        from .optim.cmaes import optimize
    except Exception as e:  # pragma: no cover - import-time error
        raise typer.BadParameter(str(e))

    if toy:
        from typing import Sequence

        def sphere(x: Sequence[float]) -> float:
            return float(sum(v * v for v in x))

        x0 = [0.5, 0.5]
        best_x, hist = optimize(
            sphere, x0=x0, bounds=(-1.0, 1.0), budget=budget, sigma0=0.3, seed=seed
        )
        console.print(f"Best x: {best_x}\nBest score: {min(hist) if hist else float('inf')}")
        return

    # Boundary-mode objective using placeholder evaluator
    from .physics.constel_api import example_boundary  # noqa: I001
    from .eval import forward as eval_forward_metrics  # noqa: I001
    from .eval.boundary_param import validate as validate_boundary  # noqa: I001

    from typing import Sequence

    def make_boundary(x: Sequence[float]) -> dict[str, Any]:
        b = example_boundary()
        b["n_field_periods"] = int(nfp)
        b["r_cos"][1][5] = float(-abs(x[0]))
        b["z_sin"][1][5] = float(abs(x[1]))
        validate_boundary(b)
        return b

    def objective(x: Sequence[float]) -> float:
        from .eval import score as eval_score_agg

        m = eval_forward_metrics(make_boundary(x))
        # Use shared aggregator for consistency with other commands
        return float(eval_score_agg(m))

    x0 = [0.05, 0.05]
    best_x, hist = optimize(
        objective, x0=x0, bounds=(-0.2, 0.2), budget=budget, sigma0=0.05, seed=seed
    )
    console.print(f"Best x: {best_x}\nBest score: {min(hist) if hist else float('inf')}")


# -------------------- SURROGATE --------------------
sur_app = typer.Typer(help="Train simple surrogate models")
app.add_typer(sur_app, name="surrogate")


@sur_app.command("train")
def surrogate_train(
    cache_dir: Path = Path("data/cache"),
    output_dir: Path = typer.Option(
        Path("outputs/surrogates/mlp"), "--out-dir", "--output-dir", help="Model output directory"
    ),
    use_pbfm: bool = typer.Option(False, help="Use PBFM conflict-free loss combination"),
) -> None:
    from .surrogate.train import train_simple_mlp

    output_dir.mkdir(parents=True, exist_ok=True)
    train_simple_mlp(cache_dir=cache_dir, output_dir=output_dir, use_pbfm=use_pbfm)
    console.print(f"Saved surrogate to {output_dir}")


# -------------------- AGENT --------------------
agent_app = typer.Typer(help="Multi-step agent loop (propose → simulate → select → refine)")
app.add_typer(agent_app, name="agent")


@agent_app.command("run")
def agent_run(
    nfp: int = typer.Option(3, help="Number of field periods for random boundaries."),
    budget: int = typer.Option(50, help="Total number of evaluations to run."),
    algo: str = typer.Option("random", help="Optimization algorithm: random or cmaes."),
    seed: int = typer.Option(0, help="Global seed for reproducibility."),
    runs_dir: Path = typer.Option(Path("runs"), help="Directory to store artifacts."),
    resume: Optional[Path] = typer.Option(None, help="Resume from an existing run directory."),
    max_workers: int = typer.Option(1, help="Parallel evaluator workers for agent evals."),
    cache_dir: Optional[Path] = typer.Option(None, help="Cache directory for agent evals."),
    use_physics: bool = typer.Option(
        False,
        "--use-physics",
        help="Prefer VMEC validation if constellaration is installed; fallback otherwise.",
    ),
    problem: Optional[str] = typer.Option(
        None, help="Problem id when using --use-physics (p1|p2|p3)."
    ),
    init_seeds: Optional[Path] = typer.Option(
        None, help="JSONL of initial boundary seeds to evaluate first."
    ),
    # PCFM tuning (applies when --correction pcfm)
    pcfm_gn_iters: Optional[int] = typer.Option(
        None, help="PCFM Gauss–Newton iterations (override constraints file)."
    ),
    pcfm_damping: Optional[float] = typer.Option(
        None, help="PCFM initial damping lambda (override constraints file)."
    ),
    pcfm_tol: Optional[float] = typer.Option(
        None, help="PCFM residual tolerance (override constraints file)."
    ),
    correction: Optional[str] = typer.Option(
        None,
        help="Optional correction hook to apply to boundaries (e.g., 'eci_linear').",
    ),
    constraints_file: Optional[Path] = typer.Option(
        None,
        help=(
            "JSON file with constraints for correction hooks.\n"
            "- eci_linear: [{rhs: float, coeffs: [{field,i,j,c}]}]\n"
            "- pcfm: [{type:'norm_eq', radius:float, terms:[{field,i,j,w}]}]"
        ),
    ),
) -> None:
    from .agents.simple_agent import AgentConfig, run as run_agent  # noqa: I001

    runs_dir.mkdir(parents=True, exist_ok=True)
    if use_physics and not problem:
        raise typer.BadParameter("--problem is required when --use-physics is set (p1|p2|p3)")
    # Load constraints if provided
    constraints: list[dict[str, Any]] | None = None
    # Allow constraints JSON to be a dict with overrides {constraints:[...], gn_iters, damping, tol}
    if constraints_file is not None:
        data = json.loads(constraints_file.read_text())
        if isinstance(data, dict) and "constraints" in data:
            raw = data.get("constraints")
            # pick up optional overrides if not set via CLI
            if pcfm_gn_iters is None and isinstance(data.get("gn_iters"), int):
                pcfm_gn_iters = int(data["gn_iters"])  # noqa: PLW2901 (reassign ok)
            if pcfm_damping is None and isinstance(data.get("damping"), (int, float)):
                pcfm_damping = float(data["damping"])  # noqa: PLW2901 (reassign ok)
            if pcfm_tol is None and isinstance(data.get("tol"), (int, float)):
                pcfm_tol = float(data["tol"])  # noqa: PLW2901 (reassign ok)
        else:
            raw = data
        if not isinstance(raw, list):
            raise typer.BadParameter(
                "constraints file must contain a list under 'constraints' or be a list"
            )
        constraints = [dict(x) for x in raw]

    out = run_agent(
        AgentConfig(
            nfp=nfp,
            seed=seed,
            out_dir=runs_dir,
            algo=algo,
            budget=budget,
            resume=resume,
            max_workers=max_workers,
            cache_dir=cache_dir,
            correction=correction,
            constraints=constraints,
            use_physics=use_physics,
            problem=problem,
            init_seeds=init_seeds,
            pcfm_gn_iters=pcfm_gn_iters,
            pcfm_damping=pcfm_damping,
            pcfm_tol=pcfm_tol,
        )
    )
    console.print(f"Run complete. Artifacts in: [bold]{out}[/bold]")


if __name__ == "__main__":
    app()
