from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .data.dataset import fetch_dataset, save_subset
from .physics.constel_api import example_boundary, evaluate_boundary
from .optim.evolution import run_cma_es_baseline
from .surrogate.train import train_simple_mlp

app = typer.Typer(help="ConstelX CLI — ConStellaration starter tools")
console = Console()


@app.command()
def version():
    """Print version."""
    from . import __version__

    console.print(f"constelx {__version__}")


@app.command()
def hello(name: str = "world"):
    """Sanity check."""
    console.print(f"Hello, {name} 👋")


# -------------------- DATA --------------------
data_app = typer.Typer(help="Data access and preprocessing")
app.add_typer(data_app, name="data")


@data_app.command("fetch")
def data_fetch(
    cache_dir: Path = typer.Option(Path("data/cache"), help="Where to store HF cache/parquets."),
    nfp: Optional[int] = typer.Option(None, help="Filter by number of field periods (boundary.n_field_periods)."),
    limit: Optional[int] = typer.Option(1000, help="Take first N examples for quick experiments."),
):
    ds = fetch_dataset()
    if nfp is not None:
        ds = ds.filter(lambda x: x == nfp, input_columns=["boundary.n_field_periods"], num_proc=4)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    out = save_subset(ds, cache_dir)
    console.print(f"Saved subset to: [bold]{out}[/bold]")


# -------------------- EVAL --------------------
eval_app = typer.Typer(help="Run physics/evaluator metrics via constellaration")
app.add_typer(eval_app, name="eval")


@eval_app.command("forward")
def eval_forward(
    boundary_json: Optional[Path] = typer.Option(None, help="Path to a JSON boundary (SurfaceRZFourier)."),
    example: bool = typer.Option(False, "--example", help="Use a synthetic example."),
):
    if example:
        b = example_boundary()
    else:
        if boundary_json is None:
            raise typer.BadParameter("Provide --boundary-json or use --example")
        b = json.loads(boundary_json.read_text())

    result = evaluate_boundary(b)
    table = Table(title="Forward metrics")
    table.add_column("metric"); table.add_column("value")
    for k, v in result.items():
        table.add_row(k, f"{v:.6g}" if isinstance(v, (int, float)) else str(v))
    console.print(table)


# -------------------- OPTIMIZATION --------------------
opt_app = typer.Typer(help="Optimization baselines")
app.add_typer(opt_app, name="opt")


@opt_app.command("baseline")
def opt_baseline(
    steps: int = 50,
    seed: int = 0,
    algo: str = typer.Option("cma-es", help="One of: cma-es"),
):
    random.seed(seed)
    if algo == "cma-es":
        best = run_cma_es_baseline(steps=steps)
        console.print(f"Best score: {best}")
    else:
        raise typer.BadParameter(f"Unknown algo: {algo}")


# -------------------- SURROGATE --------------------
sur_app = typer.Typer(help="Train simple surrogate models")
app.add_typer(sur_app, name="surrogate")


@sur_app.command("train")
def surrogate_train(
    cache_dir: Path = Path("data/cache"),
    output_dir: Path = Path("outputs/surrogates/mlp"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_simple_mlp(cache_dir=cache_dir, output_dir=output_dir)
    console.print(f"Saved surrogate to {output_dir}")


# -------------------- AGENT --------------------
agent_app = typer.Typer(help="Multi-step agent loop (propose → simulate → select → refine)")
app.add_typer(agent_app, name="agent")


@agent_app.command("run")
def agent_run(
    iterations: int = 3,
    population: int = 8,
):
    """Minimal loop stub to be extended by the coding agent."""
    console.print(f"[bold]Agent[/bold] starting: iterations={iterations}, population={population}")
    console.print("TODO: implement propose/simulate/select/refine using constellaration forward model.")


if __name__ == "__main__":
    app()
