from __future__ import annotations

# ruff: noqa: I001  (import-sorting suppressed for local grouping inside commands)

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

# Load environment variables from a .env file if python-dotenv is available.
try:  # optional dependency; safe no-op if missing
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    pass


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

    count = int(limit) if limit is not None else 128
    ds = fetch_dataset(count=count, nfp=int(nfp) if nfp is not None else 3)
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
    near_axis: bool = typer.Option(
        False,
        "--near-axis",
        help="Use a near-axis QS/QI-friendly seed (deterministic with --seed).",
    ),
    nfp: int = typer.Option(3, help="NFP used with --random."),
    seed: int = typer.Option(0, help="Seed used with --random."),
    cache_dir: Path = typer.Option(
        Path(".cache/eval"), help="Cache directory for metrics (diskcache/json)."
    ),
    use_physics: bool = typer.Option(
        False,
        "--use-physics",
        "--use-real",
        help="Use real evaluator if available.",
    ),
    problem: str = typer.Option("p1", help="Problem id for scoring/metrics (e.g., p1/p2/p3)."),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON metrics."),
    vmec_level: Optional[str] = typer.Option(
        None,
        help="VMEC resolution level (auto|low|medium|high). Defaults to env/auto.",
    ),
    vmec_hot_restart: Optional[bool] = typer.Option(
        None,
        "--vmec-hot-restart/--no-vmec-hot-restart",
        help="Enable VMEC hot restart when available (defaults via env).",
    ),
    vmec_restart_key: Optional[str] = typer.Option(
        None,
        help="Optional VMEC hot-restart key to reuse cached states.",
    ),
) -> None:
    if sum([bool(example), boundary_json is not None, bool(random_boundary), bool(near_axis)]) != 1:
        raise typer.BadParameter(
            "Choose exactly one of --example, --boundary-json, --random, or --near-axis"
        )

    if example:
        from .physics.constel_api import example_boundary

        b = example_boundary()
    elif random_boundary or near_axis:
        from .eval.boundary_param import (
            sample_random,
            sample_near_axis_qs,
            validate as validate_boundary,
        )

        if near_axis:
            b = sample_near_axis_qs(nfp=nfp, seed=seed)
        else:
            b = sample_random(nfp=nfp, seed=seed)
        validate_boundary(b)
    else:
        assert boundary_json is not None
        b = json.loads(boundary_json.read_text())
        from .eval.boundary_param import validate as validate_boundary

        validate_boundary(b)

    from .eval import forward as eval_forward_metrics

    result = eval_forward_metrics(
        b,
        cache_dir=cache_dir,
        use_real=use_physics,
        problem=problem,
        vmec_level=vmec_level,
        vmec_hot_restart=vmec_hot_restart,
        vmec_restart_key=vmec_restart_key,
    )
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
        from .problems import get_spec

        metrics = json.loads(Path(metrics_json).read_text())
        # Prefer official score passthrough if present
        has_score = (
            isinstance(metrics, dict)
            and "score" in metrics
            and isinstance(metrics["score"], (int, float))
        )
        spec = get_spec(problem)
        if isinstance(metrics, dict) and spec is not None:
            missing = spec.missing_keys(metrics)
            if missing:
                msg = (
                    f"[yellow]Warning:[/yellow] metrics missing expected keys for {spec.pid}: "
                    f"{missing}"
                )
                console.print(msg)
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


@eval_app.command("problems")
def eval_problems() -> None:
    """List known problem specs and expected metrics."""
    from .problems import list_specs

    table = Table(title="ConStelX Problems")
    table.add_column("id")
    table.add_column("name")
    table.add_column("required")
    table.add_column("optional")
    for spec in list_specs():
        table.add_row(
            spec.pid,
            spec.name,
            ", ".join(spec.required_metrics) or "-",
            ", ".join(spec.optional_metrics) or "-",
        )
    console.print(table)


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


@opt_app.command("run")
def opt_run(
    baseline: str = typer.Option("trust-constr", help="Baseline: trust-constr|alm|cmaes"),
    nfp: int = typer.Option(3, help="Boundary NFP for boundary-mode optimization."),
    budget: int = typer.Option(50, help="Iteration budget (outer*inner for ALM)."),
    seed: int = typer.Option(0, help="Random seed (reserved for future use)."),
    use_physics: bool = typer.Option(
        False, "--use-physics", help="Use official evaluator when available."
    ),
    problem: Optional[str] = typer.Option(
        None, help="Problem id when using --use-physics (p1|p2|p3)."
    ),
    vmec_level: Optional[str] = typer.Option(
        None,
        help="VMEC resolution level for physics evals (auto|low|medium|high).",
    ),
    vmec_hot_restart: Optional[bool] = typer.Option(
        None,
        "--vmec-hot-restart/--no-vmec-hot-restart",
        help="Enable VMEC hot restart when available (defaults via env).",
    ),
    vmec_restart_key: Optional[str] = typer.Option(
        None,
        help="Optional VMEC hot-restart key for shared state.",
    ),
) -> None:
    """Run an optimization baseline in boundary mode (2D helical coefficients)."""
    if use_physics and not problem:
        # Emit a simple, testable message and exit with error code.
        typer.echo("--problem is required", err=True)
        raise typer.Exit(code=2)

    if baseline.lower() == "cmaes":
        # Delegate to existing CMA-ES command with boundary mode settings
        opt_cmaes(nfp=nfp, budget=budget, seed=seed, toy=False)
        return None

    from .optim.baselines import BaselineConfig, run_alm, run_trust_constr

    cfg = BaselineConfig(
        nfp=nfp,
        budget=budget,
        seed=seed,
        use_physics=use_physics,
        problem=(problem or "p1"),
        vmec_level=vmec_level,
        vmec_hot_restart=vmec_hot_restart,
        vmec_restart_key=vmec_restart_key,
    )
    if baseline.lower() in {"trust", "trust-constr", "trust_constr"}:
        x, val = run_trust_constr(cfg)
    elif baseline.lower() in {"alm", "augmented-lagrangian"}:
        x, val = run_alm(cfg)
    else:
        raise typer.BadParameter(f"Unknown baseline: {baseline}")
    console.print(f"Best x: {list(map(float, x))}\nBest score: {val}")


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
    steps: int = typer.Option(20, help="Number of optimizer steps"),
) -> None:
    from .surrogate.train import train_simple_mlp

    output_dir.mkdir(parents=True, exist_ok=True)
    train_simple_mlp(
        cache_dir=cache_dir, output_dir=output_dir, use_pbfm=use_pbfm, steps=int(steps)
    )
    console.print(f"Saved surrogate to {output_dir}")


# -------------------- AGENT --------------------
agent_app = typer.Typer(help="Multi-step agent loop (propose → simulate → select → refine)")
app.add_typer(agent_app, name="agent")


@agent_app.command("run")
def agent_run(
    nfp: int = typer.Option(
        3,
        help=("Number of field periods for random boundaries (default when --nfp-list not set)."),
    ),
    nfp_list: Optional[str] = typer.Option(
        None,
        help=(
            "Comma or space separated list of NFP values to explore in one run; "
            "budget is shared and candidates are allocated round-robin (e.g., '3,4,5')."
        ),
    ),
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
    vmec_level: Optional[str] = typer.Option(
        None,
        help="VMEC resolution (auto|low|medium|high) when physics is enabled.",
    ),
    vmec_hot_restart: Optional[bool] = typer.Option(
        None,
        "--vmec-hot-restart/--no-vmec-hot-restart",
        help="Enable VMEC hot restart when available (defaults via env).",
    ),
    vmec_restart_key: Optional[str] = typer.Option(
        None,
        help="Optional VMEC restart key shared across evaluations.",
    ),
    init_seeds: Optional[Path] = typer.Option(
        None, help="JSONL of initial boundary seeds to evaluate first."
    ),
    guard_simple: bool = typer.Option(
        False, help="Apply simple pre-screen guard (clamp R0, cap helical amps)."
    ),
    guard_geo: bool = typer.Option(
        False, help="Apply geometric nudge (tighten helical amps and align ratio)."
    ),
    guard_geom_validate: bool = typer.Option(
        False, help="Strict geometry validity check; skip invalid candidates."
    ),
    guard_r0_min: float = typer.Option(
        0.05, help="Guard: minimum base radius (R0)", show_default=True
    ),
    guard_r0_max: float = typer.Option(
        5.0, help="Guard: maximum base radius (R0)", show_default=True
    ),
    guard_helical_ratio_max: float = typer.Option(
        0.5, help="Guard: max total |m=1| amplitude as a fraction of R0", show_default=True
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
    # Multi-fidelity proxy gating
    mf_proxy: bool = typer.Option(False, help="Enable proxy gating before real evals"),
    mf_threshold: Optional[float] = typer.Option(
        None, help="Proxy score threshold (keep <= threshold)."
    ),
    mf_quantile: Optional[float] = typer.Option(
        None, help="Proxy keep quantile in [0,1] when threshold not set."
    ),
    mf_max_high: Optional[int] = typer.Option(
        None, help="Cap number of real-eval survivors per batch."
    ),
    mf_proxy_metric: str = typer.Option(
        "score",
        help=(
            "Proxy metric used for gating when --mf-proxy is enabled: "
            "score|placeholder_metric|qs_residual|qi_residual|helical_energy|mirror_ratio."
        ),
    ),
    seed_mode: str = typer.Option(
        "random",
        help="Seed generator for new proposals: random|near-axis",
        case_sensitive=False,
    ),
    # Novelty gating
    novelty_skip: bool = typer.Option(
        False,
        "--novelty-skip",
        help="Skip near-duplicate boundaries to save evaluator calls.",
    ),
    novelty_metric: str = typer.Option(
        "l2",
        help="Novelty metric: l2|cosine|allclose (window-based check).",
    ),
    novelty_eps: float = typer.Option(
        1e-6,
        help="Novelty threshold (distance<=eps considered duplicate).",
    ),
    novelty_window: int = typer.Option(
        128,
        help="Window size of recent boundaries kept for novelty checks.",
    ),
    novelty_db: Optional[Path] = typer.Option(
        None,
        help="Optional JSONL path to persist novelty vectors across runs.",
    ),
    surrogate_screen: bool = typer.Option(
        False,
        help="Enable surrogate screening before evaluator calls.",
    ),
    surrogate_model: Optional[Path] = typer.Option(
        None,
        help="Path to surrogate weights (e.g., outputs/surrogates/mlp/mlp.pt).",
    ),
    surrogate_metadata: Optional[Path] = typer.Option(
        None,
        help="Optional metadata JSON (defaults to metadata.json next to the model).",
    ),
    surrogate_threshold: Optional[float] = typer.Option(
        None,
        help="Keep proposals with surrogate score <= threshold.",
    ),
    surrogate_quantile: Optional[float] = typer.Option(
        None,
        help="Surrogate keep quantile when threshold not set (default 0.5).",
    ),
    surrogate_keep_max: Optional[int] = typer.Option(
        None,
        help="Maximum survivors per batch after surrogate screening.",
    ),
) -> None:
    from .agents.simple_agent import AgentConfig, run as run_agent  # noqa: I001

    runs_dir.mkdir(parents=True, exist_ok=True)
    if use_physics and not problem:
        typer.echo("--problem is required", err=True)
        raise typer.Exit(code=2)
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

    if mf_proxy:
        try:
            from constelx.eval import MF_PROXY_METRICS

            allowed = {m.strip().lower() for m in MF_PROXY_METRICS}
        except Exception:
            allowed = {
                "score",
                "placeholder_metric",
                "qs_residual",
                "qi_residual",
                "helical_energy",
                "mirror_ratio",
            }
        metric_norm = (mf_proxy_metric or "score").strip().lower()
        if metric_norm not in allowed:
            supported = ", ".join(sorted(allowed))
            raise typer.BadParameter(
                f"--mf-proxy-metric must be one of: {supported}",
                param_hint="--mf-proxy-metric",
            )

    out = run_agent(
        AgentConfig(
            nfp=nfp,
            nfp_list=(
                [int(x) for x in ([s for s in nfp_list.replace(",", " ").split() if s.strip()])]
                if isinstance(nfp_list, str) and nfp_list.strip()
                else None
            ),
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
            vmec_level=vmec_level,
            vmec_hot_restart=vmec_hot_restart,
            vmec_restart_key=vmec_restart_key,
            init_seeds=init_seeds,
            guard_simple=guard_simple,
            guard_geo=guard_geo,
            pcfm_gn_iters=pcfm_gn_iters,
            pcfm_damping=pcfm_damping,
            pcfm_tol=pcfm_tol,
            guard_geom_validate=guard_geom_validate,
            guard_geom_r0_min=guard_r0_min,
            guard_geom_r0_max=guard_r0_max,
            guard_geom_helical_ratio_max=guard_helical_ratio_max,
            mf_proxy=mf_proxy,
            mf_threshold=mf_threshold,
            mf_quantile=mf_quantile,
            mf_max_high=mf_max_high,
            mf_proxy_metric=mf_proxy_metric,
            seed_mode=seed_mode,
            novelty_skip=novelty_skip,
            novelty_metric=novelty_metric,
            novelty_eps=novelty_eps,
            novelty_window=novelty_window,
            novelty_db=novelty_db,
            surrogate_screen=surrogate_screen,
            surrogate_model=surrogate_model,
            surrogate_metadata=surrogate_metadata,
            surrogate_threshold=surrogate_threshold,
            surrogate_quantile=surrogate_quantile,
            surrogate_keep_max=surrogate_keep_max,
        )
    )
    console.print(f"Run complete. Artifacts in: [bold]{out}[/bold]")


# -------------------- SUBMIT --------------------
submit_app = typer.Typer(help="Submission packaging helpers")
app.add_typer(submit_app, name="submit")


@submit_app.command("pack")
def submit_pack(
    run_dir: Path = typer.Argument(..., help="Path to a completed run directory (runs/<ts>)."),
    out: Path = typer.Option(Path("submission.zip"), help="Output zip file path"),
    top_k: int = typer.Option(1, help="Include top-K boundaries as boundaries.jsonl (K>1)"),
) -> None:
    """Pack a run directory into a submission zip.

    Writes:
    - boundary.json (best boundary by aggregate score)
    - best.json (if present in the run folder)
    - metadata.json (includes problem, scoring_version, git_sha, top_k)
    - boundaries.jsonl (only when --top-k > 1): one JSON per line with
      {iteration,index,agg_score,evaluator_score,feasible,fail_reason,
       source,scoring_version,boundary}.
    """
    from .submit.pack import pack_run

    out_path = pack_run(run_dir, out, top_k=top_k)
    console.print(f"Created submission: [bold]{out_path}[/bold]")


# -------------------- ABLATION --------------------
ablate_app = typer.Typer(help="Ablation harness for pipeline components")
app.add_typer(ablate_app, name="ablate")


@ablate_app.command("run")
def ablate_run(
    nfp: int = typer.Option(3, help="Boundary NFP (passed to agent)."),
    budget: int = typer.Option(6, help="Per-run budget for quick ablations."),
    seed: int = typer.Option(0, help="Seed for reproducibility."),
    spec: Optional[Path] = typer.Option(
        None, "--spec", "--spec-file", help="JSON spec defining a full ablation plan"
    ),
    components: str = typer.Option(
        "guard_simple,mf_proxy",
        help=(
            "Comma-separated list of components to toggle one-at-a-time. "
            "Known: guard_simple, guard_geo, mf_proxy, algo=cmaes"
        ),
    ),
    runs_dir: Path = typer.Option(
        Path("runs/ablations"), "--runs-dir", help="Root directory for ablation runs."
    ),
    cache_dir: Path = typer.Option(Path(".cache/eval"), help="Cache directory for evals."),
    max_workers: int = typer.Option(1, help="Worker count for evaluator."),
    use_physics: bool = typer.Option(
        False, "--use-physics", help="Use official evaluator when available."
    ),
    problem: Optional[str] = typer.Option(
        None, help="Problem id when using --use-physics (p1|p2|p3)."
    ),
) -> None:
    """Run a small ablation over selected components or a JSON spec plan.

    When --spec is provided, it must be a JSON file with the schema:
    {
      "base": {AgentConfig-like fields},
      "seeds": [0,1,2],
      "variants": [{"name": "baseline", "overrides": {...}}, ...]
    }
    It writes per-variant/per-seed folders and summary CSVs.
    Without --spec, it runs a baseline and one run per component toggle.
    """
    if use_physics and not problem:
        typer.echo("--problem is required", err=True)
        raise typer.Exit(code=2)

    from .agents.simple_agent import AgentConfig

    runs_dir.mkdir(parents=True, exist_ok=True)

    if spec is not None:
        data = json.loads(spec.read_text())
        base = dict(data.get("base", {}))
        # Ensure base provides minimal fields; override with CLI values if present
        base.setdefault("nfp", nfp)
        base.setdefault("budget", budget)
        base.setdefault("algo", "random")
        if use_physics:
            base["use_physics"] = True
            if problem:
                base["problem"] = problem
        seeds = data.get("seeds") or [seed]
        variants_in = data.get("variants", [])
        from .agents.ablation import AblationPlan, PlanVariant, run_ablation_plan

        variants = [
            PlanVariant(name=str(v.get("name")), overrides=dict(v.get("overrides", {})))
            for v in variants_in
        ]
        plan = AblationPlan(base=base, seeds=[int(s) for s in seeds], variants=variants)
        root = run_ablation_plan(plan=plan, out_root=runs_dir)
        console.print(f"Ablation (spec) complete. Artifacts in: [bold]{root}[/bold]")
        return

    # Component-toggles path
    from .agents.ablation import run_ablation

    cfg = AgentConfig(
        nfp=nfp,
        seed=seed,
        out_dir=runs_dir,
        algo="random",
        budget=budget,
        resume=None,
        max_workers=max_workers,
        cache_dir=cache_dir,
        use_physics=use_physics,
        problem=(problem or "p1"),
    )
    comps = [c.strip() for c in components.split(",") if c.strip()]
    results = run_ablation(base=cfg, components=comps, out_root=runs_dir)

    # Print a compact summary
    table = Table(title="Ablation results")
    table.add_column("component")
    table.add_column("best_score")
    table.add_column("run_dir")
    for r in results:
        table.add_row(r.name, f"{r.best_score:.6g}", str(r.run_dir))
    console.print(table)


if __name__ == "__main__":
    app()
