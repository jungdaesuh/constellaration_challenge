from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from constelx.eval import forward as eval_forward
from constelx.eval import score as eval_score
from constelx.optim.pareto import pareto_indices
from constelx.physics.constel_api import example_boundary


@dataclass
class SweepPoint:
    r_amp: float
    z_amp: float
    objective: float
    qs_residual: float
    helical_energy: float
    feasible: bool
    metrics: dict[str, float]


def make_boundary(r_amp: float, z_amp: float) -> dict[str, float | list[list[float]]]:
    b = example_boundary()
    b["r_cos"][1][5] = float(r_amp)
    b["z_sin"][1][5] = float(z_amp)
    return b


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-0.05, 0.05, 5)
    sweep: List[SweepPoint] = []
    for r_amp in grid:
        for z_amp in grid:
            bnd = make_boundary(r_amp, z_amp)
            try:
                metrics = eval_forward(
                    bnd,
                    use_real=True,
                    prefer_vmec=True,
                    problem="p1",
                )
                agg = float(eval_score(metrics, problem="p1"))
                qs_res = float(metrics.get("qs_residual", float("nan")))
                he_eng = float(metrics.get("helical_energy", float("nan")))
                sweep.append(
                    SweepPoint(
                        r_amp=float(r_amp),
                        z_amp=float(z_amp),
                        objective=agg,
                        qs_residual=qs_res,
                        helical_energy=he_eng,
                        feasible=bool(metrics.get("feasible", True)),
                        metrics={
                            k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
                        },
                    )
                )
            except Exception as exc:  # pragma: no cover - best effort sweep
                sweep.append(
                    SweepPoint(
                        r_amp=float(r_amp),
                        z_amp=float(z_amp),
                        objective=float("inf"),
                        qs_residual=float("inf"),
                        helical_energy=float("nan"),
                        feasible=False,
                        metrics={"fail_reason": str(exc)},
                    )
                )
    sweep_records = [asdict(s) for s in sweep]
    (out_dir / "sweep.json").write_text(json.dumps(sweep_records, indent=2))

    valid = [s for s in sweep if np.isfinite(s.objective) and np.isfinite(s.qs_residual)]
    if not valid:
        return
    points = [(s.objective, s.qs_residual) for s in valid]
    # pareto_indices assumes minimize by default
    front_idx = pareto_indices(points, minimize=True)
    front = [valid[i] for i in front_idx]
    (out_dir / "pareto_front.json").write_text(json.dumps([asdict(s) for s in front], indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        [s.qs_residual for s in valid],
        [s.objective for s in valid],
        c="tab:blue",
        label="Candidates",
    )
    ax.scatter(
        [s.qs_residual for s in front],
        [s.objective for s in front],
        c="tab:orange",
        label="Pareto front",
    )
    ax.set_xlabel("QS residual (↓)")
    ax.set_ylabel("Aggregate score (↓)")
    ax.set_title("Real evaluator Pareto sweep (nfp=3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=150)


if __name__ == "__main__":
    main(Path(__file__).parent)
