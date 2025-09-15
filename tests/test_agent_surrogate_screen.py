from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Tuple

from constelx.agents.simple_agent import AgentConfig, run


def _prepare_surrogate(
    tmp_path: Path, modules: Tuple[object, type, object, type]
) -> tuple[Path, Path]:
    torch_module, MLP, _, _ = modules
    model = MLP(2, 1)
    with torch_module.no_grad():
        for layer in model.net:  # type: ignore[attr-defined]
            if isinstance(layer, torch_module.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        final_layer = model.net[-1]
        if isinstance(final_layer, torch_module.nn.Linear):
            final_layer.bias.fill_(2.0)
    model_path = tmp_path / "mlp.pt"
    metadata_path = tmp_path / "metadata.json"
    torch_module.save(model.state_dict(), model_path)
    metadata_path.write_text(
        json.dumps({"feature_columns": ["boundary.r_cos.1.5", "boundary.z_sin.1.5"]}, indent=2)
    )
    return model_path, metadata_path


def test_agent_records_surrogate_filtered_rows(
    tmp_path: Path, surrogate_modules: Tuple[object, type, object, type]
) -> None:
    model_path, metadata_path = _prepare_surrogate(tmp_path, surrogate_modules)
    out_dir = tmp_path / "runs"
    cfg = AgentConfig(
        nfp=3,
        seed=0,
        out_dir=out_dir,
        budget=4,
        max_workers=2,
        cache_dir=None,
        surrogate_screen=True,
        surrogate_model=model_path,
        surrogate_metadata=metadata_path,
        surrogate_threshold=1.0,
    )
    run_dir = run(cfg)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "metrics.csv should not be empty"
    surrogate_rows = [r for r in rows if (r.get("phase") or "").lower() == "surrogate"]
    assert surrogate_rows, "surrogate filter row missing"

    real_before = [r for r in rows if (r.get("phase") or "").lower() != "surrogate"]

    resume_cfg = replace(cfg, resume=run_dir, budget=cfg.budget + 1)
    run(resume_cfg)

    with metrics_path.open(newline="") as fh:
        rows_after = list(csv.DictReader(fh))

    real_after = [r for r in rows_after if (r.get("phase") or "").lower() != "surrogate"]
    assert len(real_after) > len(real_before), "resume run must add real evaluations"
