from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - skip when torch unavailable
    pytest.skip("PyTorch not installed", allow_module_level=True)
    raise

from constelx.agents.simple_agent import AgentConfig, run
from constelx.surrogate.train import MLP


def _prepare_surrogate(tmp_path: Path) -> tuple[Path, Path]:
    model = MLP(2, 1)
    with torch.no_grad():
        for layer in model.net:  # type: ignore[attr-defined]
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        final = model.net[-1]
        if isinstance(final, torch.nn.Linear):
            final.bias.fill_(2.0)
    model_path = tmp_path / "mlp.pt"
    metadata_path = tmp_path / "metadata.json"
    torch.save(model.state_dict(), model_path)
    metadata_path.write_text(
        json.dumps({"feature_columns": ["boundary.r_cos.1.5", "boundary.z_sin.1.5"]}, indent=2)
    )
    return model_path, metadata_path


def test_agent_records_surrogate_filtered_rows(tmp_path: Path) -> None:
    model_path, metadata_path = _prepare_surrogate(tmp_path)
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
        reader = list(csv.DictReader(fh))
    assert reader, "metrics.csv should not be empty"
    surrogate_rows = [r for r in reader if (r.get("phase") or "").lower() == "surrogate"]
    assert surrogate_rows, "surrogate filter row missing"

    real_before = [r for r in reader if (r.get("phase") or "").lower() != "surrogate"]

    resume_cfg = replace(cfg, resume=run_dir, budget=cfg.budget + 1)
    run(resume_cfg)

    with metrics_path.open(newline="") as fh:
        reader_after = list(csv.DictReader(fh))

    real_after = [r for r in reader_after if (r.get("phase") or "").lower() != "surrogate"]
    assert len(real_after) > len(real_before), "resume run must add real evaluations"
