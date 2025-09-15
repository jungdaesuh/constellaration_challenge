from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - skip when torch unavailable
    torch = None  # type: ignore[assignment]

from constelx.agents.simple_agent import AgentConfig, run
from constelx.surrogate.train import MLP

requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _prepare_surrogate(tmp_path: Path) -> tuple[Path, Path]:
    model = MLP(2, 1)
    if torch is None:  # pragma: no cover - guarded by marker
        raise RuntimeError("torch required")
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


@requires_torch
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
    rows = metrics_path.read_text().strip().splitlines()
    assert rows, "metrics.csv should not be empty"
    assert any("filtered_surrogate" in row for row in rows[1:]), "surrogate filter row missing"
