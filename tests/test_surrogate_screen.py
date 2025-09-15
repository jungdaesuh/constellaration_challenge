from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - skip when torch unavailable
    torch = None  # type: ignore[assignment]

from constelx.surrogate.screen import SurrogateScreenError, load_scorer
from constelx.surrogate.train import MLP


requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _write_dummy_model(tmp_path: Path) -> tuple[Path, Path]:
    model = MLP(2, 1)
    if torch is None:  # pragma: no cover - guarded by marker
        raise SurrogateScreenError("torch required")
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
def test_surrogate_scorer_scores_boundary(tmp_path: Path) -> None:
    model_path, metadata_path = _write_dummy_model(tmp_path)
    scorer = load_scorer(model_path, metadata_path)
    boundary = {
        "r_cos": [[0.0, 0.0, 0.0, 0.0, 0.0, -0.1]],
        "z_sin": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.1]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    assert pytest.approx(scorer.score_one(boundary), rel=1e-6) == 2.0
    preds = scorer.score_many([boundary, boundary])
    assert preds == pytest.approx([2.0, 2.0], rel=1e-6)
