from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest


def test_surrogate_scorer_scores_boundary(
    tmp_path: Path, surrogate_modules: Tuple[object, type, object, type]
) -> None:
    torch_module, MLP, load_scorer, _ = surrogate_modules

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

    scorer = load_scorer(model_path, metadata_path)
    boundary = {
        "r_cos": [[0.0, 0.0, 0.0, 0.0, 0.0, -0.1]],
        "z_sin": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.1]],
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    assert scorer.score_one(boundary) == pytest.approx(2.0, rel=1e-6)
    preds = scorer.score_many([boundary, boundary])
    assert preds == pytest.approx([2.0, 2.0], rel=1e-6)
