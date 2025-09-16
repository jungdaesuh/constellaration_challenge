from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence

from .train import MLP

_torch_module: Optional[ModuleType]
try:
    module = importlib.import_module("torch")
except Exception:  # pragma: no cover - import guard
    _torch_module = None
else:
    _torch_module = module

TORCH_MODULE = _torch_module


class SurrogateScreenError(RuntimeError):
    """Raised when surrogate screening cannot be configured."""


@dataclass(frozen=True)
class SurrogateMetadata:
    feature_columns: Sequence[str]


def _load_metadata(path: Path) -> SurrogateMetadata:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:  # noqa: PERF203 (fine for clarity)
        raise SurrogateScreenError(f"metadata file not found: {path}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise SurrogateScreenError(f"failed to parse metadata: {path}") from exc
    cols = data.get("feature_columns") if isinstance(data, dict) else None
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise SurrogateScreenError("metadata missing feature_columns list")
    if not cols:
        raise SurrogateScreenError("metadata feature_columns list is empty")
    return SurrogateMetadata(tuple(cols))


def _coeff(boundary: Mapping[str, Any], field: str, m: int, n: int) -> float:
    arr = boundary.get(field)
    if not isinstance(arr, list) or m < 0 or n < 0:
        return 0.0
    if m >= len(arr):
        return 0.0
    row = arr[m]
    if not isinstance(row, list) or n >= len(row):
        return 0.0
    try:
        return float(row[n])
    except Exception:
        return 0.0


def _feature_value(boundary: Mapping[str, Any], feature: str) -> float:
    parts = feature.split(".")
    if len(parts) != 4:
        return 0.0
    _, field, m_str, n_str = parts
    if field not in {"r_cos", "z_sin"}:
        return 0.0
    try:
        m = int(m_str)
        n = int(n_str)
    except Exception:
        return 0.0
    return _coeff(boundary, field, m, n)


class SurrogateScorer:
    """Utility to score boundaries with a trained surrogate."""

    def __init__(
        self, model: MLP, feature_columns: Sequence[str], torch_module: Any, device: Any
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = device
        self.features = list(feature_columns)
        self._torch = torch_module

    def _vectors(self, boundaries: Sequence[Mapping[str, Any]]) -> Any:
        rows = [[_feature_value(b, feature) for feature in self.features] for b in boundaries]
        return self._torch.tensor(rows, dtype=self._torch.float32, device=self.device)

    def score_many(self, boundaries: Sequence[Mapping[str, Any]]) -> list[float]:
        if not boundaries:
            return []
        with self._torch.no_grad():
            vecs = self._vectors(boundaries)
            preds = self.model(vecs)
        flat = preds.reshape(-1)
        return [float(x) for x in flat.detach().cpu().tolist()]

    def score_one(self, boundary: Mapping[str, Any]) -> float:
        return self.score_many([boundary])[0]


def load_scorer(
    model_path: Path,
    metadata_path: Path | None = None,
    *,
    device: str | Any | None = None,
) -> SurrogateScorer:
    if TORCH_MODULE is None:  # pragma: no cover - import guard
        raise SurrogateScreenError("PyTorch is required for surrogate screening")
    torch_module = TORCH_MODULE
    model_file = Path(model_path)
    if not model_file.exists():
        raise SurrogateScreenError(f"surrogate model not found: {model_file}")
    meta_file = (
        Path(metadata_path) if metadata_path is not None else model_file.parent / "metadata.json"
    )
    metadata = _load_metadata(meta_file)
    feat_dim = len(metadata.feature_columns)
    try:
        model = MLP(feat_dim, 1)
    except Exception as exc:  # pragma: no cover - defensive
        raise SurrogateScreenError("failed to construct surrogate model") from exc
    try:
        state = torch_module.load(model_file, map_location="cpu")
        model.load_state_dict(state)
    except Exception as exc:
        raise SurrogateScreenError(f"failed to load surrogate weights from {model_file}") from exc
    dev = torch_module.device(device) if device is not None else torch_module.device("cpu")
    model.to(dev)
    return SurrogateScorer(model, metadata.feature_columns, torch_module, dev)


__all__ = ["SurrogateScorer", "SurrogateScreenError", "SurrogateMetadata", "load_scorer"]
