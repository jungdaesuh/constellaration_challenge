"""Data-driven seeds prior using PCA + RF + generative models.

This module implements a lightweight training pipeline that mirrors the
ConStellaration paper's data prior: flatten boundary Fourier coefficients,
compress them with PCA, learn a random forest feasibility classifier, and fit
a generative model (Gaussian mixture or a simple quantile-based normalizing
flow) on the feasible latent manifold. The resulting model can score new
boundaries for feasibility and sample synthetic seeds for the agent.

Typical usage::

    from constelx.data.seeds_prior import (
        FeasibilitySpec,
        SeedsPriorConfig,
        train_prior,
    )

    model = train_prior(records, SeedsPriorConfig(), FeasibilitySpec())
    samples = model.sample(count=8, nfp=3, min_feasibility=0.6, seed=0)
    for s in samples:
        boundary = s.boundary
        prob = s.feasibility_score
        # feed to constelx.agent or write to seeds.jsonl

The module is intentionally self-contained so CLI commands and tests can reuse
the same training utilities without duplicating feature extraction logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from ..eval.boundary_param import validate as validate_boundary

CoeffArray = List[List[float]]


def _ensure_nested_list(values: Optional[Iterable[Iterable[Any]]]) -> Optional[CoeffArray]:
    if values is None:
        return None
    out: CoeffArray = []
    for row in values:
        out.append([float(x) for x in row])
    return out


def _nested_get(data: Mapping[str, Any], path: str) -> Any:
    if path in data:
        return data[path]
    current: Any = data
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _round_int(x: float | int, *, default: int = 3) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return default


def _make_grid(shape: tuple[int, int]) -> CoeffArray:
    m, n = shape
    return [[0.0 for _ in range(n)] for _ in range(m)]


class BoundaryVectorizer:
    """Flatten boundary dictionaries into feature vectors and invert them back."""

    def __init__(self) -> None:
        self.fields: List[str] = []
        self.m_dim: int = 0
        self.n_dim: int = 0
        self.nfp_index: int = 0
        self._offsets: dict[str, slice] = {}
        self._total_dim: int = 0

    def fit(self, boundaries: Sequence[Mapping[str, Any]]) -> BoundaryVectorizer:
        if not boundaries:
            raise ValueError("At least one boundary is required to fit the vectorizer")
        fields = ["r_cos", "z_sin"]
        if any(b.get("r_sin") not in (None, []) for b in boundaries):
            fields.append("r_sin")
        if any(b.get("z_cos") not in (None, []) for b in boundaries):
            fields.append("z_cos")
        m_dim = 0
        n_dim = 0
        for b in boundaries:
            for fname in fields:
                arr = b.get(fname)
                if arr is None:
                    continue
                try:
                    rows = len(arr)
                    cols = len(arr[0]) if rows > 0 else 0
                except Exception:
                    continue
                m_dim = max(m_dim, rows)
                n_dim = max(n_dim, cols)
        if m_dim == 0 or n_dim == 0:
            # fallback to minimal grid
            m_dim = max(m_dim, 1)
            n_dim = max(n_dim, 1)
        self.fields = fields
        self.m_dim = m_dim
        self.n_dim = n_dim
        offset = 1  # reserve first entry for nfp
        self.nfp_index = 0
        self._offsets = {}
        size = m_dim * n_dim
        for fname in fields:
            self._offsets[fname] = slice(offset, offset + size)
            offset += size
        self._total_dim = offset
        return self

    @property
    def feature_dim(self) -> int:
        return self._total_dim

    def transform(self, boundary: Mapping[str, Any]) -> NDArray[np.float64]:
        if not self.fields:
            raise RuntimeError("BoundaryVectorizer must be fit before calling transform")
        vec = np.zeros(self._total_dim, dtype=np.float64)
        vec[self.nfp_index] = float(boundary.get("n_field_periods", 3))
        for fname in self.fields:
            arr = boundary.get(fname)
            buf = np.zeros((self.m_dim, self.n_dim), dtype=np.float64)
            if arr is not None:
                try:
                    for i in range(min(len(arr), self.m_dim)):
                        row = arr[i]
                        for j in range(min(len(row), self.n_dim)):
                            buf[i, j] = float(row[j])
                except Exception:
                    pass
            vec[self._offsets[fname]] = buf.reshape(-1)
        return vec

    def inverse_transform(
        self,
        vec: NDArray[np.float64],
        *,
        override_nfp: Optional[int] = None,
        coeff_abs_max: float = 2.0,
        base_radius_min: float = 1e-3,
    ) -> dict[str, Any]:
        if vec.shape[0] != self._total_dim:
            raise ValueError(f"Expected vector of length {self._total_dim}, got {vec.shape[0]}")
        nfp_val = _round_int(vec[self.nfp_index])
        if override_nfp is not None:
            nfp_val = int(override_nfp)
        result: dict[str, Any] = {
            "n_field_periods": nfp_val,
            "is_stellarator_symmetric": True,
        }
        base_idx = min(4, self.n_dim - 1)
        for fname in self.fields:
            sl = self._offsets[fname]
            arr = vec[sl].reshape((self.m_dim, self.n_dim))
            np.clip(arr, -coeff_abs_max, coeff_abs_max, out=arr)
            if fname == "r_cos":
                arr[0, base_idx] = float(max(base_radius_min, arr[0, base_idx]))
            result[fname] = arr.tolist()
        # Optional fields were included only if seen during fit; keep None otherwise
        if "r_sin" not in self.fields:
            result["r_sin"] = None
        if "z_cos" not in self.fields:
            result["z_cos"] = None
        return result


@dataclass(frozen=True)
class FeasibilitySpec:
    """Describe how to extract feasibility labels from dataset records."""

    field: Optional[str] = "metrics.feasible"
    metric: Optional[str] = None
    threshold: Optional[float] = None
    sense: str = "lt"  # 'lt' (<= threshold) or 'gt' (>= threshold)

    def evaluate(self, record: Mapping[str, Any]) -> bool:
        if self.field:
            val = _nested_get(record, self.field)
            if val is not None:
                try:
                    return bool(val)
                except Exception:
                    pass
        if self.metric and self.threshold is not None:
            val = _nested_get(record, self.metric)
            if val is not None:
                try:
                    fv = float(val)
                    if self.sense == "gt":
                        return fv >= float(self.threshold)
                    return fv <= float(self.threshold)
                except Exception:
                    return False
        raise ValueError(
            "Unable to extract feasibility label; provide a valid field or metric/threshold"
        )


@dataclass(frozen=True)
class SeedsPriorConfig:
    pca_components: int = 8
    classifier_estimators: int = 200
    classifier_max_depth: Optional[int] = None
    generator: str = "gmm"  # 'gmm' or 'flow'
    gmm_components: int = 6
    random_state: int = 0
    quantile_bins: int = 256
    coeff_abs_max: float = 1.5
    base_radius_min: float = 0.05


class _LatentGenerator:
    def sample(
        self, n: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:  # pragma: no cover - interface
        raise NotImplementedError

    def save(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class _GmmGenerator(_LatentGenerator):
    def __init__(self, model: GaussianMixture) -> None:
        self.model = model

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        # GaussianMixture has its own RNG; clone parameters but sample manually for determinism
        # We delegate to model.sample but feed deterministic RNG via random_state attribute
        original_state = self.model.random_state
        self.model.random_state = int(rng.integers(0, 2**32 - 1))
        try:
            xs, _ = self.model.sample(n)
        finally:
            self.model.random_state = original_state
        xs_arr: NDArray[np.float64] = np.asarray(xs, dtype=np.float64)
        return xs_arr

    def save(self) -> Any:
        return self.model


class _QuantileFlowGenerator(_LatentGenerator):
    def __init__(self, transformer: QuantileTransformer) -> None:
        self.transformer = transformer

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        dim = self.transformer.n_features_in_
        normals: NDArray[np.float64] = rng.standard_normal((n, dim))
        uniforms: NDArray[np.float64] = norm.cdf(normals)
        uniforms = np.clip(uniforms, 1e-6, 1 - 1e-6)
        samples: NDArray[np.float64] = np.asarray(
            self.transformer.inverse_transform(uniforms), dtype=np.float64
        )
        return samples

    def save(self) -> Any:
        return self.transformer


@dataclass
class SeedsPriorArtifacts:
    vectorizer: BoundaryVectorizer
    scaler: StandardScaler
    pca: PCA
    classifier: RandomForestClassifier
    generator: _LatentGenerator
    generator_type: str


@dataclass
class PriorSample:
    boundary: dict[str, Any]
    feasibility_score: float
    latent: NDArray[np.float64]


class SeedsPriorModel:
    def __init__(
        self,
        artifacts: SeedsPriorArtifacts,
        config: SeedsPriorConfig,
        feasibility: FeasibilitySpec,
    ) -> None:
        self._artifacts = artifacts
        self._config = config
        self._feasibility = feasibility

    @property
    def config(self) -> SeedsPriorConfig:
        return self._config

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "config": self._config,
            "feasibility": self._feasibility,
            "vectorizer": self._artifacts.vectorizer,
            "scaler": self._artifacts.scaler,
            "pca": self._artifacts.pca,
            "classifier": self._artifacts.classifier,
            "generator_type": self._artifacts.generator_type,
            "generator": self._artifacts.generator.save(),
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: Path) -> SeedsPriorModel:
        payload = joblib.load(Path(path))
        generator_type = payload["generator_type"]
        generator: _LatentGenerator
        if generator_type == "gmm":
            generator = _GmmGenerator(payload["generator"])
        elif generator_type == "flow":
            generator = _QuantileFlowGenerator(payload["generator"])
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown generator type: {generator_type}")
        artifacts = SeedsPriorArtifacts(
            vectorizer=payload["vectorizer"],
            scaler=payload["scaler"],
            pca=payload["pca"],
            classifier=payload["classifier"],
            generator=generator,
            generator_type=generator_type,
        )
        return cls(artifacts, payload["config"], payload["feasibility"])

    def predict_feasibility(self, boundary: Mapping[str, Any]) -> float:
        vec = self._artifacts.vectorizer.transform(boundary)
        scaled = self._artifacts.scaler.transform(vec.reshape(1, -1))
        latent = self._artifacts.pca.transform(scaled)
        prob = self._artifacts.classifier.predict_proba(latent)[0][1]
        return float(prob)

    def sample(
        self,
        count: int,
        *,
        nfp: Optional[int] = None,
        min_feasibility: float = 0.5,
        max_draw_batches: int = 50,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[PriorSample]:
        rng = np.random.default_rng(seed)
        accepted: List[PriorSample] = []
        draws_per_batch = batch_size or max(count * 2, 8)
        tries = 0
        classifier = self._artifacts.classifier
        pca = self._artifacts.pca
        scaler = self._artifacts.scaler
        vectorizer = self._artifacts.vectorizer
        coeff_abs_max = self._config.coeff_abs_max
        base_radius_min = self._config.base_radius_min
        while len(accepted) < count and tries < max_draw_batches:
            latent = self._artifacts.generator.sample(draws_per_batch, rng)
            probs = classifier.predict_proba(latent)[:, 1]
            for z, prob in zip(latent, probs):
                if prob < min_feasibility:
                    continue
                scaled_vec = pca.inverse_transform(z.reshape(1, -1))
                raw_vec = scaler.inverse_transform(scaled_vec)[0]
                if nfp is not None:
                    raw_vec[vectorizer.nfp_index] = float(nfp)
                boundary = vectorizer.inverse_transform(
                    raw_vec,
                    override_nfp=nfp,
                    coeff_abs_max=coeff_abs_max,
                    base_radius_min=base_radius_min,
                )
                try:
                    validate_boundary(boundary)
                except Exception:
                    continue
                accepted.append(
                    PriorSample(boundary=boundary, feasibility_score=float(prob), latent=z)
                )
                if len(accepted) >= count:
                    break
            tries += 1
        return accepted


def _extract_boundary(record: Mapping[str, Any]) -> dict[str, Any]:
    boundary = record.get("boundary")
    if isinstance(boundary, Mapping):
        out: dict[str, Any] = {
            "r_cos": _ensure_nested_list(boundary.get("r_cos")),
            "r_sin": _ensure_nested_list(boundary.get("r_sin")),
            "z_cos": _ensure_nested_list(boundary.get("z_cos")),
            "z_sin": _ensure_nested_list(boundary.get("z_sin")),
            "n_field_periods": _round_int(boundary.get("n_field_periods", 3)),
            "is_stellarator_symmetric": bool(boundary.get("is_stellarator_symmetric", True)),
        }
        if out["r_cos"] is None:
            raise ValueError("Boundary missing r_cos coefficients")
        if out["z_sin"] is None:
            raise ValueError("Boundary missing z_sin coefficients")
        return out

    # Reconstruct from flattened keys boundary.<field>.<m>.<n>
    r_fields: dict[str, MutableMapping[tuple[int, int], float]] = {
        "r_cos": {},
        "z_sin": {},
        "r_sin": {},
        "z_cos": {},
    }
    nfp_val = _round_int(record.get("boundary.n_field_periods", 3))
    for key, val in record.items():
        if not isinstance(key, str) or not key.startswith("boundary."):
            continue
        parts = key.split(".")
        if len(parts) == 2 and parts[1] == "n_field_periods":
            nfp_val = _round_int(val)
            continue
        if len(parts) != 4:
            continue
        field = parts[1]
        if field not in r_fields:
            continue
        try:
            m = int(parts[2])
            n = int(parts[3])
            r_fields[field][(m, n)] = float(val)
        except Exception:
            continue

    def _assemble(field: str) -> Optional[CoeffArray]:
        entries = r_fields[field]
        if not entries:
            return None if field in {"r_sin", "z_cos"} else _make_grid((1, 1))
        max_m = max(m for m, _ in entries.keys())
        max_n = max(n for _, n in entries.keys())
        grid = _make_grid((max_m + 1, max_n + 1))
        for (m, n), v in entries.items():
            if m < len(grid) and n < len(grid[0]):
                grid[m][n] = v
        return grid

    r_cos = _assemble("r_cos")
    z_sin = _assemble("z_sin")
    if r_cos is None or z_sin is None:
        raise ValueError("Unable to reconstruct boundary coefficients")
    return {
        "r_cos": r_cos,
        "r_sin": _assemble("r_sin"),
        "z_cos": _assemble("z_cos"),
        "z_sin": z_sin,
        "n_field_periods": nfp_val,
        "is_stellarator_symmetric": True,
    }


def train_prior(
    records: Sequence[Mapping[str, Any]],
    config: SeedsPriorConfig,
    feasibility: FeasibilitySpec,
    *,
    nfp: Optional[int] = None,
    min_feasible: int = 8,
) -> SeedsPriorModel:
    if not records:
        raise ValueError("No records provided to train prior")
    boundaries: List[dict[str, Any]] = []
    labels: List[bool] = []
    for rec in records:
        b = _extract_boundary(rec)
        if nfp is not None and b.get("n_field_periods") != nfp:
            continue
        try:
            label = feasibility.evaluate(rec)
        except ValueError:
            continue
        boundaries.append(b)
        labels.append(bool(label))
    if not boundaries:
        raise ValueError("No boundaries remaining after filtering for NFP/labels")
    positives = sum(1 for x in labels if x)
    if positives < min_feasible:
        raise ValueError(f"Not enough feasible examples ({positives}) to fit generative model")
    vectorizer = BoundaryVectorizer().fit(boundaries)
    X = np.stack([vectorizer.transform(b) for b in boundaries])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    latent_dim = min(config.pca_components, X_scaled.shape[1], len(boundaries))
    latent_dim = max(1, latent_dim)
    pca = PCA(n_components=latent_dim, random_state=config.random_state)
    Z = pca.fit_transform(X_scaled)
    clf = RandomForestClassifier(
        n_estimators=config.classifier_estimators,
        max_depth=config.classifier_max_depth,
        random_state=config.random_state,
        class_weight="balanced",
    )
    clf.fit(Z, labels)
    feasible_idx = np.array(labels, dtype=bool)
    Z_feasible = Z[feasible_idx]
    generator: _LatentGenerator
    generator_type: str
    if config.generator == "flow":
        qt = QuantileTransformer(
            n_quantiles=min(config.quantile_bins, len(Z_feasible)),
            output_distribution="uniform",
            random_state=config.random_state,
        )
        qt.fit(Z_feasible)
        generator = _QuantileFlowGenerator(qt)
        generator_type = "flow"
    else:
        n_components = min(config.gmm_components, len(Z_feasible))
        n_components = max(1, n_components)
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=config.random_state,
        )
        gmm.fit(Z_feasible)
        generator = _GmmGenerator(gmm)
        generator_type = "gmm"
    artifacts = SeedsPriorArtifacts(
        vectorizer=vectorizer,
        scaler=scaler,
        pca=pca,
        classifier=clf,
        generator=generator,
        generator_type=generator_type,
    )
    return SeedsPriorModel(artifacts, config, feasibility)


def load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


__all__ = [
    "BoundaryVectorizer",
    "FeasibilitySpec",
    "SeedsPriorConfig",
    "SeedsPriorModel",
    "PriorSample",
    "train_prior",
    "load_jsonl",
]
