from __future__ import annotations

from constelx.data.seeds_prior import (
    FeasibilitySpec,
    SeedsPriorConfig,
    SeedsPriorModel,
    train_prior,
)
from constelx.eval.boundary_param import validate


def test_train_prior_gmm_and_sample(seeds_prior_records, tmp_path):
    config = SeedsPriorConfig(
        pca_components=5,
        generator="gmm",
        gmm_components=4,
        random_state=0,
    )
    model = train_prior(
        seeds_prior_records,
        config,
        FeasibilitySpec(field="metrics.feasible"),
        nfp=3,
    )
    samples = model.sample(count=6, nfp=3, min_feasibility=0.05, seed=1)
    assert len(samples) == 6
    for sample in samples:
        validate(sample.boundary)
        assert sample.boundary["n_field_periods"] == 3
        assert sample.feasibility_score >= 0.05
    model_path = tmp_path / "prior.joblib"
    model.save(model_path)
    loaded = SeedsPriorModel.load(model_path)
    score = loaded.predict_feasibility(samples[0].boundary)
    assert 0.0 <= score <= 1.0


def test_train_prior_flow_metric_threshold(seeds_prior_records):
    spec = FeasibilitySpec(
        field=None,
        metric="metrics.metric",
        threshold=0.09,
        sense="lt",
    )
    config = SeedsPriorConfig(
        pca_components=4,
        generator="flow",
        quantile_bins=48,
        random_state=3,
    )
    model = train_prior(seeds_prior_records, config, spec, nfp=3)
    samples = model.sample(count=4, nfp=3, min_feasibility=0.2, seed=3)
    # Flow sampler may reject a few draws; ensure we still get at least two
    assert len(samples) >= 2
    for sample in samples:
        validate(sample.boundary)
