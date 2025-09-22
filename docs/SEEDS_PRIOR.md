# Seeds Prior (HF Parquet)

We trained lightweight priors on the Hugging Face Parquet slice produced by
`constelx data fetch --source hf --nfp 3 --limit 256`.

```
constelx data prior-train data/cache/subset.parquet \
  --feasible-metric metrics.vacuum_well --feasible-threshold -0.12 --feasible-sense lt \
  --min-feasible 50 --coeff-abs-max 0.3 --base-radius-min 0.3 \
  --generator gmm --out models/seeds_prior_hf_gmm.joblib

constelx data prior-train ... --generator flow --out models/seeds_prior_hf_flow.joblib
```

Sampled seeds (placeholder evaluator; `min_feasibility=0.4`, 8 draws, seed=0):

| Generator | mean placeholder score ↓ | median | mean predicted feasibility |
|-----------|--------------------------|--------|-----------------------------|
| GMM       | 5.40e-2                  | 5.40e-2| 0.50                        |
| Flow      | 5.40e-2                  | 5.40e-2| 0.50                        |

Both models currently behave the same on the placeholder path; we keep the GMM
variant (`models/seeds_prior_hf_gmm.joblib`) as the default for
`constelx agent run --seed-mode prior`. Flow samples remain available for
ablation under `models/seeds_prior_hf_flow.joblib`.

> NOTE: The real evaluator is sensitive to minor coefficient drift. Before
> deploying new priors, validate samples with `--use-physics` and tune
> `coeff_abs_max`/`base_radius_min` if feasibility drops.
