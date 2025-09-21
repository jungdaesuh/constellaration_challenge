# Data Ingestion Examples

Quick commands to mirror parts of the ConStellaration dataset locally as Parquet and derive seeds.

Synthetic (no network)
- `constelx data fetch --nfp 3 --limit 32 --cache-dir data/cache`

Real dataset from the Hub
- `constelx data fetch --source hf --nfp 3 --limit 128 --cache-dir data/cache`
  - Produces `data/cache/subset.parquet` with flattened columns like `boundary.r_cos.1.5`.

Seeds from HF
- `constelx data seeds --nfp 3 --k 64 --out data/seeds.jsonl`

Notes
- The Parquet slice keeps only `boundary.*` and `metrics.*` columns and flattens nested objects into dot-separated keys for easy selection.
- Use this Parquet file for surrogate training and quick experiments.
