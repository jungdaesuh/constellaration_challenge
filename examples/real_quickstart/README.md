# Real Quickstart

This directory contains a minimal reference for exercising the real Constellaration
path on a workstation or CI node with the `physics` extras installed.

## Prerequisites

```bash
pip install -e '.[physics]'
export CONSTELX_USE_REAL_EVAL=1
```

If you are running in CI, make sure the VMEC/NetCDF system packages are present
(e.g. `sudo apt-get install -y libnetcdf-dev netcdf-bin`).

## Forward evaluation smoke

You can reproduce the bundled `best.json` by evaluating a near-axis seed with
real physics enabled:

```bash
constelx eval forward --near-axis --use-physics --problem p1 --json \
  > examples/real_quickstart/best.json
```

The resulting metrics include `"source": "real"` and VMEC provenance to confirm
that the physics stack was exercised successfully.

## Small agent run

For a tiny end-to-end loop, start from the parameters in `config.yaml`:

```bash
constelx agent run --nfp 3 --budget 3 --use-physics --problem p1 \
  --seed 0 --runs-dir runs/real_quickstart
```

When physics extras are available the run produces a metrics table with
`source=real` and a `best.json` shaped like the reference.
