# SWIN3D Dual-Decoder InSAR Inversion

Reusable package for physics-aware inversion of InSAR land deformation using a compact 3D Swin Transformer with dual decoder heads for `S0` and `Sg`.

This repository is the clean reusable package extracted from the Punjab experiments. It is designed for users who want to run the same inversion workflow on their own deformation stacks stored in NetCDF or HDF5.

## What It Does

- loads a deformation stack from `NetCDF` or `HDF5`
- tiles the stack into spatiotemporal windows
- trains a dual-decoder 3D Swin model with forward-consistency and smoothness regularization
- exports stitched `S0` and `Sg` predictions back to NetCDF

## Input Requirements

The current package assumes:

- one deformation variable with shape `time x y x x` after reordering
- optional `time`, `lat`, and `lon` variables
- optional mask variable

The variable names and dimension order are controlled in a JSON config file, so users do not need to rename their data files.

## Quick Start

### Docker

Build the image:

```bash
docker build -t swin3d-dual-decoder-insar-inversion .
```

Inspect your dataset:

```bash
docker run --rm \
  -v $(pwd)/example_data:/workspace/data \
  -v $(pwd)/runs:/workspace/runs \
  swin3d-dual-decoder-insar-inversion \
  inspect --config configs/example_netcdf_train.json
```

Train:

```bash
docker run --rm \
  -v $(pwd)/example_data:/workspace/data \
  -v $(pwd)/runs:/workspace/runs \
  swin3d-dual-decoder-insar-inversion \
  train --config configs/example_netcdf_train.json
```

Predict:

```bash
docker run --rm \
  -v $(pwd)/example_data:/workspace/data \
  -v $(pwd)/runs:/workspace/runs \
  swin3d-dual-decoder-insar-inversion \
  predict --config configs/example_netcdf_train.json --checkpoint runs/example/checkpoints/best.pt
```

### Local Python

```bash
pip install -e .
```

Then run:

```bash
swin3d-insar-inversion inspect --config configs/example_netcdf_train.json
swin3d-insar-inversion train --config configs/example_netcdf_train.json
swin3d-insar-inversion predict --config configs/example_netcdf_train.json --checkpoint runs/example/checkpoints/best.pt
```

## Repository Layout

- `src/swin3d_dual_decoder_insar_inversion/`
  Core package
- `configs/`
  Example JSON configs for NetCDF and HDF5 inputs
- `Dockerfile`
  Container build
- `docker-compose.yml`
  Optional local container workflow

## Output

Training writes:

- `history.csv`
- `metrics.json`
- `checkpoints/best.pt`
- `resolved_config.json`

Prediction writes:

- `predictions.nc`

with stitched `S0_pred`, `Sg_pred`, and `prediction_count`.

## Notes

- `S0` and `Sg` are model outputs in the inversion state space. Their physical interpretation depends on the training formulation and the scaling of the forward physics.
- The default training workflow is unsupervised with respect to the latent storage fields. It uses observed deformation for forward-consistency plus regularization terms.
- The package is intentionally smaller and cleaner than the Punjab working repository.
