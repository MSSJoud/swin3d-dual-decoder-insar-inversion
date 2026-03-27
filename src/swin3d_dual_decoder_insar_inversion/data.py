from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from .config import InversionConfig


@dataclass(frozen=True)
class DataStack:
    deformation: np.ndarray
    time: np.ndarray
    lat: np.ndarray | None
    lon: np.ndarray | None
    mask: np.ndarray | None
    source_path: str
    source_variable: str


def _decode_time(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.arange(0)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    if values.dtype.kind in {"S", "U", "O"}:
        return pd.to_datetime(values.astype(str)).to_numpy(dtype="datetime64[ns]")
    return np.asarray(values)


def _infer_format(path: Path, declared: str) -> str:
    if declared != "auto":
        return declared
    suffix = path.suffix.lower()
    if suffix in {".nc", ".nc4", ".cdf"}:
        return "netcdf"
    if suffix in {".h5", ".hdf5"}:
        return "hdf5"
    raise ValueError(f"Could not infer input format from {path}")


def load_stack(config: InversionConfig) -> DataStack:
    path = Path(config.input.path)
    fmt = _infer_format(path, config.input.format)

    if fmt == "netcdf":
        ds = xr.open_dataset(path)
        var = ds[config.input.variable]
        if config.input.dim_order is not None:
            var = var.transpose(*config.input.dim_order)
        deformation = np.asarray(var.values, dtype=np.float32)
        time = (
            _decode_time(np.asarray(ds[config.input.time_name].values))
            if config.input.time_name and config.input.time_name in ds
            else np.arange(deformation.shape[0])
        )
        lat = None
        lon = None
        if config.input.lat_name and config.input.lat_name in ds:
            lat = np.asarray(ds[config.input.lat_name].values)
        if config.input.lon_name and config.input.lon_name in ds:
            lon = np.asarray(ds[config.input.lon_name].values)
        mask = None
        if config.input.mask_variable and config.input.mask_variable in ds:
            mask = np.asarray(ds[config.input.mask_variable].values)
        ds.close()
    elif fmt == "hdf5":
        with h5py.File(path, "r") as handle:
            deformation = np.asarray(handle[config.input.variable][...], dtype=np.float32)
            if config.input.transpose_order is not None:
                deformation = np.transpose(deformation, config.input.transpose_order)
            time = (
                _decode_time(np.asarray(handle[config.input.time_name][...]))
                if config.input.time_name and config.input.time_name in handle
                else np.arange(deformation.shape[0])
            )
            lat = np.asarray(handle[config.input.lat_name][...]) if config.input.lat_name and config.input.lat_name in handle else None
            lon = np.asarray(handle[config.input.lon_name][...]) if config.input.lon_name and config.input.lon_name in handle else None
            mask = (
                np.asarray(handle[config.input.mask_variable][...])
                if config.input.mask_variable and config.input.mask_variable in handle
                else None
            )
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if deformation.ndim != 3:
        raise ValueError(f"Expected deformation stack with 3 dimensions after reordering, got shape {deformation.shape}")

    return DataStack(
        deformation=deformation,
        time=time,
        lat=lat,
        lon=lon,
        mask=mask,
        source_path=str(path),
        source_variable=config.input.variable,
    )


def inspect_input(config: InversionConfig) -> dict:
    stack = load_stack(config)
    return {
        "source_path": stack.source_path,
        "variable": stack.source_variable,
        "shape": tuple(int(v) for v in stack.deformation.shape),
        "finite_fraction": float(np.isfinite(stack.deformation).mean()),
        "time_start": str(stack.time[0]) if stack.time.size else None,
        "time_end": str(stack.time[-1]) if stack.time.size else None,
        "lat_shape": None if stack.lat is None else tuple(int(v) for v in np.shape(stack.lat)),
        "lon_shape": None if stack.lon is None else tuple(int(v) for v in np.shape(stack.lon)),
        "mask_shape": None if stack.mask is None else tuple(int(v) for v in np.shape(stack.mask)),
    }


@dataclass(frozen=True)
class SampleIndex:
    tile_id: int
    y0: int
    x0: int
    end_t: int


class DeformationWindowDataset(Dataset):
    def __init__(
        self,
        stack: DataStack,
        indices: list[SampleIndex],
        window_size: int,
        tile_size: int,
        input_mean: float | None = None,
        input_std: float | None = None,
        obs_mean: float | None = None,
        obs_std: float | None = None,
    ):
        self.stack = stack
        self.indices = indices
        self.window_size = window_size
        self.tile_size = tile_size
        self.input_mean = float(input_mean) if input_mean is not None else 0.0
        self.input_std = float(input_std) if input_std is not None else 1.0
        self.obs_mean = float(obs_mean) if obs_mean is not None else 0.0
        self.obs_std = float(obs_std) if obs_std is not None else 1.0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.indices[idx]
        start_t = sample.end_t - self.window_size + 1
        cube = self.stack.deformation[start_t : sample.end_t + 1, sample.y0 : sample.y0 + self.tile_size, sample.x0 : sample.x0 + self.tile_size]
        obs = cube[-1]
        valid_mask = np.isfinite(obs).astype(np.float32)
        cube_filled = np.nan_to_num(cube, nan=self.input_mean)
        cube_norm = (cube_filled - self.input_mean) / max(self.input_std, 1e-6)
        obs_filled = np.nan_to_num(obs, nan=self.obs_mean)
        obs_norm = (obs_filled - self.obs_mean) / max(self.obs_std, 1e-6)
        return {
            "x": torch.tensor(cube_norm[None, ...], dtype=torch.float32),
            "obs_norm": torch.tensor(obs_norm[None, ...], dtype=torch.float32),
            "obs_raw": torch.tensor(obs_filled[None, ...], dtype=torch.float32),
            "valid_mask": torch.tensor(valid_mask[None, ...], dtype=torch.float32),
            "tile_id": torch.tensor(sample.tile_id, dtype=torch.long),
            "end_t": torch.tensor(sample.end_t, dtype=torch.long),
            "y0": torch.tensor(sample.y0, dtype=torch.long),
            "x0": torch.tensor(sample.x0, dtype=torch.long),
        }


def _build_indices(stack: DataStack, config: InversionConfig) -> list[SampleIndex]:
    window = config.tiling.window_size
    tile = config.tiling.tile_size
    stride = config.tiling.stride
    min_valid_fraction = config.tiling.min_valid_fraction

    time_count, height, width = stack.deformation.shape
    samples: list[SampleIndex] = []

    def start_positions(size: int, tile_size: int, stride_size: int) -> list[int]:
        if size <= tile_size:
            return [0]
        values = list(range(0, size - tile_size + 1, stride_size))
        last = size - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    tile_id = 0
    for y0 in start_positions(height, tile, stride):
        for x0 in start_positions(width, tile, stride):
            tile_slice = stack.deformation[:, y0 : y0 + tile, x0 : x0 + tile]
            if tile_slice.shape[1] != tile or tile_slice.shape[2] != tile:
                continue
            tile_has_samples = False
            for end_t in range(window - 1, time_count):
                cube = tile_slice[end_t - window + 1 : end_t + 1]
                valid_fraction = float(np.isfinite(cube).mean())
                if valid_fraction >= min_valid_fraction:
                    samples.append(SampleIndex(tile_id=tile_id, y0=y0, x0=x0, end_t=end_t))
                    tile_has_samples = True
            if tile_has_samples:
                tile_id += 1
    if not samples:
        raise RuntimeError("No valid training samples were generated from the provided stack and tiling settings.")
    return samples


def _compute_stats(stack: DataStack, indices: list[SampleIndex], config: InversionConfig) -> tuple[float, float, float, float]:
    window = config.tiling.window_size
    tile = config.tiling.tile_size
    input_values = []
    obs_values = []
    for sample in indices:
        start_t = sample.end_t - window + 1
        cube = stack.deformation[start_t : sample.end_t + 1, sample.y0 : sample.y0 + tile, sample.x0 : sample.x0 + tile]
        input_values.append(cube[np.isfinite(cube)])
        obs = cube[-1]
        obs_values.append(obs[np.isfinite(obs)])
    input_concat = np.concatenate([v for v in input_values if v.size > 0])
    obs_concat = np.concatenate([v for v in obs_values if v.size > 0])
    input_mean = float(input_concat.mean())
    input_std = float(max(input_concat.std(), 1e-6))
    obs_mean = float(obs_concat.mean())
    obs_std = float(max(obs_concat.std(), 1e-6))
    return input_mean, input_std, obs_mean, obs_std


def build_datasets(config: InversionConfig) -> tuple[DataStack, DeformationWindowDataset, DeformationWindowDataset, dict]:
    stack = load_stack(config)
    indices = _build_indices(stack, config)
    if len(indices) < 2:
        raise RuntimeError("At least two valid samples are required so the package can create train/validation splits.")
    split = int((1.0 - config.training.val_fraction) * len(indices))
    split = min(max(split, 1), len(indices) - 1)
    train_indices = indices[:split]
    val_indices = indices[split:]
    input_mean, input_std, obs_mean, obs_std = _compute_stats(stack, train_indices, config)
    train_ds = DeformationWindowDataset(
        stack,
        train_indices,
        config.tiling.window_size,
        config.tiling.tile_size,
        input_mean=input_mean,
        input_std=input_std,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    val_ds = DeformationWindowDataset(
        stack,
        val_indices,
        config.tiling.window_size,
        config.tiling.tile_size,
        input_mean=input_mean,
        input_std=input_std,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    meta = {
        "n_samples": len(indices),
        "n_train_samples": len(train_indices),
        "n_val_samples": len(val_indices),
        "n_tiles": int(len({sample.tile_id for sample in indices})),
        "input_mean": input_mean,
        "input_std": input_std,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
    }
    return stack, train_ds, val_ds, meta
