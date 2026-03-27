from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from .config import InversionConfig
from .data import build_datasets
from .models import DualDecoderFrequencySeparatedSwinUNet3D


def _load_model(checkpoint_path: str | Path, config: InversionConfig, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device)
    model = DualDecoderFrequencySeparatedSwinUNet3D(
        base_dim=config.model.base_dim,
        time_patch=config.model.time_patch,
        spatial_patch=config.model.spatial_patch,
        num_heads=config.model.num_heads,
        window_size=config.model.window_size,
        merge_scale=config.model.merge_scale,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def predict_to_netcdf(config: InversionConfig, checkpoint_path: str | Path, output_path: str | Path | None = None) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stack, train_ds, val_ds, meta = build_datasets(config)
    all_indices = train_ds.indices + val_ds.indices
    dataset = train_ds.__class__(
        stack,
        all_indices,
        config.tiling.window_size,
        config.tiling.tile_size,
        input_mean=meta["input_mean"],
        input_std=meta["input_std"],
        obs_mean=meta["obs_mean"],
        obs_std=meta["obs_std"],
    )
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)
    model = _load_model(checkpoint_path, config, device)

    n_time, height, width = stack.deformation.shape
    s0_sum = np.zeros((n_time, height, width), dtype=np.float32)
    sg_sum = np.zeros((n_time, height, width), dtype=np.float32)
    count = np.zeros((n_time, height, width), dtype=np.float32)

    with torch.no_grad():
        for batch in loader:
            pred = model(batch["x"].to(device)).cpu().numpy()
            end_t = batch["end_t"].numpy()
            y0 = batch["y0"].numpy()
            x0 = batch["x0"].numpy()
            for i in range(pred.shape[0]):
                t_idx = int(end_t[i])
                yy = int(y0[i])
                xx = int(x0[i])
                s0_sum[t_idx, yy : yy + config.tiling.tile_size, xx : xx + config.tiling.tile_size] += pred[i, 0]
                sg_sum[t_idx, yy : yy + config.tiling.tile_size, xx : xx + config.tiling.tile_size] += pred[i, 1]
                count[t_idx, yy : yy + config.tiling.tile_size, xx : xx + config.tiling.tile_size] += 1.0

    valid = count > 0
    s0 = np.full_like(s0_sum, np.nan)
    sg = np.full_like(sg_sum, np.nan)
    s0[valid] = s0_sum[valid] / count[valid]
    sg[valid] = sg_sum[valid] / count[valid]

    coords: dict[str, object] = {"time": stack.time, "y": np.arange(height), "x": np.arange(width)}
    if stack.lat is not None:
        coords["lat"] = ("y", stack.lat) if np.ndim(stack.lat) == 1 else (("y", "x"), stack.lat)
    if stack.lon is not None:
        coords["lon"] = ("x", stack.lon) if np.ndim(stack.lon) == 1 else (("y", "x"), stack.lon)
    ds = xr.Dataset(
        data_vars={
            "S0_pred": (("time", "y", "x"), s0),
            "Sg_pred": (("time", "y", "x"), sg),
            "prediction_count": (("time", "y", "x"), count),
        },
        coords=coords,
    )
    output_path = Path(output_path or (Path(config.training.output_dir) / "predictions.nc"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    ds.close()
    return str(output_path)

