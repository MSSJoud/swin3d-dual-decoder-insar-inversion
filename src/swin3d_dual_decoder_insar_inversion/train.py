from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import InversionConfig
from .data import build_datasets
from .metrics import anisotropic_total_variation, masked_mse
from .models import DualDecoderFrequencySeparatedSwinUNet3D
from .physics import build_fft_kernels, forward_two_layer_torch, set_seed


def _build_model(config: InversionConfig, device: torch.device) -> DualDecoderFrequencySeparatedSwinUNet3D:
    model = DualDecoderFrequencySeparatedSwinUNet3D(
        base_dim=config.model.base_dim,
        time_patch=config.model.time_patch,
        spatial_patch=config.model.spatial_patch,
        num_heads=config.model.num_heads,
        window_size=config.model.window_size,
        merge_scale=config.model.merge_scale,
    )
    return model.to(device)


def _run_epoch(
    model: DualDecoderFrequencySeparatedSwinUNet3D,
    loader: DataLoader,
    device: torch.device,
    config: InversionConfig,
    obs_mean: float,
    obs_std: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    g_load_fft, g_poro_fft = build_fft_kernels(
        config.tiling.tile_size,
        config.tiling.tile_size,
        config.physics,
        device=device,
    )
    prev_by_tile: dict[int, torch.Tensor] = {}
    totals = {
        "loss": 0.0,
        "forward": 0.0,
        "spatial_s0": 0.0,
        "spatial_sg": 0.0,
        "temporal_s0": 0.0,
        "temporal_sg": 0.0,
        "n_batches": 0.0,
    }
    for batch in loader:
        x = batch["x"].to(device)
        obs_norm = batch["obs_norm"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        tile_ids = batch["tile_id"].tolist()
        with torch.set_grad_enabled(train_mode):
            pred = model(x)
            forward = forward_two_layer_torch(pred, g_load_fft, g_poro_fft, config.physics)
            forward_norm = (forward - obs_mean) / max(obs_std, 1e-6)
            loss_forward = masked_mse(forward_norm, obs_norm, valid_mask)
            loss_spatial_s0 = anisotropic_total_variation(pred[:, 0])
            loss_spatial_sg = anisotropic_total_variation(pred[:, 1])
            temporal_s0 = torch.tensor(0.0, device=device)
            temporal_sg = torch.tensor(0.0, device=device)
            temporal_count = 0
            for batch_idx, tile_id in enumerate(tile_ids):
                previous = prev_by_tile.get(int(tile_id))
                if previous is not None:
                    temporal_s0 = temporal_s0 + torch.mean(torch.abs(pred[batch_idx, 0] - previous[0]))
                    temporal_sg = temporal_sg + torch.mean(torch.abs(pred[batch_idx, 1] - previous[1]))
                    temporal_count += 1
                prev_by_tile[int(tile_id)] = pred[batch_idx].detach()
            if temporal_count > 0:
                temporal_s0 = temporal_s0 / temporal_count
                temporal_sg = temporal_sg / temporal_count
            loss = (
                config.training.lambda_forward * loss_forward
                + config.training.lambda_spatial_s0 * loss_spatial_s0
                + config.training.lambda_spatial_sg * loss_spatial_sg
                + config.training.lambda_temporal_s0 * temporal_s0
                + config.training.lambda_temporal_sg * temporal_sg
            )
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        totals["loss"] += float(loss.item())
        totals["forward"] += float(loss_forward.item())
        totals["spatial_s0"] += float(loss_spatial_s0.item())
        totals["spatial_sg"] += float(loss_spatial_sg.item())
        totals["temporal_s0"] += float(temporal_s0.item())
        totals["temporal_sg"] += float(temporal_sg.item())
        totals["n_batches"] += 1.0

    denom = max(totals["n_batches"], 1.0)
    return {key: (value / denom if key != "n_batches" else value) for key, value in totals.items()}


def train_model(config: InversionConfig) -> dict:
    output_dir = Path(config.training.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stack, train_ds, val_ds, meta = build_datasets(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle_train,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    model = _build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, config.training.epochs + 1):
        train_stats = _run_epoch(model, train_loader, device, config, meta["obs_mean"], meta["obs_std"], optimizer)
        val_stats = _run_epoch(model, val_loader, device, config, meta["obs_mean"], meta["obs_std"], None)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_forward": train_stats["forward"],
            "train_spatial_s0": train_stats["spatial_s0"],
            "train_spatial_sg": train_stats["spatial_sg"],
            "train_temporal_s0": train_stats["temporal_s0"],
            "train_temporal_sg": train_stats["temporal_sg"],
            "val_loss": val_stats["loss"],
            "val_forward": val_stats["forward"],
            "val_spatial_s0": val_stats["spatial_s0"],
            "val_spatial_sg": val_stats["spatial_sg"],
            "val_temporal_s0": val_stats["temporal_s0"],
            "val_temporal_sg": val_stats["temporal_sg"],
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        best_state = deepcopy(model.state_dict())

    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "model_state": best_state,
            "config": config.to_dict(),
            "normalization": {
                "input_mean": meta["input_mean"],
                "input_std": meta["input_std"],
                "obs_mean": meta["obs_mean"],
                "obs_std": meta["obs_std"],
            },
            "data_summary": {
                "shape": tuple(int(v) for v in stack.deformation.shape),
                "n_samples": meta["n_samples"],
                "n_tiles": meta["n_tiles"],
            },
        },
        checkpoint_path,
    )

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    (output_dir / "resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2))

    summary = {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(output_dir / "history.csv"),
        "best_val_loss": best_val,
        **meta,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary

