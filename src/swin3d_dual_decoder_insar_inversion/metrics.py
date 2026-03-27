from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weight = mask.to(dtype=pred.dtype)
    denom = weight.sum().clamp_min(eps)
    return ((pred - target).square() * weight).sum() / denom


def anisotropic_total_variation(field: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(field[..., :, 1:] - field[..., :, :-1]).mean()
    dy = torch.abs(field[..., 1:, :] - field[..., :-1, :]).mean()
    return dx + dy

