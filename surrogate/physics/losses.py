"""Shared loss helpers used by direct and FSB trainers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def compute_volume_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    volumes: torch.Tensor,
    alpha: float = 0.5,
    w_max: float = 1000.0,
    wall_layers: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute MSE weighted by cell volume sensitivity."""
    eps = torch.tensor(1.0e-12, dtype=volumes.dtype, device=volumes.device)

    if volumes.ndim == 2:
        volumes = volumes.unsqueeze(0).expand(pred.shape[0], -1, -1)

    if wall_layers is not None:
        pred = pred[:, :, :wall_layers, :]
        target = target[:, :, :wall_layers, :]
        volumes = volumes[:, :wall_layers, :]

    vol_mean = volumes.mean()
    weights = torch.pow(vol_mean / (volumes + eps), alpha)
    weights = torch.clamp(weights, max=w_max)
    weights = weights / weights.mean()
    weights = weights.unsqueeze(1)

    weighted_mse = (weights * (pred - target) ** 2).mean()
    metrics = {
        "vw_w_min": weights.min().item(),
        "vw_w_max": weights.max().item(),
        "vw_w_mean": weights.mean().item(),
        "vw_w_std": weights.std().item(),
    }
    return weighted_mse, metrics


__all__ = ["compute_volume_weighted_mse"]
