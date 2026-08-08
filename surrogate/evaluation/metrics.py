"""Field-level evaluation metrics for surrogate predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from surrogate.physics.losses import compute_volume_weighted_mse


@dataclass
class FieldMetricsConfig:
    """Controls for field error metrics."""

    use_volume_weighted_mse: bool = False
    volume_weight_alpha: float = 0.5
    volume_weight_max: float = 1000.0
    wall_layers: Optional[int] = None
    channel_names: Optional[Sequence[str]] = None


def _crop_wall_layers(
    pred: torch.Tensor,
    target: torch.Tensor,
    wall_layers: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if wall_layers is None:
        return pred, target
    layers = min(max(int(wall_layers), 1), int(pred.shape[-2]))
    return pred[:, :, :layers, :], target[:, :, :layers, :]


def _metric_value(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _resolve_volumes(batch: Optional[Mapping[str, Any]], volumes: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if isinstance(volumes, torch.Tensor):
        return volumes
    if batch is None:
        return None
    candidate = batch.get("cell_volumes")
    if candidate is None:
        candidate = batch.get("volumes")
    return candidate if isinstance(candidate, torch.Tensor) else None


def compute_field_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    batch: Optional[Mapping[str, Any]] = None,
    volumes: Optional[torch.Tensor] = None,
    config: Optional[FieldMetricsConfig | Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Compute field error metrics without mutating training/inference state."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}")

    if isinstance(config, FieldMetricsConfig):
        metric_config = config
    else:
        metric_config = FieldMetricsConfig(**dict(config or {}))

    pred_for_loss, target_for_loss = _crop_wall_layers(pred, target, metric_config.wall_layers)
    diff = pred_for_loss - target_for_loss
    metrics: Dict[str, float] = {
        "mse": _metric_value(torch.mean(diff ** 2)),
        "rmse": _metric_value(torch.sqrt(torch.mean(diff ** 2))),
        "mae": _metric_value(torch.mean(torch.abs(diff))),
        "max_abs": _metric_value(torch.max(torch.abs(diff))),
    }

    channel_names = list(metric_config.channel_names or [])
    for channel_idx in range(int(pred_for_loss.shape[1])):
        name = channel_names[channel_idx] if channel_idx < len(channel_names) else f"channel_{channel_idx}"
        channel_diff = diff[:, channel_idx]
        metrics[f"mse_{name}"] = _metric_value(torch.mean(channel_diff ** 2))

    volume_tensor = _resolve_volumes(batch, volumes)
    if metric_config.use_volume_weighted_mse and isinstance(volume_tensor, torch.Tensor):
        vw_loss, vw_metrics = compute_volume_weighted_mse(
            pred,
            target,
            volume_tensor.to(device=pred.device, dtype=pred.dtype),
            alpha=metric_config.volume_weight_alpha,
            w_max=metric_config.volume_weight_max,
            wall_layers=metric_config.wall_layers,
        )
        metrics["volume_weighted_mse"] = _metric_value(vw_loss)
        metrics.update({key: float(value) for key, value in vw_metrics.items()})

    return metrics


__all__ = [
    "FieldMetricsConfig",
    "compute_field_metrics",
]
