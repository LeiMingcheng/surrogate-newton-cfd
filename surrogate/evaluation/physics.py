"""Physics-facing evaluation helpers.

These helpers evaluate predicted fields with physics utilities, but they do
not feed residuals or force data back into model conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from surrogate.physics.forces import ForceCoefficientsCalculator, compute_force_components_ogrid_torch


@dataclass
class ForceMetricsConfig:
    """Controls for aerodynamic force metric evaluation."""

    gamma: float = 1.4
    chord_ref: float = 1.0
    area_ref: float = 1.0
    moment_center: tuple[float, float] = (0.0, 0.0)
    compute_viscous: bool = True
    t_inf: float = 300.0


@dataclass
class ResidualMetricsConfig:
    """Controls for PDE residual metric evaluation."""

    targets: tuple[str, ...] = ("pred",)
    weights: Mapping[str, float] = field(default_factory=dict)
    wall_layers: Optional[int] = None
    spatial_wall_layers: Optional[int] = None
    periodic_xi: bool = True
    dtype: Optional[str] = None
    preserve_residual_dtype: bool = False
    state_is_adflow_consistent: bool = False
    state_is_adflow_mixed: bool = False
    return_components: bool = True


def _detach_cpu_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _mean_tensor(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().float().mean().item())
    return float(np.asarray(value, dtype=np.float64).mean())


def _as_float_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray([_detach_cpu_float(value) for value in values], dtype=np.float64)


def _scalar_record_value(value: Any) -> Optional[float]:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return float(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_force_config(config: Optional[ForceMetricsConfig | Mapping[str, Any]]) -> ForceMetricsConfig:
    if isinstance(config, ForceMetricsConfig):
        return config
    return ForceMetricsConfig(**dict(config or {}))


def compute_force_coefficients_from_fields(
    fields: torch.Tensor | np.ndarray,
    coords_vertex: torch.Tensor | np.ndarray,
    flow_conditions: torch.Tensor | np.ndarray,
    *,
    config: Optional[ForceMetricsConfig | Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute CL/CD/Cm for a batch of predicted or target fields."""
    force_config = _to_force_config(config)
    if isinstance(fields, torch.Tensor):
        coords_tensor = (
            coords_vertex.to(device=fields.device, dtype=fields.dtype)
            if isinstance(coords_vertex, torch.Tensor)
            else torch.as_tensor(coords_vertex, device=fields.device, dtype=fields.dtype)
        )
        flow_tensor = (
            flow_conditions.to(device=fields.device, dtype=fields.dtype)
            if isinstance(flow_conditions, torch.Tensor)
            else torch.as_tensor(flow_conditions, device=fields.device, dtype=fields.dtype)
        )
        return compute_force_components_ogrid_torch(
            fields,
            coords_tensor,
            flow_tensor,
            gamma=force_config.gamma,
            chord_ref=force_config.chord_ref,
            area_ref=force_config.area_ref,
            moment_center=force_config.moment_center,
            compute_viscous=force_config.compute_viscous,
            T_inf=force_config.t_inf,
        )

    calculator = ForceCoefficientsCalculator(
        gamma=force_config.gamma,
        chord_ref=force_config.chord_ref,
        area_ref=force_config.area_ref,
        moment_center=force_config.moment_center,
    )
    return calculator.compute_coefficients_batch(
        fields,
        coords_vertex,
        flow_conditions,
        compute_viscous=force_config.compute_viscous,
        T_inf=force_config.t_inf,
    )


def compute_force_metrics(
    pred_fields: torch.Tensor | np.ndarray,
    target_fields: torch.Tensor | np.ndarray,
    coords_vertex: torch.Tensor | np.ndarray,
    flow_conditions: torch.Tensor | np.ndarray,
    *,
    config: Optional[ForceMetricsConfig | Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Compute coefficient means and pred-vs-target force errors."""
    pred = compute_force_coefficients_from_fields(
        pred_fields,
        coords_vertex,
        flow_conditions,
        config=config,
    )
    target = compute_force_coefficients_from_fields(
        target_fields,
        coords_vertex,
        flow_conditions,
        config=config,
    )

    metrics: Dict[str, float] = {}
    for key in ("CL", "CD", "Cm"):
        pred_value = pred[key]
        target_value = target[key]
        if isinstance(pred_value, torch.Tensor) or isinstance(target_value, torch.Tensor):
            pred_tensor = pred_value if isinstance(pred_value, torch.Tensor) else torch.as_tensor(pred_value)
            target_tensor = target_value if isinstance(target_value, torch.Tensor) else torch.as_tensor(target_value, device=pred_tensor.device)
            error = pred_tensor - target_tensor
            metrics[f"force_pred_{key}_mean"] = _mean_tensor(pred_tensor)
            metrics[f"force_target_{key}_mean"] = _mean_tensor(target_tensor)
            metrics[f"force_mae_{key}"] = _detach_cpu_float(torch.mean(torch.abs(error)))
            metrics[f"force_rmse_{key}"] = _detach_cpu_float(torch.sqrt(torch.mean(error ** 2)))
        else:
            pred_array = np.asarray(pred_value, dtype=np.float64)
            target_array = np.asarray(target_value, dtype=np.float64)
            error = pred_array - target_array
            metrics[f"force_pred_{key}_mean"] = float(pred_array.mean())
            metrics[f"force_target_{key}_mean"] = float(target_array.mean())
            metrics[f"force_mae_{key}"] = float(np.mean(np.abs(error)))
            metrics[f"force_rmse_{key}"] = float(np.sqrt(np.mean(error ** 2)))
    return metrics


def compute_residual_batch_metrics(
    residual_calculator: Any,
    samples: Iterable[Mapping[str, Any]],
) -> Dict[str, float]:
    """Aggregate scalar residual scores from a residual calculator."""
    metrics, _ = compute_residual_batch_evaluation(
        residual_calculator,
        samples,
        prefix="residual",
        include_records=False,
    )
    return metrics


def compute_residual_batch_evaluation(
    residual_calculator: Any,
    samples: Iterable[Mapping[str, Any]],
    *,
    prefix: str = "residual",
    include_records: bool = False,
) -> tuple[Dict[str, float], list[Dict[str, float]]]:
    """Aggregate residual metrics and optionally keep per-sample scalar records."""
    sample_list = [dict(sample) for sample in samples]
    if not sample_list:
        raise ValueError("compute_residual_batch_evaluation requires at least one sample")

    scores, results = residual_calculator.compute_batch_residuals(sample_list)
    score_array = _as_float_array(scores)
    metrics = {
        f"{prefix}_score_mean": float(score_array.mean()),
        f"{prefix}_score_std": float(score_array.std()),
        f"{prefix}_score_min": float(score_array.min()),
        f"{prefix}_score_max": float(score_array.max()),
    }
    records: list[Dict[str, float]] = []
    if include_records:
        for score, result in zip(score_array, results):
            record: Dict[str, float] = {f"{prefix}_score": float(score)}
            if isinstance(result, Mapping):
                for key, value in result.items():
                    scalar = _scalar_record_value(value)
                    if scalar is not None:
                        record[f"{prefix}_{key}"] = scalar
            records.append(record)
    return metrics, records


def compute_residual_pair_metrics(
    pred_scores: Iterable[Any],
    target_scores: Iterable[Any],
    *,
    prefix: str = "residual",
) -> Dict[str, float]:
    """Compare predicted-field residual scores against target-field scores."""
    pred_array = _as_float_array(pred_scores)
    target_array = _as_float_array(target_scores)
    if pred_array.shape != target_array.shape:
        raise ValueError("pred_scores and target_scores must have the same shape")
    diff = pred_array - target_array
    return {
        f"{prefix}_score_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_score_rmse": float(np.sqrt(np.mean(diff ** 2))),
        f"{prefix}_score_bias": float(np.mean(diff)),
    }


__all__ = [
    "ForceMetricsConfig",
    "ResidualMetricsConfig",
    "compute_force_coefficients_from_fields",
    "compute_force_metrics",
    "compute_residual_batch_evaluation",
    "compute_residual_batch_metrics",
    "compute_residual_pair_metrics",
]
