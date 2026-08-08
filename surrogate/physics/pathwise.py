"""Pathwise residual objectives for differentiable FSB training."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor


_RESIDUAL_WEIGHT_DEFAULTS = {
    "continuity": 1.0,
    "momentum_x": 0.5,
    "momentum_y": 0.5,
    "energy": 1.0,
    "turbulence": 1.0,
}


def _ensure_bchw(x: Tensor) -> Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected a 3D/4D residual field, got shape {tuple(x.shape)}")


def build_residual_weight_vector(
    weights: Optional[Mapping[str, float]],
    *,
    n_channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    values = dict(_RESIDUAL_WEIGHT_DEFAULTS)
    values.update(dict(weights or {}))
    if "momentum" in values:
        values.setdefault("momentum_x", float(values["momentum"]))
        values.setdefault("momentum_y", float(values["momentum"]))
    ordered = [
        float(values.get("continuity", 1.0)),
        float(values.get("momentum_x", values.get("momentum", 0.5))),
        float(values.get("momentum_y", values.get("momentum", 0.5))),
        float(values.get("energy", 1.0)),
        float(values.get("turbulence", 1.0)),
    ]
    return torch.tensor(ordered[: int(n_channels)], device=device, dtype=dtype)


def compute_eq_rms_scales(
    signed_residual_field: Tensor,
    *,
    wall_layers: Optional[int],
    eps: float = 1.0e-12,
) -> Tensor:
    field = _ensure_bchw(signed_residual_field)
    if wall_layers is not None:
        field = field[:, :, : int(wall_layers), :]
    return torch.sqrt(field.pow(2).mean(dim=(2, 3)) + float(eps))


def compute_residual_field_objective(
    signed_residual_field: Tensor,
    *,
    weights: Optional[Mapping[str, float]],
    wall_layers: Optional[int],
    scale0: Optional[Tensor] = None,
    norm: str = "l2_sq",
    huber_delta: float = 1.0,
    eps: float = 1.0e-12,
) -> Tensor:
    field = _ensure_bchw(signed_residual_field)
    if wall_layers is not None:
        field = field[:, :, : int(wall_layers), :]

    if scale0 is not None:
        scale = scale0.to(device=field.device, dtype=field.dtype).clamp_min(float(eps))
        if scale.ndim != 2:
            raise ValueError(f"scale0 must be (B, C), got shape {tuple(scale.shape)}")
        field = field / scale[:, :, None, None]

    weight = build_residual_weight_vector(
        weights,
        n_channels=field.shape[1],
        device=field.device,
        dtype=field.dtype,
    )
    norm_name = str(norm).lower()
    if norm_name in {"l2_sq", "l2sq", "mse"}:
        return (field.pow(2) * weight[None, :, None, None]).mean(dim=(2, 3)).sum(dim=1)
    if norm_name in {"l1", "mae"}:
        return (field.abs() * weight[None, :, None, None]).mean(dim=(2, 3)).sum(dim=1)
    if norm_name in {"l2", "rms"}:
        return (torch.sqrt(field.pow(2).mean(dim=(2, 3)) + float(eps)) * weight[None, :]).sum(dim=1)
    if norm_name == "huber":
        delta = float(huber_delta)
        if delta <= 0.0:
            raise ValueError(f"huber_delta must be positive, got {delta}")
        abs_field = field.abs()
        quadratic = 0.5 * field.pow(2)
        linear = delta * (abs_field - 0.5 * delta)
        penalty = torch.where(abs_field <= delta, quadratic, linear)
        return (penalty * weight[None, :, None, None]).mean(dim=(2, 3)).sum(dim=1)
    raise ValueError(f"Unknown residual objective norm {norm!r}")


__all__ = [
    "build_residual_weight_vector",
    "compute_eq_rms_scales",
    "compute_residual_field_objective",
]
