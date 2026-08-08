"""Shared training-loss configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SharedTrainingLossConfig:
    """Loss controls common to direct and FSB surrogate trainers."""

    use_volume_weighted_mse: bool = False
    volume_weight_alpha: float = 0.5
    volume_weight_max: float = 1000.0
    wall_layers: Optional[int] = None
    wall_mse_weight: float = 0.0
    wall_mse_layers: int = 5
    wall_cp_weight: float = 0.0
    wall_cp_reduction: str = "smooth_l1"
    gradient_loss_weight: float = 0.0
    gradient_loss_wall_layers: Optional[int] = None
    flow_perceptual_weight: float = 0.0
    flow_perceptual_level_weights: Optional[list[float]] = None
    flow_perceptual_n_levels: int = 3
    flow_perceptual_warmup_epochs: int = 0
    flow_perceptual_warmup_center: int = 0
    flow_perceptual_warmup_sharpness: int = 1
    flow_perceptual_encoder_checkpoint: Optional[str] = None
    flow_perceptual_encoder_config: Optional[dict[str, Any]] = None
    flow_perceptual_manifold_prior: Optional[dict[str, Any]] = None
    res_loss_enabled: bool = False
    res_loss_weight: float = 0.0
    res_loss_reduction: str = "l1"
    res_loss_warmup_epochs: int = 0
    res_loss_warmup_center: int = 0
    res_loss_warmup_sharpness: int = 1
    res_field_scale_mode: str = "none"
    res_field_eps: float = 1.0e-12
    res_field_norm: str = "l2_sq"
    res_field_huber_delta: float = 1.0
    res_loss_weights: Optional[dict[str, float]] = None
    res_loss_wall_layers: Optional[int] = None
    res_loss_norm_mode: str = "standard"
    res_loss_dtype: str = "float32"
    residual_smoothing_enabled: bool = False
    residual_smoothing_lambda_eta: float = 4.0
    residual_smoothing_lambda_xi: float = 1.0

    def validate(self) -> None:
        self.use_volume_weighted_mse = bool(self.use_volume_weighted_mse)
        self.volume_weight_alpha = float(self.volume_weight_alpha)
        self.volume_weight_max = float(self.volume_weight_max)
        self.wall_mse_weight = float(self.wall_mse_weight)
        self.wall_mse_layers = int(self.wall_mse_layers)
        self.wall_cp_weight = float(self.wall_cp_weight)
        self.wall_cp_reduction = str(self.wall_cp_reduction).lower()
        self.gradient_loss_weight = float(self.gradient_loss_weight)
        self.flow_perceptual_weight = float(self.flow_perceptual_weight)
        self.flow_perceptual_n_levels = int(self.flow_perceptual_n_levels)
        self.flow_perceptual_warmup_epochs = int(self.flow_perceptual_warmup_epochs)
        self.flow_perceptual_warmup_center = int(self.flow_perceptual_warmup_center)
        self.flow_perceptual_warmup_sharpness = int(self.flow_perceptual_warmup_sharpness)
        self.res_loss_enabled = bool(self.res_loss_enabled)
        self.res_loss_weight = float(self.res_loss_weight)
        self.res_loss_reduction = str(self.res_loss_reduction).lower()
        self.res_loss_warmup_epochs = int(self.res_loss_warmup_epochs)
        self.res_loss_warmup_center = int(self.res_loss_warmup_center)
        self.res_loss_warmup_sharpness = int(self.res_loss_warmup_sharpness)
        self.res_field_scale_mode = str(self.res_field_scale_mode).lower()
        self.res_field_eps = float(self.res_field_eps)
        self.res_field_norm = str(self.res_field_norm).lower()
        self.res_field_huber_delta = float(self.res_field_huber_delta)
        self.res_loss_norm_mode = str(self.res_loss_norm_mode).lower()
        self.res_loss_dtype = str(self.res_loss_dtype).lower()
        self.residual_smoothing_enabled = bool(self.residual_smoothing_enabled)
        self.residual_smoothing_lambda_eta = float(self.residual_smoothing_lambda_eta)
        self.residual_smoothing_lambda_xi = float(self.residual_smoothing_lambda_xi)

        if self.volume_weight_alpha < 0:
            raise ValueError("training.loss.volume_weight_alpha must be non-negative")
        if self.volume_weight_max <= 0:
            raise ValueError("training.loss.volume_weight_max must be positive")
        if self.wall_layers is not None:
            self.wall_layers = int(self.wall_layers)
            if self.wall_layers <= 0:
                raise ValueError("training.loss.wall_layers must be positive when set")
        if self.wall_mse_weight < 0 or self.wall_mse_layers < 0:
            raise ValueError("training.loss wall MSE controls must be non-negative")
        if self.wall_cp_weight < 0:
            raise ValueError("training.loss.wall_cp_weight must be non-negative")
        if self.wall_cp_reduction not in {"l1", "l2", "smooth_l1", "huber"}:
            raise ValueError("training.loss.wall_cp_reduction must be l1, l2, smooth_l1, or huber")
        if self.gradient_loss_weight < 0:
            raise ValueError("training.loss.gradient_loss_weight must be non-negative")
        if self.gradient_loss_wall_layers is not None:
            self.gradient_loss_wall_layers = int(self.gradient_loss_wall_layers)
            if self.gradient_loss_wall_layers <= 0:
                raise ValueError("training.loss.gradient_loss_wall_layers must be positive when set")
        if self.flow_perceptual_weight < 0:
            raise ValueError("training.loss.flow_perceptual_weight must be non-negative")
        if self.flow_perceptual_level_weights is not None:
            self.flow_perceptual_level_weights = [float(v) for v in self.flow_perceptual_level_weights]
        if self.flow_perceptual_n_levels <= 0:
            raise ValueError("training.loss.flow_perceptual_n_levels must be positive")
        if self.flow_perceptual_warmup_sharpness <= 0:
            raise ValueError("training.loss.flow_perceptual_warmup_sharpness must be positive")
        if self.res_loss_weight < 0:
            raise ValueError("training.loss.res_loss_weight must be non-negative")
        if self.res_loss_reduction not in {"l1", "l2", "log1p_l1"}:
            raise ValueError("training.loss.res_loss_reduction must be l1, l2, or log1p_l1")
        if self.res_loss_warmup_sharpness <= 0:
            raise ValueError("training.loss.res_loss_warmup_sharpness must be positive")
        if self.res_field_scale_mode not in {"none", "raw", "baseline", "baseline_eq_rms"}:
            raise ValueError("training.loss.res_field_scale_mode is invalid")
        if self.res_field_norm not in {"l2_sq", "l2sq", "mse", "l1", "mae", "l2", "rms", "huber"}:
            raise ValueError("training.loss.res_field_norm is invalid")
        if self.res_field_eps <= 0:
            raise ValueError("training.loss.res_field_eps must be positive")
        if self.res_field_huber_delta <= 0:
            raise ValueError("training.loss.res_field_huber_delta must be positive")
        if self.res_loss_wall_layers is not None:
            self.res_loss_wall_layers = int(self.res_loss_wall_layers)
            if self.res_loss_wall_layers <= 0:
                raise ValueError("training.loss.res_loss_wall_layers must be positive when set")
        if self.residual_smoothing_lambda_eta < 0 or self.residual_smoothing_lambda_xi < 0:
            raise ValueError("training.loss residual smoothing lambdas must be non-negative")

    def to_shared_loss_kwargs(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in SharedTrainingLossConfig.__dataclass_fields__.keys()
        }


__all__ = ["SharedTrainingLossConfig"]
