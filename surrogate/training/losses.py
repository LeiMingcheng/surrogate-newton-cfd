"""Shared loss terms used by direct and FSB surrogate trainers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from surrogate.common.components import load_state_dict_with_stability_head_compat
from surrogate.models.flow_perceptual import FlowPerceptualDecoder, FlowPerceptualEncoder
from surrogate.physics.forces import extract_wall_cp_ogrid_torch
from surrogate.physics.losses import compute_volume_weighted_mse
from surrogate.physics.pathwise import compute_eq_rms_scales, compute_residual_field_objective
from surrogate.physics.pde.geometry import compute_cell_volume_adflow
from surrogate.physics.residual import get_residual_calculator
from surrogate.physics.smoothing import SobolevResidualSmoother, SobolevSmoothingConfig
from surrogate.training.loss_config import SharedTrainingLossConfig


class OGridPad(nn.Module):
    """O-grid padding: circular along xi and reflect along eta."""

    def __init__(self, pad: int) -> None:
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.pad
        if pad <= 0:
            return x
        x = F.pad(x, (pad, pad, 0, 0), mode="circular")
        return F.pad(x, (0, 0, pad, pad), mode="reflect")


def compute_gradient_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wall_layers: Optional[int] = None,
) -> torch.Tensor:
    """Compute finite-difference gradient alignment loss."""

    if wall_layers is not None:
        pred = pred[:, :, :wall_layers, :]
        target = target[:, :, :wall_layers, :]

    pred_dxi = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dxi = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_deta = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_deta = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.mse_loss(pred_dxi, target_dxi) + F.mse_loss(pred_deta, target_deta)


class PhysicsDerivedFeatures(nn.Module):
    """Differentiable physics-derived channels for flow-perceptual loss."""

    def __init__(self, gamma: float = 1.4) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.ogrid_pad = OGridPad(1)

    def forward(self, fields: torch.Tensor, flow_conditions: torch.Tensor) -> torch.Tensor:
        rho = fields[:, 0:1].clamp(min=1.0e-6)
        u = fields[:, 1:2]
        v = fields[:, 2:3]
        pressure = fields[:, 3:4].clamp(min=1.0e-6)

        velocity = torch.sqrt(u.pow(2) + v.pow(2) + 1.0e-8)
        sound_speed = torch.sqrt(self.gamma * pressure / rho + 1.0e-8)
        local_mach = velocity / (sound_speed + 1.0e-8)
        entropy = torch.log(pressure / (rho.pow(self.gamma) + 1.0e-8) + 1.0e-8)

        mach_inf = flow_conditions[:, 0:1].unsqueeze(-1).unsqueeze(-1).clamp(min=0.01)
        cp = (pressure - 1.0) / (0.5 * self.gamma * mach_inf.pow(2) + 1.0e-8)

        pressure_padded = self.ogrid_pad(pressure)
        dp_dxi = (pressure_padded[:, :, 1:-1, 2:] - pressure_padded[:, :, 1:-1, :-2]) / 2.0
        dp_deta = (pressure_padded[:, :, 2:, 1:-1] - pressure_padded[:, :, :-2, 1:-1]) / 2.0
        grad_pressure = torch.sqrt(dp_dxi.pow(2) + dp_deta.pow(2) + 1.0e-8)

        v_padded = self.ogrid_pad(v)
        u_padded = self.ogrid_pad(u)
        dv_dxi = (v_padded[:, :, 1:-1, 2:] - v_padded[:, :, 1:-1, :-2]) / 2.0
        du_deta = (u_padded[:, :, 2:, 1:-1] - u_padded[:, :, :-2, 1:-1]) / 2.0
        vorticity = dv_dxi - du_deta

        return torch.cat([velocity, local_mach, entropy, cp, grad_pressure, vorticity], dim=1)


def _make_gaussian_kernel(sigma: float, kernel_size: int) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (coords / float(sigma)).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)


class MultiScaleFeaturePyramid(nn.Module):
    """Fixed multi-scale feature pyramid used by flow-perceptual loss."""

    def __init__(self, n_levels: int = 3) -> None:
        super().__init__()
        self.n_levels = int(n_levels)
        self.ogrid_pad = OGridPad(1)
        if self.n_levels >= 2:
            self.register_buffer("gauss_kernel_1", _make_gaussian_kernel(1.5, 5).unsqueeze(0).unsqueeze(0))
        if self.n_levels >= 3:
            self.register_buffer("gauss_kernel_2", _make_gaussian_kernel(3.0, 9).unsqueeze(0).unsqueeze(0))

    def _sobel_features(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = self.ogrid_pad(x)
        dx = (x_pad[:, :, 1:-1, 2:] - x_pad[:, :, 1:-1, :-2]) / 2.0
        dy = (x_pad[:, :, 2:, 1:-1] - x_pad[:, :, :-2, 1:-1]) / 2.0
        return torch.cat([dx, dy], dim=1)

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        kernel = kernel.to(device=x.device, dtype=x.dtype)
        channels = x.shape[1]
        kernel = kernel.expand(channels, -1, -1, -1)
        pad_h = (kernel.shape[2] - 1) // 2
        pad_w = (kernel.shape[3] - 1) // 2
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="reflect")
        return F.conv2d(x, kernel, groups=channels)

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        levels = [torch.cat([features, self._sobel_features(features)], dim=1)]
        if self.n_levels >= 2:
            blurred = self._gaussian_blur(features, self.gauss_kernel_1)
            down = F.avg_pool2d(blurred, kernel_size=2, stride=2)
            levels.append(torch.cat([down, self._sobel_features(down)], dim=1))
        if self.n_levels >= 3:
            blurred = self._gaussian_blur(features, self.gauss_kernel_2)
            down = F.avg_pool2d(blurred, kernel_size=4, stride=4)
            levels.append(torch.cat([down, self._sobel_features(down)], dim=1))
        return levels


class FlowPerceptualLoss(nn.Module):
    """Physics-aware feature loss shared by direct and FSB training."""

    def __init__(
        self,
        *,
        level_weights: Optional[list[float]] = None,
        n_levels: int = 3,
        normalizer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.physics_features = PhysicsDerivedFeatures()
        self.pyramid = MultiScaleFeaturePyramid(n_levels=n_levels)
        self.level_weights = list(level_weights or [1.0, 0.5, 0.25])
        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.use_encoder = False
        self.use_manifold_prior = False
        self.lambda_prior = 0.1
        self.lambda_stay = 0.05

    @staticmethod
    def _instance_normalize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + 1.0e-6
        return (x - mean) / std

    def _compute_prior_loss(self, pred_input: torch.Tensor, flow_conditions: torch.Tensor) -> torch.Tensor:
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("FlowPerceptualLoss prior requires encoder and decoder")
        features = self.encoder.extract_features(pred_input, flow_conditions)
        reconstructed = self.decoder(features)
        return F.l1_loss(reconstructed, pred_input)

    def _compute_stay_loss(
        self,
        pred_input: torch.Tensor,
        original_input: torch.Tensor,
        flow_conditions: torch.Tensor,
    ) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("FlowPerceptualLoss stay loss requires encoder")
        pred_features = self.encoder.extract_features(pred_input, flow_conditions)
        with torch.no_grad():
            original_features = self.encoder.extract_features(original_input, flow_conditions)
        return F.l1_loss(pred_features[-1], original_features[-1])

    def forward(
        self,
        *,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords_center: Optional[torch.Tensor] = None,
        original_norm: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.normalizer is not None:
            pred_phys = self.normalizer.inverse_transform(pred_norm)
            target_phys = self.normalizer.inverse_transform(target_norm)
        else:
            pred_phys = pred_norm
            target_phys = target_norm

        pred_features = self.physics_features(pred_phys, flow_conditions)
        target_features = self.physics_features(target_phys, flow_conditions)

        if self.use_encoder and self.encoder is not None:
            pred_input = torch.cat([pred_phys, pred_features], dim=1)
            target_input = torch.cat([target_phys, target_features], dim=1)
            if coords_center is not None:
                pred_input = torch.cat([pred_input, coords_center], dim=1)
                target_input = torch.cat([target_input, coords_center], dim=1)
            pred_levels = self.encoder(pred_input, flow_conditions)
            with torch.no_grad():
                target_levels = self.encoder(target_input, flow_conditions)
        else:
            pred_levels = self.pyramid(pred_features)
            with torch.no_grad():
                target_levels = self.pyramid(target_features)

        total = torch.tensor(0.0, device=pred_norm.device, dtype=pred_norm.dtype)
        metrics: Dict[str, torch.Tensor] = {}
        for index, (pred_level, target_level) in enumerate(zip(pred_levels, target_levels)):
            weight = self.level_weights[index] if index < len(self.level_weights) else 1.0
            level_loss = F.l1_loss(
                self._instance_normalize(pred_level),
                self._instance_normalize(target_level),
            )
            total = total + float(weight) * level_loss
            metrics[f"fpl_level_{index}"] = level_loss.detach()

        if self.use_manifold_prior and self.decoder is not None and self.encoder is not None:
            pred_input_full = torch.cat([pred_phys, pred_features], dim=1)
            if coords_center is not None:
                pred_input_full = torch.cat([pred_input_full, coords_center], dim=1)
            prior_loss = self._compute_prior_loss(pred_input_full, flow_conditions)
            total = total + self.lambda_prior * prior_loss
            metrics["fpl_prior"] = prior_loss.detach()

            if original_norm is not None:
                if self.normalizer is not None:
                    original_phys = self.normalizer.inverse_transform(original_norm)
                else:
                    original_phys = original_norm
                original_features = self.physics_features(original_phys, flow_conditions)
                original_input_full = torch.cat([original_phys, original_features], dim=1)
                if coords_center is not None:
                    original_input_full = torch.cat([original_input_full, coords_center], dim=1)
                stay_loss = self._compute_stay_loss(pred_input_full, original_input_full, flow_conditions)
                total = total + self.lambda_stay * stay_loss
                metrics["fpl_stay"] = stay_loss.detach()

        metrics["fpl_total"] = total.detach()
        return total, metrics


class SharedTrainingLosses:
    """Common differentiable training losses for direct and FSB models."""

    def __init__(
        self,
        config: SharedTrainingLossConfig,
        *,
        normalizer: Optional[torch.nn.Module],
        device: str | torch.device,
        global_step_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self.config = config
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.global_step_fn = global_step_fn or (lambda: 0)
        self.flow_perceptual_loss = self._build_flow_perceptual_loss()
        self.residual_calculator = self._build_residual_calculator()
        self.residual_smoother = self._build_residual_smoother()

    @staticmethod
    def volumes_from(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        volumes = batch.get("cell_volumes")
        if volumes is None:
            volumes = batch.get("volumes")
        return volumes if isinstance(volumes, torch.Tensor) else None

    @staticmethod
    def coords_vertex_from(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        coords_vertex = batch.get("coords_vertex")
        return coords_vertex if isinstance(coords_vertex, torch.Tensor) else None

    def crop_wall_layers(self, tensor: torch.Tensor, layers: Optional[int] = None) -> torch.Tensor:
        crop_layers = self.config.wall_layers if layers is None else layers
        if crop_layers is None:
            return tensor
        return tensor[..., : int(crop_layers), :]

    def cell_volumes_from(self, batch: Dict[str, Any], pred: torch.Tensor) -> Optional[torch.Tensor]:
        volumes = self.volumes_from(batch)
        if volumes is not None:
            volumes = volumes.to(device=pred.device, dtype=pred.dtype)
            if volumes.shape[-2] != pred.shape[-2]:
                volumes = volumes[..., : pred.shape[-2], :]
            return volumes
        coords_vertex = self.coords_vertex_from(batch)
        if coords_vertex is None:
            return None
        volumes, _ = compute_cell_volume_adflow(coords_vertex.to(device=pred.device), periodic_xi=True)
        volumes = volumes.to(device=pred.device, dtype=pred.dtype)
        if volumes.shape[-2] != pred.shape[-2]:
            volumes = volumes[..., : pred.shape[-2], :]
        return volumes

    @staticmethod
    def compute_weighted_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if loss_weight is None:
            return F.mse_loss(pred, target, reduction="mean")

        weight = loss_weight.to(device=pred.device, dtype=pred.dtype)
        if weight.ndim != 4:
            raise ValueError(f"loss_weight must be 4D, got {weight.ndim}D")
        if weight.shape[1] == 1 and pred.shape[1] != 1:
            weight = weight.expand(-1, pred.shape[1], -1, -1)
        elif weight.shape[1] != pred.shape[1]:
            raise ValueError(
                f"loss_weight channel count {weight.shape[1]} does not match pred channels {pred.shape[1]}"
            )

        weight = weight / weight.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-12)
        return (weight * (pred - target).pow(2)).mean()

    def compute_supervised_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        loss_weight: Optional[torch.Tensor] = None,
        use_l1: bool = False,
    ) -> torch.Tensor:
        if use_l1:
            return F.l1_loss(pred, target, reduction="mean")
        return self.compute_weighted_mse(pred, target, loss_weight)

    def compute_main_metric_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        batch: Optional[Dict[str, Any]] = None,
        loss_weight: Optional[torch.Tensor] = None,
        use_l1: bool = False,
    ) -> torch.Tensor:
        if use_l1:
            return F.l1_loss(pred, target, reduction="mean")
        if self.config.use_volume_weighted_mse and batch is not None:
            volumes = self.cell_volumes_from(batch, pred)
            if volumes is not None:
                vw_loss, _ = compute_volume_weighted_mse(
                    pred,
                    target,
                    volumes,
                    alpha=self.config.volume_weight_alpha,
                    w_max=self.config.volume_weight_max,
                    wall_layers=None,
                )
                return vw_loss
        return self.compute_weighted_mse(pred, target, loss_weight)

    def compute_physical_supervised_loss(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        *,
        loss_weight: Optional[torch.Tensor] = None,
        use_l1: bool = False,
    ) -> torch.Tensor:
        if self.normalizer is not None:
            pred_phys = self.normalizer.inverse_transform(pred_norm)
            target_phys = self.normalizer.inverse_transform(target_norm)
        else:
            pred_phys = pred_norm
            target_phys = target_norm
        return self.compute_supervised_loss(
            pred_phys,
            target_phys,
            loss_weight=loss_weight,
            use_l1=use_l1,
        )

    def compute_physical_main_metric_loss(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        *,
        batch: Optional[Dict[str, Any]] = None,
        loss_weight: Optional[torch.Tensor] = None,
        use_l1: bool = False,
    ) -> torch.Tensor:
        if self.normalizer is not None:
            pred_phys = self.normalizer.inverse_transform(pred_norm)
            target_phys = self.normalizer.inverse_transform(target_norm)
        else:
            pred_phys = pred_norm
            target_phys = target_norm
        return self.compute_main_metric_loss(
            pred_phys,
            target_phys,
            batch=batch,
            loss_weight=loss_weight,
            use_l1=use_l1,
        )

    def _build_flow_perceptual_loss(self) -> Optional[FlowPerceptualLoss]:
        if self.config.flow_perceptual_weight <= 0.0:
            return None
        module = FlowPerceptualLoss(
            level_weights=self.config.flow_perceptual_level_weights,
            n_levels=self.config.flow_perceptual_n_levels,
            normalizer=self.normalizer,
        ).to(self.device)
        ckpt_path = self.config.flow_perceptual_encoder_checkpoint
        if ckpt_path:
            path = Path(ckpt_path)
            if not path.exists():
                raise FileNotFoundError(f"Flow-perceptual encoder checkpoint not found: {path}")
            payload = torch.load(str(path), map_location=self.device, weights_only=False)
            encoder_config = dict(self.config.flow_perceptual_encoder_config or payload.get("config", {}) or {})
            encoder = FlowPerceptualEncoder(encoder_config)
            load_state_dict_with_stability_head_compat(
                encoder,
                payload["encoder_state_dict"],
                context=f"flow_perceptual_encoder {path}",
            )
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad_(False)
            module.encoder = encoder.to(self.device)
            module.use_encoder = True

            manifold_prior = dict(self.config.flow_perceptual_manifold_prior or {})
            if bool(manifold_prior.get("enabled", False)):
                decoder = FlowPerceptualDecoder(encoder_config)
                load_state_dict_with_stability_head_compat(
                    decoder,
                    payload["decoder_state_dict"],
                    context=f"flow_perceptual_decoder {path}",
                )
                decoder.eval()
                for param in decoder.parameters():
                    param.requires_grad_(False)
                module.decoder = decoder.to(self.device)
                module.use_manifold_prior = True
                module.lambda_prior = float(manifold_prior.get("lambda_prior", 0.1))
                module.lambda_stay = float(manifold_prior.get("lambda_stay", 0.05))
        return module

    def _build_residual_calculator(self) -> Any:
        if not self.config.res_loss_enabled or self.config.res_loss_weight <= 0.0:
            return None
        return get_residual_calculator(
            backend="torch",
            device=str(self.device),
            residual_norm_mode=self.config.res_loss_norm_mode,
            force_new=False,
        )

    def _build_residual_smoother(self) -> Optional[SobolevResidualSmoother]:
        if not self.config.residual_smoothing_enabled:
            return None
        return SobolevResidualSmoother(
            SobolevSmoothingConfig(
                lambda_eta=float(self.config.residual_smoothing_lambda_eta),
                lambda_xi=float(self.config.residual_smoothing_lambda_xi),
            )
        )

    def warmup_weight(
        self,
        *,
        base_weight: float,
        enabled: bool,
        epoch: int,
        warmup_epochs: int,
        warmup_center: int,
        warmup_sharpness: int,
    ) -> float:
        if not enabled or base_weight <= 0.0:
            return 0.0
        if int(epoch) < int(warmup_epochs):
            return 0.0
        x = (int(self.global_step_fn()) - int(warmup_center)) / max(float(warmup_sharpness), 1.0)
        x = max(-50.0, min(50.0, x))
        return float(base_weight) / (1.0 + math.exp(-x))

    def compute_wall_mse_loss(self, pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
        wall_layers = min(int(self.config.wall_mse_layers), pred_norm.shape[-2])
        wall_pred = pred_norm[:, :, :wall_layers, :]
        wall_target = target_norm[:, :, :wall_layers, :]
        sigma = max(wall_layers / 3.0, 1.0)
        j_idx = torch.arange(wall_layers, device=pred_norm.device, dtype=pred_norm.dtype)
        weights = torch.exp(-0.5 * (j_idx / sigma).pow(2)).view(1, 1, wall_layers, 1)
        return (weights * (wall_pred - wall_target).pow(2)).mean()

    def compute_wall_cp_loss(
        self,
        *,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        coords_vertex = self.coords_vertex_from(batch)
        if coords_vertex is None:
            raise KeyError("wall Cp loss requires batch['coords_vertex']")
        if self.normalizer is not None:
            pred_phys = self.normalizer.inverse_transform(pred_norm)
            target_phys = self.normalizer.inverse_transform(target_norm)
        else:
            pred_phys = pred_norm
            target_phys = target_norm
        pred_cp, wall_weights = extract_wall_cp_ogrid_torch(
            pred_phys,
            coords_vertex.to(device=pred_norm.device),
            batch["flow_conditions"].to(device=pred_norm.device, dtype=pred_norm.dtype),
            gamma=1.4,
        )
        target_cp, _ = extract_wall_cp_ogrid_torch(
            target_phys,
            coords_vertex.to(device=pred_norm.device),
            batch["flow_conditions"].to(device=pred_norm.device, dtype=pred_norm.dtype),
            gamma=1.4,
        )
        reduction = str(self.config.wall_cp_reduction).lower()
        if reduction == "l2":
            point_loss = (pred_cp - target_cp).pow(2)
        elif reduction in {"smooth_l1", "huber"}:
            point_loss = F.smooth_l1_loss(pred_cp, target_cp, reduction="none")
        else:
            point_loss = (pred_cp - target_cp).abs()
        return (point_loss * wall_weights).sum() / wall_weights.sum().clamp_min(1.0e-8)

    @staticmethod
    def sanitize_physical_fields_for_pde(fields_phys: torch.Tensor) -> torch.Tensor:
        out = torch.nan_to_num(fields_phys, nan=0.0, posinf=0.0, neginf=0.0)
        if out.ndim == 4 and out.shape[1] >= 4:
            rho = out[:, 0:1].clamp_min(1.0e-4)
            uv = out[:, 1:3]
            pressure = out[:, 3:4].clamp_min(1.0e-4)
            tail = out[:, 4:]
            out = torch.cat([rho, uv, pressure, tail], dim=1)
        if out.ndim == 4 and out.shape[1] >= 5:
            nu_tilde = out[:, 4:5].clamp_min(0.0)
            out = torch.cat([out[:, :4], nu_tilde, out[:, 5:]], dim=1)
        return out

    def maybe_smooth_signed_residual_field(
        self,
        signed_field: torch.Tensor,
        wall_layers: Optional[int],
    ) -> torch.Tensor:
        if self.residual_smoother is None:
            return signed_field
        return self.residual_smoother.apply(signed_field, wall_layers=wall_layers)

    @staticmethod
    def _merge_cropped_prediction(
        pred_norm: torch.Tensor,
        *,
        full_template_norm: Optional[torch.Tensor],
        replacement_layers: Optional[int],
    ) -> torch.Tensor:
        if full_template_norm is None:
            return pred_norm
        layers = int(replacement_layers or pred_norm.shape[-2])
        layers = min(layers, pred_norm.shape[-2], full_template_norm.shape[-2])
        if layers == full_template_norm.shape[-2] and layers == pred_norm.shape[-2]:
            return pred_norm
        head = pred_norm[:, :, :layers, :]
        tail = full_template_norm.detach()[:, :, layers:, :]
        return torch.cat([head, tail], dim=-2)

    def compute_res_loss(
        self,
        *,
        pred_norm: torch.Tensor,
        batch: Dict[str, Any],
        baseline_norm: Optional[torch.Tensor] = None,
        full_template_norm: Optional[torch.Tensor] = None,
        replacement_layers: Optional[int] = None,
        default_wall_layers: Optional[int] = None,
    ) -> torch.Tensor:
        if self.residual_calculator is None:
            return torch.tensor(0.0, device=pred_norm.device, dtype=pred_norm.dtype)

        coords_center = batch.get("coords_center_pde", batch.get("coords_center", batch.get("coords")))
        coords_vertex = batch.get("coords_vertex")
        if not isinstance(coords_center, torch.Tensor) or not isinstance(coords_vertex, torch.Tensor):
            raise KeyError("residual-field loss requires coords_center_pde/coords_center and coords_vertex")

        eval_norm = self._merge_cropped_prediction(
            pred_norm,
            full_template_norm=full_template_norm,
            replacement_layers=replacement_layers,
        )
        if self.normalizer is not None:
            eval_phys = self.normalizer.inverse_transform(eval_norm)
        else:
            eval_phys = eval_norm
        eval_phys = self.sanitize_physical_fields_for_pde(eval_phys)

        loss_layers = self.config.res_loss_wall_layers
        if loss_layers is None:
            loss_layers = int(default_wall_layers or eval_phys.shape[-2])

        grid = {
            "center": coords_center.to(device=pred_norm.device),
            "vertex": coords_vertex.to(device=pred_norm.device),
        }
        wall_distance = batch.get("wall_distance")
        if isinstance(wall_distance, torch.Tensor):
            wall_distance = wall_distance.to(device=pred_norm.device)
            if wall_distance.shape[-2] != eval_phys.shape[-2]:
                wall_distance = wall_distance[..., : eval_phys.shape[-2], :]

        scale0 = None
        if self.config.res_field_scale_mode in {"baseline_eq_rms", "baseline"}:
            if baseline_norm is None:
                raise ValueError("res_field_scale_mode='baseline_eq_rms' requires baseline_norm")
            if self.normalizer is not None:
                baseline_phys = self.normalizer.inverse_transform(baseline_norm.detach())
            else:
                baseline_phys = baseline_norm.detach()
            _, baseline_result = self.residual_calculator.compute_residual(
                fields=self.sanitize_physical_fields_for_pde(baseline_phys),
                coords=grid,
                flow_conditions=batch["flow_conditions"].to(device=pred_norm.device),
                weights=self.config.res_loss_weights,
                return_spatial=False,
                return_components=False,
                return_signed_field=True,
                wall_layers=loss_layers,
                wall_distance=wall_distance,
                dtype=self.config.res_loss_dtype,
            )
            baseline_signed = baseline_result.get("signed_residual_field")
            if baseline_signed is not None:
                baseline_signed = self.maybe_smooth_signed_residual_field(baseline_signed, loss_layers)
                scale0 = compute_eq_rms_scales(
                    baseline_signed,
                    wall_layers=loss_layers,
                    eps=self.config.res_field_eps,
                )

        _, result = self.residual_calculator.compute_residual(
            fields=eval_phys,
            coords=grid,
            flow_conditions=batch["flow_conditions"].to(device=pred_norm.device),
            weights=self.config.res_loss_weights,
            return_spatial=False,
            return_components=False,
            return_signed_field=True,
            wall_layers=loss_layers,
            wall_distance=wall_distance,
            dtype=self.config.res_loss_dtype,
        )
        signed_field = result.get("signed_residual_field")
        if signed_field is None:
            return torch.tensor(0.0, device=pred_norm.device, dtype=pred_norm.dtype)
        signed_field = self.maybe_smooth_signed_residual_field(signed_field, loss_layers)
        residual_objective = compute_residual_field_objective(
            signed_field,
            weights=self.config.res_loss_weights,
            wall_layers=loss_layers,
            scale0=scale0,
            norm=self.config.res_field_norm,
            huber_delta=self.config.res_field_huber_delta,
            eps=self.config.res_field_eps,
        ).clamp_min(1.0e-12)

        reduction = self.config.res_loss_reduction
        if reduction == "log1p_l1":
            return torch.log1p(residual_objective).pow(2).mean()
        if reduction == "l2":
            return residual_objective.pow(2).mean()
        return residual_objective.mean()

    def compute_final_pde(
        self,
        *,
        pred_norm: torch.Tensor,
        batch: Dict[str, Any],
        full_template_norm: Optional[torch.Tensor] = None,
        replacement_layers: Optional[int] = None,
        default_wall_layers: Optional[int] = None,
    ) -> torch.Tensor:
        if self.residual_calculator is None:
            return torch.tensor(0.0, device=pred_norm.device, dtype=pred_norm.dtype)

        coords_center = batch.get("coords_center_pde", batch.get("coords_center", batch.get("coords")))
        coords_vertex = batch.get("coords_vertex")
        if not isinstance(coords_center, torch.Tensor) or not isinstance(coords_vertex, torch.Tensor):
            raise KeyError("final_pde diagnostic requires coords_center_pde/coords_center and coords_vertex")

        eval_norm = self._merge_cropped_prediction(
            pred_norm,
            full_template_norm=full_template_norm,
            replacement_layers=replacement_layers,
        )
        if self.normalizer is not None:
            eval_phys = self.normalizer.inverse_transform(eval_norm)
        else:
            eval_phys = eval_norm
        eval_phys = self.sanitize_physical_fields_for_pde(eval_phys)

        loss_layers = self.config.res_loss_wall_layers
        if loss_layers is None:
            loss_layers = int(default_wall_layers or eval_phys.shape[-2])

        grid = {
            "center": coords_center.to(device=pred_norm.device),
            "vertex": coords_vertex.to(device=pred_norm.device),
        }
        wall_distance = batch.get("wall_distance")
        if isinstance(wall_distance, torch.Tensor):
            wall_distance = wall_distance.to(device=pred_norm.device)
            if wall_distance.shape[-2] != eval_phys.shape[-2]:
                wall_distance = wall_distance[..., : eval_phys.shape[-2], :]

        with torch.no_grad():
            confidence, _ = self.residual_calculator.compute_residual(
                fields=eval_phys,
                coords=grid,
                flow_conditions=batch["flow_conditions"].to(device=pred_norm.device),
                weights=self.config.res_loss_weights,
                return_spatial=False,
                return_components=False,
                return_signed_field=False,
                wall_layers=loss_layers,
                wall_distance=wall_distance,
                dtype=self.config.res_loss_dtype,
            )
        if isinstance(confidence, torch.Tensor):
            return (-confidence).clamp_min(1.0e-8).mean().to(device=pred_norm.device, dtype=pred_norm.dtype)
        return torch.tensor(
            max(-float(confidence), 1.0e-8),
            device=pred_norm.device,
            dtype=pred_norm.dtype,
        )

__all__ = [
    "FlowPerceptualLoss",
    "MultiScaleFeaturePyramid",
    "OGridPad",
    "PhysicsDerivedFeatures",
    "SharedTrainingLossConfig",
    "SharedTrainingLosses",
    "compute_gradient_alignment_loss",
    "compute_volume_weighted_mse",
]
