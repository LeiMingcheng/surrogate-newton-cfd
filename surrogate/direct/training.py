"""Direct-model supervised training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from surrogate.physics.losses import compute_volume_weighted_mse
from surrogate.training.losses import (
    SharedTrainingLossConfig,
    SharedTrainingLosses,
    compute_gradient_alignment_loss,
)


@dataclass
class DirectTrainerConfig(SharedTrainingLossConfig):
    """Supervised direct-trainer controls."""

    gradient_clip_norm: Optional[float] = None


class DirectTrainer:
    """Trainer for single-step direct surrogate models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        config: Optional[DirectTrainerConfig | Dict[str, Any]] = None,
        normalizer: Optional[torch.nn.Module] = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        if isinstance(config, DirectTrainerConfig):
            self.config = config
        elif hasattr(config, "to_direct_trainer_config"):
            self.config = DirectTrainerConfig(**config.to_direct_trainer_config())
        else:
            self.config = DirectTrainerConfig(**dict(config or {}))
        self.normalizer = normalizer.to(self.device) if hasattr(normalizer, "to") else normalizer
        self.global_step = 0
        self.losses = SharedTrainingLosses(
            self.config,
            normalizer=self.normalizer,
            device=self.device,
            global_step_fn=lambda: self.global_step,
        )
        self.model.to(self.device)

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    @staticmethod
    def _coords_from(batch: Dict[str, Any]) -> torch.Tensor:
        if "coords" in batch:
            return batch["coords"]
        if "coords_center" in batch:
            return batch["coords_center"]
        raise KeyError("DirectTrainer batch requires 'coords' or 'coords_center'")

    @staticmethod
    def _target_from(batch: Dict[str, Any]) -> torch.Tensor:
        if "target" in batch:
            return batch["target"]
        if "fields" in batch:
            return batch["fields"]
        raise KeyError("DirectTrainer batch requires 'target' or 'fields'")

    def _crop_tensor_wall_layers(self, tensor: torch.Tensor, layers: Optional[int] = None) -> torch.Tensor:
        return self.losses.crop_wall_layers(tensor, layers)

    def _cell_volumes_from(self, batch: Dict[str, Any], pred: torch.Tensor) -> Optional[torch.Tensor]:
        return self.losses.cell_volumes_from(batch, pred)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: Dict[str, Any],
        *,
        epoch: int = 0,
        use_volume_weighted_base: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        pred_for_loss = self._crop_tensor_wall_layers(pred)
        target_for_loss = self._crop_tensor_wall_layers(target)

        mse_loss = F.mse_loss(pred_for_loss, target_for_loss)
        base_loss = mse_loss
        metrics_tensors: Dict[str, torch.Tensor] = {"mse": mse_loss}

        if self.config.use_volume_weighted_mse:
            volumes = self._cell_volumes_from(batch, pred_for_loss)
            if volumes is not None:
                vw_loss, vw_metrics = compute_volume_weighted_mse(
                    pred_for_loss,
                    target_for_loss,
                    volumes,
                    alpha=self.config.volume_weight_alpha,
                    w_max=self.config.volume_weight_max,
                    wall_layers=None,
                )
                if use_volume_weighted_base:
                    base_loss = vw_loss
                metrics_tensors["volume_weighted_mse"] = vw_loss
                metrics_tensors.update({
                    key: torch.tensor(value, device=pred.device, dtype=pred.dtype)
                    for key, value in vw_metrics.items()
                })

        total_loss = base_loss
        metrics_tensors["base_loss"] = base_loss

        if self.config.wall_mse_weight > 0.0 and self.config.wall_mse_layers > 0:
            wall_mse_loss = self.losses.compute_wall_mse_loss(pred, target)
            total_loss = total_loss + float(self.config.wall_mse_weight) * wall_mse_loss
            metrics_tensors["wall_mse_loss"] = wall_mse_loss

        if self.config.wall_cp_weight > 0.0:
            wall_cp_loss = self.losses.compute_wall_cp_loss(
                pred_norm=pred,
                target_norm=target,
                batch=batch,
            )
            total_loss = total_loss + float(self.config.wall_cp_weight) * wall_cp_loss
            metrics_tensors["wall_cp_loss"] = wall_cp_loss

        if self.config.gradient_loss_weight > 0.0:
            gradient_loss = compute_gradient_alignment_loss(
                pred,
                target,
                wall_layers=self.config.gradient_loss_wall_layers,
            )
            total_loss = total_loss + float(self.config.gradient_loss_weight) * gradient_loss
            metrics_tensors["gradient_loss"] = gradient_loss

        flow_perceptual_weight = self.losses.warmup_weight(
            base_weight=self.config.flow_perceptual_weight,
            enabled=self.losses.flow_perceptual_loss is not None,
            epoch=int(epoch),
            warmup_epochs=self.config.flow_perceptual_warmup_epochs,
            warmup_center=self.config.flow_perceptual_warmup_center,
            warmup_sharpness=self.config.flow_perceptual_warmup_sharpness,
        )
        if self.losses.flow_perceptual_loss is not None:
            flow_perceptual_loss, _ = self.losses.flow_perceptual_loss(
                pred_norm=pred,
                target_norm=target,
                flow_conditions=batch["flow_conditions"],
                coords_center=self._coords_from(batch),
                original_norm=batch.get("initial_field"),
            )
            if flow_perceptual_weight > 0.0:
                total_loss = total_loss + float(flow_perceptual_weight) * flow_perceptual_loss
            metrics_tensors["flow_perceptual_loss"] = flow_perceptual_loss
            metrics_tensors["flow_perceptual_weight"] = torch.tensor(
                flow_perceptual_weight,
                device=pred.device,
                dtype=pred.dtype,
            )

        res_loss_weight = self.losses.warmup_weight(
            base_weight=self.config.res_loss_weight,
            enabled=self.config.res_loss_enabled and self.losses.residual_calculator is not None,
            epoch=int(epoch),
            warmup_epochs=self.config.res_loss_warmup_epochs,
            warmup_center=self.config.res_loss_warmup_center,
            warmup_sharpness=self.config.res_loss_warmup_sharpness,
        )
        if self.config.res_loss_enabled:
            res_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            if res_loss_weight > 0.0:
                res_loss = self.losses.compute_res_loss(
                    pred_norm=pred,
                    batch=batch,
                    baseline_norm=target.detach(),
                    default_wall_layers=pred.shape[-2],
                )
                total_loss = total_loss + float(res_loss_weight) * res_loss
            metrics_tensors["res_loss"] = res_loss
            metrics_tensors["res_loss_weight"] = torch.tensor(
                res_loss_weight,
                device=pred.device,
                dtype=pred.dtype,
            )

        metrics_tensors["total_loss"] = total_loss

        metrics = {key: float(value.detach().item()) for key, value in metrics_tensors.items()}
        return total_loss, metrics

    def compute_validation_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: Dict[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Validation metric path for first-eval checks."""

        pred_for_loss = self._crop_tensor_wall_layers(pred)
        target_for_loss = self._crop_tensor_wall_layers(target)
        mse_loss = F.mse_loss(pred_for_loss, target_for_loss)
        metrics_tensors: Dict[str, torch.Tensor] = {
            "mse": mse_loss,
            "base_loss": mse_loss,
            "total_loss": mse_loss,
        }
        if self.config.use_volume_weighted_mse:
            volumes = self._cell_volumes_from(batch, pred_for_loss)
            if volumes is not None:
                vw_loss, vw_metrics = compute_volume_weighted_mse(
                    pred_for_loss,
                    target_for_loss,
                    volumes,
                    alpha=self.config.volume_weight_alpha,
                    w_max=self.config.volume_weight_max,
                    wall_layers=None,
                )
                metrics_tensors["volume_weighted_mse"] = vw_loss
                metrics_tensors.update({
                    key: torch.tensor(value, device=pred.device, dtype=pred.dtype)
                    for key, value in vw_metrics.items()
                })
        if self.config.wall_cp_weight > 0.0:
            metrics_tensors["wall_cp_loss"] = self.losses.compute_wall_cp_loss(
                pred_norm=pred,
                target_norm=target,
                batch=batch,
            )
        if self.config.gradient_loss_weight > 0.0:
            metrics_tensors["gradient_loss"] = compute_gradient_alignment_loss(
                pred_for_loss,
                target_for_loss,
                wall_layers=self.config.gradient_loss_wall_layers,
            )
        return mse_loss, {key: float(value.detach().item()) for key, value in metrics_tensors.items()}

    def forward_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        batch = self._move_batch(batch)
        initial_field = batch.get("initial_field")
        return self.model(
            batch["geometry"],
            batch["flow_conditions"],
            self._coords_from(batch),
            initial_field=initial_field,
        )

    def train_step(self, batch: Dict[str, Any], *, epoch: int = 0) -> Dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("DirectTrainer.train_step requires an optimizer")

        self.model.train()
        batch = self._move_batch(batch)
        target = self._target_from(batch)
        pred = self.model(
            batch["geometry"],
            batch["flow_conditions"],
            self._coords_from(batch),
            initial_field=batch.get("initial_field"),
        )
        loss, metrics = self.compute_loss(
            pred,
            target,
            batch,
            epoch=epoch,
            use_volume_weighted_base=True,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.gradient_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.config.gradient_clip_norm),
            )
            metrics["grad_norm"] = float(grad_norm)
        self.optimizer.step()
        self.global_step += 1
        return {"loss": loss.detach(), "metrics": metrics, "prediction": pred.detach()}

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, Any], *, epoch: int = 0) -> Dict[str, Any]:
        self.model.eval()
        batch = self._move_batch(batch)
        target = self._target_from(batch)
        pred = self.model(
            batch["geometry"],
            batch["flow_conditions"],
            self._coords_from(batch),
            initial_field=batch.get("initial_field"),
        )
        del epoch
        loss, metrics = self.compute_validation_loss(pred, target, batch)
        return {"loss": loss.detach(), "metrics": metrics, "prediction": pred}

__all__ = ["DirectTrainer", "DirectTrainerConfig"]
