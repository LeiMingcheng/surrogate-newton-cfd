"""FSB training entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from surrogate.fsb.bridge import I2SBBridge, create_i2sb_bridge
from surrogate.evaluation import compute_force_metrics
from surrogate.training.losses import (
    SharedTrainingLossConfig,
    SharedTrainingLosses,
    compute_gradient_alignment_loss,
)


InitialFieldFn = Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor]


@dataclass
class FSBTrainerConfig(SharedTrainingLossConfig):
    """Trainer controls for flow-state-bridge models."""

    lambda_reconstruction: float = 0.0
    use_l1_reconstruction: bool = False
    gradient_clip_norm: Optional[float] = None
    loss_prediction_type: Optional[str] = None


class FSBTrainer:
    """Trainer for flow-state-bridge models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        bridge: Optional[I2SBBridge] = None,
        experiment_config: Any = None,
        config: Optional[FSBTrainerConfig | Dict[str, Any]] = None,
        initial_field_fn: Optional[InitialFieldFn] = None,
        normalizer: Optional[torch.nn.Module] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device(device)
        if isinstance(config, FSBTrainerConfig):
            self.config = config
        elif hasattr(config, "to_fsb_trainer_config"):
            self.config = FSBTrainerConfig(**config.to_fsb_trainer_config())
        else:
            self.config = FSBTrainerConfig(**dict(config or {}))
        self.bridge = bridge or create_i2sb_bridge(experiment_config or {}, self.device)
        self.initial_field_fn = initial_field_fn
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
        raise KeyError("FSBTrainer batch requires 'coords' or 'coords_center'")

    @staticmethod
    def _target_from(batch: Dict[str, Any]) -> torch.Tensor:
        if "target" in batch:
            return batch["target"]
        if "fields" in batch:
            return batch["fields"]
        raise KeyError("FSBTrainer batch requires 'target' or 'fields'")

    def _initial_field_from(self, batch: Dict[str, Any], target: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if "x1" in batch:
            return batch["x1"]
        if "initial_field" in batch:
            return batch["initial_field"]
        if self.initial_field_fn is not None:
            return self.initial_field_fn(batch["flow_conditions"], coords, batch)
        raise KeyError(
            "FSBTrainer requires 'x1'/'initial_field' in the batch or an initial_field_fn"
        )

    def _crop_tensor_wall_layers(self, tensor: torch.Tensor, layers: Optional[int] = None) -> torch.Tensor:
        return self.losses.crop_wall_layers(tensor, layers)

    def _cell_volumes_from(self, batch: Dict[str, Any], pred: torch.Tensor) -> Optional[torch.Tensor]:
        return self.losses.cell_volumes_from(batch, pred)

    def prepare_training_state(
        self,
        batch: Dict[str, Any],
        *,
        epoch: int = 0,
    ) -> Dict[str, Any]:
        batch = self._move_batch(batch)
        target_full = self._target_from(batch)
        coords_full = self._coords_from(batch)
        target = self._crop_tensor_wall_layers(target_full)
        coords = self._crop_tensor_wall_layers(coords_full)
        x1_full = self._initial_field_from(batch, target_full, coords_full)
        x1 = self._crop_tensor_wall_layers(x1_full)
        forward_data = self.bridge.prepare_forward(
            target_field=target,
            input_field=x1,
            epoch=epoch,
            batch_size=int(target.shape[0]),
        )
        return {
            "batch": batch,
            "target": target,
            "target_full": target_full,
            "coords": coords,
            "coords_full": coords_full,
            "x1": x1,
            "x1_full": x1_full,
            **forward_data,
        }

    def forward_training_state(self, state: Dict[str, Any]) -> torch.Tensor:
        batch = state["batch"]
        return self.model(
            noisy_fields=state["x_t"],
            timesteps=state["timesteps"],
            geometry=batch["geometry"],
            flow_conditions=batch["flow_conditions"],
            coords=state["coords"],
        )

    def compute_loss(
        self,
        pred: torch.Tensor,
        state: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        reconstructed_x0 = self.bridge.reconstruct_x0(
            state["x_t"],
            pred,
            state["timesteps"],
        )
        if self.config.loss_prediction_type == "v_prediction" and self.bridge.scheduler.prediction_type == "x0":
            pred_for_loss = self.bridge.scheduler.compute_v(state["x_t"], pred, state["timesteps"])
            target_for_loss = self.bridge.scheduler.compute_v(state["x_t"], state["target"], state["timesteps"])
        else:
            pred_for_loss = pred
            target_for_loss = state["target"]

        direction_loss = self.losses.compute_main_metric_loss(
            pred_for_loss,
            target_for_loss,
            batch=batch,
            loss_weight=batch.get("loss_weight"),
            use_l1=self.config.use_l1_reconstruction,
        )
        target_x0 = state["target"]
        reconstruction_loss = self.losses.compute_main_metric_loss(
            reconstructed_x0,
            target_x0,
            batch=batch,
            loss_weight=batch.get("loss_weight"),
            use_l1=self.config.use_l1_reconstruction,
        )
        supervised_loss = direction_loss + self.config.lambda_reconstruction * reconstruction_loss
        total_loss = supervised_loss
        losses = {
            "sup_loss": supervised_loss,
            "direction_loss": direction_loss,
            "reconstruction_loss": reconstruction_loss,
        }

        if self.config.wall_mse_weight > 0.0 and self.config.wall_mse_layers > 0:
            wall_mse_loss = self.losses.compute_wall_mse_loss(reconstructed_x0, target_x0)
            total_loss = total_loss + float(self.config.wall_mse_weight) * wall_mse_loss
            losses["wall_mse_loss"] = wall_mse_loss

        if self.config.wall_cp_weight > 0.0:
            wall_cp_loss = self.losses.compute_wall_cp_loss(
                pred_norm=reconstructed_x0,
                target_norm=target_x0,
                batch=batch,
            )
            total_loss = total_loss + float(self.config.wall_cp_weight) * wall_cp_loss
            losses["wall_cp_loss"] = wall_cp_loss

        if self.config.gradient_loss_weight > 0.0:
            gradient_loss = compute_gradient_alignment_loss(
                reconstructed_x0,
                target_x0,
                wall_layers=self.config.gradient_loss_wall_layers,
            )
            total_loss = total_loss + float(self.config.gradient_loss_weight) * gradient_loss
            losses["gradient_loss"] = gradient_loss

        flow_perceptual_weight = self.losses.warmup_weight(
            base_weight=self.config.flow_perceptual_weight,
            enabled=self.losses.flow_perceptual_loss is not None,
            epoch=int(state.get("epoch", 0)),
            warmup_epochs=self.config.flow_perceptual_warmup_epochs,
            warmup_center=self.config.flow_perceptual_warmup_center,
            warmup_sharpness=self.config.flow_perceptual_warmup_sharpness,
        )
        if self.losses.flow_perceptual_loss is not None:
            flow_perceptual_loss, _ = self.losses.flow_perceptual_loss(
                pred_norm=reconstructed_x0,
                target_norm=target_x0,
                flow_conditions=batch["flow_conditions"],
                coords_center=state["coords"],
                original_norm=state["x1"],
            )
            if flow_perceptual_weight > 0.0:
                total_loss = total_loss + float(flow_perceptual_weight) * flow_perceptual_loss
            losses["flow_perceptual_loss"] = flow_perceptual_loss
            losses["flow_perceptual_weight"] = torch.tensor(
                flow_perceptual_weight,
                device=reconstructed_x0.device,
                dtype=reconstructed_x0.dtype,
            )

        res_loss_weight = self.losses.warmup_weight(
            base_weight=self.config.res_loss_weight,
            enabled=self.config.res_loss_enabled and self.losses.residual_calculator is not None,
            epoch=int(state.get("epoch", 0)),
            warmup_epochs=self.config.res_loss_warmup_epochs,
            warmup_center=self.config.res_loss_warmup_center,
            warmup_sharpness=self.config.res_loss_warmup_sharpness,
        )
        if self.config.res_loss_enabled:
            res_loss = torch.tensor(0.0, device=reconstructed_x0.device, dtype=reconstructed_x0.dtype)
            if res_loss_weight > 0.0:
                res_loss = self.losses.compute_res_loss(
                    pred_norm=reconstructed_x0,
                    batch=batch,
                    baseline_norm=state["x1_full"].detach(),
                    full_template_norm=state["target_full"],
                    replacement_layers=int(self.config.wall_layers or reconstructed_x0.shape[-2]),
                    default_wall_layers=int(self.config.wall_layers or reconstructed_x0.shape[-2]),
                )
                total_loss = total_loss + float(res_loss_weight) * res_loss
            losses["res_loss"] = res_loss
            losses["res_loss_weight"] = torch.tensor(
                res_loss_weight,
                device=reconstructed_x0.device,
                dtype=reconstructed_x0.dtype,
            )

        normalized_supervised_loss = self.losses.compute_main_metric_loss(
            reconstructed_x0,
            target_x0,
            batch=batch,
            loss_weight=batch.get("loss_weight"),
            use_l1=self.config.use_l1_reconstruction,
        )
        physical_supervised_loss = self.losses.compute_physical_main_metric_loss(
            reconstructed_x0,
            target_x0,
            batch=batch,
            loss_weight=batch.get("loss_weight"),
            use_l1=self.config.use_l1_reconstruction,
        )
        losses["normalized_loss"] = normalized_supervised_loss
        losses["physical_loss"] = physical_supervised_loss
        losses["total_loss"] = total_loss
        return losses

    def compute_validation_loss(
        self,
        pred: torch.Tensor,
        state: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        reconstructed_x0 = self.bridge.reconstruct_x0(
            state["x_t"],
            pred,
            state["timesteps"],
        )
        if self.config.loss_prediction_type == "v_prediction" and self.bridge.scheduler.prediction_type == "x0":
            pred_for_loss = self.bridge.scheduler.compute_v(state["x_t"], pred, state["timesteps"])
            target_for_loss = self.bridge.scheduler.compute_v(state["x_t"], state["target"], state["timesteps"])
        else:
            pred_for_loss = pred
            target_for_loss = state["target"]

        direction_loss = self.losses.compute_main_metric_loss(
            pred_for_loss,
            target_for_loss,
            batch=batch,
            loss_weight=batch.get("loss_weight"),
            use_l1=self.config.use_l1_reconstruction,
        )
        target_x0 = state["target"]
        reconstruction_loss = self.losses.compute_main_metric_loss(
            reconstructed_x0,
            target_x0,
            batch=batch,
            loss_weight=batch.get("loss_weight"),
            use_l1=self.config.use_l1_reconstruction,
        )
        supervised_loss = direction_loss + self.config.lambda_reconstruction * reconstruction_loss
        losses = {
            "sup_loss": supervised_loss,
            "direction_loss": direction_loss,
            "reconstruction_loss": reconstruction_loss,
            "total_loss": supervised_loss,
        }

        normalized_supervised_loss = self.losses.compute_main_metric_loss(
            reconstructed_x0,
            target_x0,
            batch=batch,
            loss_weight=batch.get("loss_weight"),
            use_l1=self.config.use_l1_reconstruction,
        )
        physical_supervised_loss = self.losses.compute_physical_main_metric_loss(
            reconstructed_x0,
            target_x0,
            batch=batch,
            loss_weight=batch.get("loss_weight"),
            use_l1=self.config.use_l1_reconstruction,
        )
        losses["normalized_loss"] = normalized_supervised_loss
        losses["physical_loss"] = physical_supervised_loss

        if self.config.wall_mse_weight > 0.0 and self.config.wall_mse_layers > 0:
            losses["wall_mse_loss"] = self.losses.compute_wall_mse_loss(reconstructed_x0, target_x0)
        if self.config.wall_cp_weight > 0.0:
            losses["wall_cp_loss"] = self.losses.compute_wall_cp_loss(
                pred_norm=reconstructed_x0,
                target_norm=target_x0,
                batch=batch,
            )
        if self.config.gradient_loss_weight > 0.0:
            losses["gradient_loss"] = compute_gradient_alignment_loss(
                reconstructed_x0,
                target_x0,
                wall_layers=self.config.gradient_loss_wall_layers,
            )
        if self.config.res_loss_enabled and self.losses.residual_calculator is not None:
            losses["final_pde"] = self.losses.compute_final_pde(
                pred_norm=reconstructed_x0,
                batch=batch,
                full_template_norm=state["target_full"],
                replacement_layers=int(self.config.wall_layers or reconstructed_x0.shape[-2]),
                default_wall_layers=int(self.config.wall_layers or reconstructed_x0.shape[-2]),
            )
        return losses

    def _compute_force_mae_metrics(
        self,
        *,
        reconstructed_x0: torch.Tensor,
        target_x0: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        coords_vertex = batch.get("coords_vertex")
        flow_conditions = batch.get("flow_conditions")
        if not isinstance(coords_vertex, torch.Tensor) or not isinstance(flow_conditions, torch.Tensor):
            return {}

        if self.normalizer is not None:
            pred_phys = self.normalizer.inverse_transform(reconstructed_x0)
            target_phys = self.normalizer.inverse_transform(target_x0)
        else:
            pred_phys = reconstructed_x0
            target_phys = target_x0

        force_metrics = compute_force_metrics(
            pred_phys,
            target_phys,
            coords_vertex,
            flow_conditions,
        )
        return {
            "CL_mae": float(force_metrics["force_mae_CL"]),
            "CD_mae": float(force_metrics["force_mae_CD"]),
            "Cm_mae": float(force_metrics["force_mae_Cm"]),
        }

    def train_step(self, batch: Dict[str, Any], *, epoch: int = 0) -> Dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("FSBTrainer.train_step requires an optimizer")

        self.model.train()
        state = self.prepare_training_state(batch, epoch=epoch)
        state["epoch"] = int(epoch)
        batch = state["batch"]
        pred = self.forward_training_state(state)
        losses = self.compute_loss(pred, state, batch)
        total_loss = losses["total_loss"]

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        metrics = {key: float(value.detach().item()) for key, value in losses.items()}
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.config.gradient_clip_norm),
            )
        self.optimizer.step()
        self.global_step += 1
        return {"loss": total_loss.detach(), "metrics": metrics, "prediction": pred.detach()}

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, Any], *, epoch: int = 0) -> Dict[str, Any]:
        self.model.eval()
        state = self.prepare_training_state(batch, epoch=epoch)
        state["epoch"] = int(epoch)
        pred = self.forward_training_state(state)
        losses = self.compute_validation_loss(pred, state, state["batch"])
        metrics = {key: float(value.detach().item()) for key, value in losses.items()}
        metrics.update(
            self._compute_force_mae_metrics(
                reconstructed_x0=self.bridge.reconstruct_x0(state["x_t"], pred, state["timesteps"]),
                target_x0=state["target"],
                batch=state["batch"],
            )
        )
        return {"loss": losses["total_loss"].detach(), "metrics": metrics, "prediction": pred}

__all__ = [
    "FSBTrainer",
    "FSBTrainerConfig",
]
