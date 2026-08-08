"""Canonical inference backends for direct and FSB models."""

from __future__ import annotations

from typing import Any, Optional

import torch

from surrogate.fsb.bridge import create_i2sb_bridge
from surrogate.fsb.runtime import FSBEngine
from surrogate.inference.contracts import DirectPredictorConfig, FSBPredictorConfig, PredictionBatch
from surrogate.inference.loading import create_loaded_model, create_normalizer_from_config, load_experiment_config


class DirectPredictorBackend:
    """Inference backend for direct single-step models."""

    def __init__(
        self,
        config: DirectPredictorConfig,
        *,
        normalizer: Any = None,
    ) -> None:
        if config.config_path is None:
            raise ValueError("DirectPredictorBackend requires config.config_path")
        self.predictor_config = config
        self.config = load_experiment_config(config.config_path)
        if self.config.model.family != "direct":
            raise ValueError("DirectPredictorBackend requires model.family='direct'")
        self.device = torch.device(config.device)
        self.normalizer = normalizer if normalizer is not None else create_normalizer_from_config(self.config)
        self.model, self.checkpoint = create_loaded_model(
            self.config,
            checkpoint_path=config.checkpoint_path,
            device=self.device,
            use_ema=config.use_ema,
        )

    @torch.no_grad()
    def predict(
        self,
        batch: Optional[PredictionBatch] = None,
        *,
        geometry: Optional[torch.Tensor] = None,
        flow_conditions: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        initial_field: Optional[torch.Tensor] = None,
        inverse_transform: bool = False,
    ) -> torch.Tensor:
        if batch is not None:
            geometry = batch.geometry
            flow_conditions = batch.flow_conditions
            coords = batch.coords
            initial_field = batch.initial_field
        if geometry is None or flow_conditions is None or coords is None:
            raise ValueError("Direct prediction requires geometry, flow_conditions, and coords")

        pred = self.model(
            geometry.to(self.device),
            flow_conditions.to(self.device),
            coords.to(self.device),
            initial_field=initial_field.to(self.device) if isinstance(initial_field, torch.Tensor) else None,
        )
        if inverse_transform and self.normalizer is not None:
            return self.normalizer.inverse_transform(pred)
        return pred


class FSBPredictorBackend:
    """Inference backend for FSB multi-step models."""

    def __init__(
        self,
        config: FSBPredictorConfig,
        *,
        normalizer: Any = None,
    ) -> None:
        if config.config_path is None:
            raise ValueError("FSBPredictorBackend requires config.config_path")
        self.predictor_config = config
        self.config = load_experiment_config(config.config_path)
        if self.config.model.family != "fsb":
            raise ValueError("FSBPredictorBackend requires model.family='fsb'")
        self.device = torch.device(config.device)
        self.normalizer = normalizer if normalizer is not None else create_normalizer_from_config(self.config)
        self.model, self.checkpoint = create_loaded_model(
            self.config,
            checkpoint_path=config.checkpoint_path,
            device=self.device,
            use_ema=config.use_ema,
        )

        explicit_step_count = config.n_inference_steps is not None
        runtime_config = {
            "n_inference_steps": (
                int(config.n_inference_steps)
                if explicit_step_count
                else self.config.fsb.inference.n_steps or len(self.config.fsb.inference.custom_timesteps)
            ),
            "custom_timesteps": (
                config.custom_timesteps
                if config.custom_timesteps is not None
                else None if explicit_step_count else self.config.fsb.inference.custom_timesteps
            ),
            "eta": config.eta,
            "noise_mode": config.noise_mode,
        }
        self.engine = FSBEngine(
            model=self.model,
            config=runtime_config,
            device=self.device,
            bridge=create_i2sb_bridge(self.config, self.device),
            normalizer=self.normalizer,
        )

    @torch.no_grad()
    def predict(
        self,
        batch: Optional[PredictionBatch] = None,
        *,
        initial_field: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
        flow_conditions: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        timesteps: Optional[list[int] | torch.Tensor] = None,
        inverse_transform: bool = False,
    ) -> torch.Tensor:
        if batch is not None:
            initial_field = batch.initial_field
            geometry = batch.geometry
            flow_conditions = batch.flow_conditions
            coords = batch.coords
        if initial_field is None or geometry is None or flow_conditions is None or coords is None:
            raise ValueError("FSB prediction requires initial_field, geometry, flow_conditions, and coords")

        pred = self.engine.predict(
            initial_field=initial_field.to(self.device),
            geometry=geometry.to(self.device),
            flow_conditions=flow_conditions.to(self.device),
            coords=coords.to(self.device),
            timesteps=timesteps,
        )
        if inverse_transform and self.normalizer is not None:
            return self.normalizer.inverse_transform(pred)
        return pred


__all__ = [
    "DirectPredictorBackend",
    "FSBPredictorBackend",
]
