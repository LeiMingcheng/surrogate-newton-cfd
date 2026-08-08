"""Concrete flow-state-bridge training helper."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from surrogate.fsb.scheduler import I2SBScheduler


class I2SBBridge:
    """Schrodinger bridge interpolation between uniform-flow x1 and target x0."""

    def __init__(
        self,
        scheduler: I2SBScheduler,
        i2sb_config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.scheduler = scheduler
        self.device = device
        self.t0_sampling_power = i2sb_config.get("t0_sampling_power", 1.0)

    @property
    def num_timesteps(self) -> int:
        return self.scheduler.num_timesteps

    def sample_timesteps(self, batch_size: int) -> Tensor:
        u = torch.rand(batch_size, device=self.device)
        t_normalized = u ** self.t0_sampling_power
        return (t_normalized * (self.scheduler.num_timesteps - 1)).long()

    def prepare_forward(
        self,
        target_field: Tensor,
        input_field: Tensor,
        epoch: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        timesteps = self.sample_timesteps(batch_size)
        noise = torch.randn_like(target_field)
        x_t = self.scheduler.add_bridge_noise(target_field, input_field, timesteps, noise)
        target = self.scheduler.get_training_target(x_t, target_field, timesteps)
        return {
            "x_t": x_t,
            "timesteps": timesteps,
            "noise": noise,
            "target": target,
            "x1": input_field,
        }

    def reconstruct_x0(
        self,
        x_t: Tensor,
        direction_pred: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        return self.scheduler.reconstruct_x0(x_t, direction_pred, timesteps)

    def get_inference_timesteps(
        self,
        n_steps: int,
        device: torch.device,
    ) -> Tensor:
        step_size = max(1, (self.scheduler.num_timesteps - 1) // n_steps)
        timesteps = torch.arange(
            self.scheduler.num_timesteps - 1,
            -1,
            -step_size,
            device=device,
        )
        return timesteps[:n_steps]

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        x1: Optional[Tensor] = None,
    ) -> Tensor:
        if x1 is None:
            raise ValueError("I2SB step requires x1.")
        timestep_next = max(0, timestep - 1)
        return self.scheduler.step(model_output, timestep, sample, x1, timestep_next)

    def add_noise_for_inference(
        self,
        x1: Tensor,
        t_start: int,
    ) -> Tensor:
        t_tensor = torch.full((x1.shape[0],), t_start, device=self.device, dtype=torch.long)
        return self.scheduler.add_noise_from_x1(x1, t_tensor)


def _as_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _bridge_config_from(config: Any) -> Dict[str, Any]:
    if hasattr(config, "fsb"):
        return _bridge_config_from(getattr(config, "fsb"))
    if hasattr(config, "bridge"):
        return _as_mapping(getattr(config, "bridge"))
    config_dict = _as_mapping(config)
    if "fsb" in config_dict:
        return _bridge_config_from(config_dict.get("fsb"))
    return _as_mapping(config_dict.get("bridge", config_dict))


def create_i2sb_bridge(
    config: Any,
    device: torch.device,
) -> I2SBBridge:
    i2sb_config = _bridge_config_from(config)
    scheduler = I2SBScheduler(
        num_timesteps=i2sb_config.get("num_timesteps", 1000),
        beta_max=i2sb_config.get("beta_max", 0.3),
        beta_schedule=i2sb_config.get("beta_schedule", "symmetric_sine"),
        timestep_spacing=i2sb_config.get("timestep_spacing", "quadratic"),
        clip_sample=i2sb_config.get("clip_sample", False),
        prediction_type=i2sb_config.get("prediction_type", "epsilon"),
    ).to(device)
    return I2SBBridge(
        scheduler=scheduler,
        i2sb_config=i2sb_config,
        device=device,
    )


__all__ = ["I2SBBridge", "create_i2sb_bridge"]
