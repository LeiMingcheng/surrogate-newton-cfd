"""Flow-state-bridge inference runtime."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from surrogate.fsb.bridge import I2SBBridge, create_i2sb_bridge


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


class FSBEngine:
    """Runtime engine for FSB inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        device: str | torch.device = "cuda",
        *,
        bridge: Optional[I2SBBridge] = None,
        normalizer: Any = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.config = config
        self.runtime_config = _as_dict(config)
        bridge_source = (
            {"bridge": self.runtime_config["i2sb"]}
            if "i2sb" in self.runtime_config
            else config
        )
        self.bridge = bridge or create_i2sb_bridge(bridge_source, self.device)
        self.i2sb_scheduler = self.bridge.scheduler

        self.normalizer = normalizer

        self.n_inference_steps = int(self.runtime_config.get("n_inference_steps", 5))
        self.eta = float(self.runtime_config.get("eta", 0.0))
        self.noise_mode = str(self.runtime_config.get("noise_mode", "zeros")).lower()
        self.project_x0_to_physical_bounds = bool(
            self.runtime_config.get("project_x0_to_physical_bounds", False)
        )
        self.last_i2sb_profile: Dict[str, Any] = {}

    def _resolve_timesteps(
        self,
        timesteps: Optional[Iterable[int] | torch.Tensor] = None,
    ) -> torch.Tensor:
        if timesteps is None:
            custom = self.runtime_config.get("custom_timesteps")
            if custom is None:
                inference_cfg = self.runtime_config.get("inference")
                if isinstance(inference_cfg, dict):
                    custom = inference_cfg.get("custom_timesteps")
            if custom is not None:
                timesteps = custom

        if timesteps is None:
            result = self.i2sb_scheduler.get_timesteps(self.n_inference_steps, self.device)
        elif isinstance(timesteps, torch.Tensor):
            result = timesteps.to(device=self.device, dtype=torch.long)
        else:
            result = torch.tensor([int(v) for v in timesteps], device=self.device, dtype=torch.long)

        if result.numel() == 0:
            raise ValueError("FSB inference requires at least one timestep")
        if result.numel() == 1 or int(result[-1].item()) != 0:
            result = torch.cat([result, result.new_zeros((1,))], dim=0)
        return result

    def prepare_conditions(
        self,
        *,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        current_field: Optional[torch.Tensor] = None,
        target_field: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Prepare model conditioning tensors for FSB inference."""
        geometry = geometry.to(self.device)
        flow_conditions = flow_conditions.to(self.device)
        coords = coords.to(self.device)
        if current_field is not None:
            current_field = current_field.to(self.device)
        if target_field is not None:
            target_field = target_field.to(self.device)

        return {
            "geometry": geometry,
            "flow_conditions": flow_conditions,
            "coords": coords,
            "current_field": current_field,
            "target_field": target_field,
        }

    def _predict_model(
        self,
        *,
        noisy_fields: torch.Tensor,
        timesteps: torch.Tensor,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        return_aux: bool = False,
        **_: Any,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Run the FSB model on bridge state and conditioning tensors."""
        model_output = self.model(
            noisy_fields=noisy_fields,
            timesteps=timesteps,
            geometry=geometry,
            flow_conditions=flow_conditions,
            coords=coords,
        )
        if return_aux:
            return {"model_output": model_output}
        return model_output

    def _initial_sample(
        self,
        x1: torch.Tensor,
        t_start: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_batch = torch.full((x1.shape[0],), int(t_start), device=self.device, dtype=torch.long)
        if noise is None and self.noise_mode == "zeros":
            noise = torch.zeros_like(x1)
        return self.i2sb_scheduler.add_noise_from_x1(x1, t_batch, noise=noise)

    def _project_x0_to_physical_bounds(self, pred_x0: torch.Tensor) -> torch.Tensor:
        """Clamp reconstructed physical states before the next bridge step."""
        if self.normalizer is not None:
            physical = self.normalizer.inverse_transform(pred_x0)
        else:
            physical = pred_x0

        physical = torch.nan_to_num(physical, nan=0.0, posinf=0.0, neginf=0.0)
        if physical.shape[1] >= 1:
            physical[:, 0] = torch.clamp(physical[:, 0], min=1.0e-4)
        if physical.shape[1] >= 4:
            physical[:, 3] = torch.clamp(physical[:, 3], min=1.0e-4)
        if physical.shape[1] >= 5:
            physical[:, 4] = torch.clamp(physical[:, 4], min=0.0)

        if self.normalizer is not None:
            return self.normalizer.transform(physical).to(dtype=pred_x0.dtype)
        return physical.to(dtype=pred_x0.dtype)

    def _i2sb_denoise(
        self,
        *,
        surrogate_fields: torch.Tensor,
        conditions: Dict[str, Any],
        original_field: Optional[torch.Tensor] = None,
        timesteps: Optional[Iterable[int] | torch.Tensor] = None,
        noise0: Optional[torch.Tensor] = None,
        eta: Optional[float] = None,
        return_intermediates: bool = False,
        **_: Any,
    ) -> torch.Tensor | Dict[str, Any]:
        """Run FSB denoising from initializer state x1 to terminal x0."""
        del original_field
        start = perf_counter()
        x1 = surrogate_fields.to(self.device)
        geometry = conditions["geometry"].to(self.device)
        flow_conditions = conditions["flow_conditions"].to(self.device)
        coords = conditions["coords"].to(self.device)

        inference_timesteps = self._resolve_timesteps(timesteps)
        x_t = self._initial_sample(x1, int(inference_timesteps[0].item()), noise=noise0)
        step_eta = self.eta if eta is None else float(eta)
        intermediates = [x_t] if return_intermediates else None

        profile: Dict[str, Any] = {
            "setup_s": 0.0,
            "add_noise_s": 0.0,
            "model_forward_s": 0.0,
            "reconstruct_x0_s": 0.0,
            "project_x0_s": 0.0,
            "scheduler_step_s": 0.0,
            "step_count": 0,
            "model_forward_calls": 0,
        }
        profile["setup_s"] = float(perf_counter() - start)

        for step_idx in range(int(inference_timesteps.numel()) - 1):
            t_current = int(inference_timesteps[step_idx].item())
            t_next = int(inference_timesteps[step_idx + 1].item())
            t_batch = torch.full((x_t.shape[0],), t_current, device=self.device, dtype=torch.long)

            model_start = perf_counter()
            model_output = self._predict_model(
                noisy_fields=x_t,
                timesteps=t_batch,
                geometry=geometry,
                flow_conditions=flow_conditions,
                coords=coords,
            )
            profile["model_forward_s"] += float(perf_counter() - model_start)
            profile["model_forward_calls"] += 1

            reconstruct_start = perf_counter()
            pred_x0 = self.i2sb_scheduler.reconstruct_x0(x_t, model_output, t_batch)
            profile["reconstruct_x0_s"] += float(perf_counter() - reconstruct_start)

            if self.project_x0_to_physical_bounds:
                project_start = perf_counter()
                pred_x0 = self._project_x0_to_physical_bounds(pred_x0)
                profile["project_x0_s"] += float(perf_counter() - project_start)

            step_start = perf_counter()
            x_t = self.i2sb_scheduler.step_from_x0(
                timestep=t_current,
                x0=pred_x0,
                x1=x1,
                sample=x_t,
                timestep_next=t_next,
                eta=step_eta,
            )
            profile["scheduler_step_s"] += float(perf_counter() - step_start)
            profile["step_count"] += 1
            if intermediates is not None:
                intermediates.append(x_t)

        self.last_i2sb_profile = profile
        if return_intermediates:
            return {
                "sample": x_t,
                "intermediates": intermediates,
                "timesteps": inference_timesteps,
                "profile": profile,
            }
        return x_t

    def _physical_from_norm(self, value: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return value
        return self.normalizer.inverse_transform(value)

    def _norm_from_physical(self, value: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return value
        return self.normalizer.transform(value)

    def _state_tensor(
        self,
        state: Mapping[str, Any],
        key: str,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if key not in state:
            raise ValueError(f"FSB staged scheduler state is missing {key!r}")
        tensor = torch.as_tensor(state[key], device=self.device, dtype=dtype)
        if tensor.numel() == 0:
            raise ValueError(f"FSB staged scheduler state {key!r} is empty")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"FSB staged scheduler state {key!r} contains non-finite values")
        return tensor

    @torch.no_grad()
    def predict_staged_denoise_step(
        self,
        *,
        initial_field: torch.Tensor,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        target_step: int,
        timesteps: Optional[Iterable[int] | torch.Tensor] = None,
        noise0: Optional[torch.Tensor] = None,
        return_physical: bool = True,
    ) -> Dict[str, Any]:
        """Predict one FSB denoise x0 and capture state for later continuation."""

        conditions = self.prepare_conditions(
            geometry=geometry,
            flow_conditions=flow_conditions,
            coords=coords,
            current_field=initial_field,
        )
        x1 = initial_field.to(self.device)
        geometry = conditions["geometry"].to(self.device)
        flow_conditions = conditions["flow_conditions"].to(self.device)
        coords = conditions["coords"].to(self.device)
        inference_timesteps = self._resolve_timesteps(timesteps)
        if int(target_step) < 0 or int(target_step) >= int(inference_timesteps.numel()) - 1:
            raise ValueError(
                "target_step must select one denoise transition in the resolved timesteps"
            )

        x_t = self._initial_sample(
            x1,
            int(inference_timesteps[0].item()),
            noise=noise0.to(self.device) if isinstance(noise0, torch.Tensor) else noise0,
        )
        step_eta = self.eta
        for step_idx in range(int(inference_timesteps.numel()) - 1):
            t_current = int(inference_timesteps[step_idx].item())
            t_next = int(inference_timesteps[step_idx + 1].item())
            t_batch = torch.full((x_t.shape[0],), t_current, device=self.device, dtype=torch.long)
            model_output = self._predict_model(
                noisy_fields=x_t,
                timesteps=t_batch,
                geometry=geometry,
                flow_conditions=flow_conditions,
                coords=coords,
            )
            pred_x0 = self.i2sb_scheduler.reconstruct_x0(x_t, model_output, t_batch)
            if self.project_x0_to_physical_bounds:
                pred_x0 = self._project_x0_to_physical_bounds(pred_x0)

            if int(step_idx) == int(target_step):
                field = self._physical_from_norm(pred_x0) if return_physical else pred_x0
                return {
                    "field": field,
                    "scheduler_state": {
                        "resolved_timesteps": inference_timesteps.detach().cpu(),
                        "target_step": int(target_step),
                        "t_current": int(t_current),
                        "t_next": int(t_next),
                        "x_t_before_step": x_t.detach().cpu(),
                        "x1_norm": x1.detach().cpu(),
                        "eta": float(step_eta),
                        "noise_mode": self.noise_mode,
                    },
                }

            x_t = self.i2sb_scheduler.step_from_x0(
                timestep=t_current,
                x0=pred_x0,
                x1=x1,
                sample=x_t,
                timestep_next=t_next,
                eta=step_eta,
            )

        raise ValueError(f"target_step {target_step} is out of range")

    @torch.no_grad()
    def continue_from_projected_x0(
        self,
        *,
        projected_x0: torch.Tensor,
        scheduler_state: Mapping[str, Any],
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        projected_x0_is_physical: bool = True,
        return_physical: bool = True,
    ) -> torch.Tensor:
        """Continue FSB denoising from an externally corrected x0."""

        dtype = projected_x0.dtype
        x_t = self._state_tensor(
            scheduler_state,
            "x_t_before_step",
            dtype=dtype,
        ).to(self.device)
        x1 = self._state_tensor(scheduler_state, "x1_norm", dtype=dtype).to(self.device)
        timesteps = torch.as_tensor(
            scheduler_state.get("resolved_timesteps"),
            device=self.device,
            dtype=torch.long,
        )
        if timesteps.numel() < 2:
            raise ValueError("FSB staged scheduler state requires at least two timesteps")
        target_step = int(scheduler_state.get("target_step"))
        if target_step < 0 or target_step >= int(timesteps.numel()) - 1:
            raise ValueError("FSB staged scheduler state target_step is out of range")
        t_current = int(scheduler_state.get("t_current", int(timesteps[target_step].item())))
        t_next = int(scheduler_state.get("t_next", int(timesteps[target_step + 1].item())))
        eta = float(scheduler_state.get("eta", self.eta))

        corrected_x0 = projected_x0.to(self.device, dtype=dtype)
        if corrected_x0.ndim == 3:
            corrected_x0 = corrected_x0.unsqueeze(0)
        if projected_x0_is_physical:
            corrected_x0 = self._norm_from_physical(corrected_x0)

        x_t = self.i2sb_scheduler.step_from_x0(
            timestep=t_current,
            x0=corrected_x0,
            x1=x1,
            sample=x_t,
            timestep_next=t_next,
            eta=eta,
        )

        conditions = self.prepare_conditions(
            geometry=geometry,
            flow_conditions=flow_conditions,
            coords=coords,
            current_field=x1,
        )
        geometry = conditions["geometry"].to(self.device)
        flow_conditions = conditions["flow_conditions"].to(self.device)
        coords = conditions["coords"].to(self.device)

        for step_idx in range(target_step + 1, int(timesteps.numel()) - 1):
            t_cur = int(timesteps[step_idx].item())
            t_nxt = int(timesteps[step_idx + 1].item())
            t_batch = torch.full((x_t.shape[0],), t_cur, device=self.device, dtype=torch.long)
            model_output = self._predict_model(
                noisy_fields=x_t,
                timesteps=t_batch,
                geometry=geometry,
                flow_conditions=flow_conditions,
                coords=coords,
            )
            pred_x0 = self.i2sb_scheduler.reconstruct_x0(x_t, model_output, t_batch)
            if self.project_x0_to_physical_bounds:
                pred_x0 = self._project_x0_to_physical_bounds(pred_x0)
            x_t = self.i2sb_scheduler.step_from_x0(
                timestep=t_cur,
                x0=pred_x0,
                x1=x1,
                sample=x_t,
                timestep_next=t_nxt,
                eta=eta,
            )

        return self._physical_from_norm(x_t) if return_physical else x_t

    @torch.no_grad()
    def predict(
        self,
        *,
        initial_field: torch.Tensor,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        timesteps: Optional[Iterable[int] | torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Dict[str, Any]:
        conditions = self.prepare_conditions(
            geometry=geometry,
            flow_conditions=flow_conditions,
            coords=coords,
            current_field=initial_field,
        )
        return self._i2sb_denoise(
            surrogate_fields=initial_field,
            conditions=conditions,
            timesteps=timesteps,
            return_intermediates=return_intermediates,
        )


__all__ = ["FSBEngine"]
