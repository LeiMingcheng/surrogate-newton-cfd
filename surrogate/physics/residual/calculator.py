"""Residual calculator facade for surrogate PDE backends."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

from surrogate.physics.residual.weights import ResidualWeights, parse_residual_weights
from surrogate.physics.residual.torch_backend import TorchResidualBackend


class PDEResidualCalculator:
    """Backend dispatcher for PDE residual computation.

    The canonical surrogate package currently provides the differentiable
    Torch backend. ADflow execution is an external runtime boundary and will
    be migrated separately under a clean ADflow-specific package.
    """

    def __init__(
        self,
        backend: str = "torch",
        viscosity_mode: str = "laminar+SA",
        residual_norm_mode: str = "standard",
        weighting: str = "uniform",
        weighting_params: Optional[Dict[str, float]] = None,
        gamma: float = 1.4,
        device: str = "cpu",
        allow_fallback_to_euler: bool = False,
        backend_config: Optional[Dict[str, Any]] = None,
        gradient_method: str = "adflow_6pt",
        compute_only_momentum: bool = False,
    ) -> None:
        backend = str(backend).lower()
        if backend == "pytorch":
            backend = "torch"
        if backend != "torch":
            raise NotImplementedError(
                "surrogate.physics.residual currently supports backend='torch' only. "
                "ADflow residual runtime will be migrated separately and must not import the old package."
            )

        if gradient_method != "adflow_6pt":
            raise ValueError(
                "Torch residual backend requires gradient_method='adflow_6pt'."
            )
        if viscosity_mode != "laminar+SA":
            raise ValueError(
                "Torch residual backend requires viscosity_mode='laminar+SA'."
            )

        torch_config = dict(backend_config or {})
        self.backend = backend
        self.device = device
        self.gamma = float(gamma)
        self.residual_norm_mode = residual_norm_mode
        self.viscosity_mode = viscosity_mode
        self.weighting = weighting
        self.allow_fallback_to_euler = bool(allow_fallback_to_euler)
        self.include_viscosity = viscosity_mode != "none"
        self._calc = TorchResidualBackend(
            viscosity_mode=viscosity_mode,
            residual_norm_mode=residual_norm_mode,
            weighting=weighting,
            weighting_params=weighting_params,
            gamma=gamma,
            device=device,
            allow_fallback_to_euler=allow_fallback_to_euler,
            gradient_method=gradient_method,
            adis=torch_config.get("adis", 0.67),
            acoustic_scale_factor=torch_config.get("acoustic_scale_factor", 1.0),
            compute_only_momentum=compute_only_momentum,
            geometry_fast_cache_max_entries=torch_config.get("geometry_fast_cache_max_entries", 1),
        )

    def compute_residual_score(
        self,
        spatial_map: Union[np.ndarray, torch.Tensor],
        weights: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Compute a scalar residual score from a precomputed spatial map."""
        if isinstance(spatial_map, torch.Tensor):
            spatial_map = spatial_map.detach().cpu().numpy()

        n_channels = spatial_map.shape[0]
        if n_channels not in (3, 4, 5):
            raise ValueError(
                "spatial_map must have 3, 4, or 5 channels, "
                f"got shape {spatial_map.shape}"
            )

        default_weights = ResidualWeights()
        wc, wmx, wmy, w_energy, w_turbulence = parse_residual_weights(
            dict(weights or {}),
            wc_default=default_weights.wc,
            wmx_default=default_weights.wmx,
            wmy_default=default_weights.wmy,
            energy_default=1.0,
            turbulence_default=1.0,
        )

        rc_norm = float(np.sqrt(np.mean(spatial_map[0] ** 2)))
        rmx_norm = float(np.sqrt(np.mean(spatial_map[1] ** 2)))
        rmy_norm = float(np.sqrt(np.mean(spatial_map[2] ** 2)))
        score = -(wc * rc_norm + wmx * rmx_norm + wmy * rmy_norm)
        if n_channels >= 4:
            score -= w_energy * float(np.sqrt(np.mean(spatial_map[3] ** 2)))
        if n_channels == 5:
            score -= w_turbulence * float(np.sqrt(np.mean(spatial_map[4] ** 2)))
        return score

    def compute_residual(self, *args: Any, **kwargs: Any):
        """Delegate residual computation to the selected backend."""
        score, result = self._calc.compute_residual(*args, **kwargs)
        return score, result

    def compute_batch_residuals(self, batch_data: list[dict[str, Any]]) -> tuple[list[float], list[dict[str, Any]]]:
        """Compute residuals for a homogeneous list as one Torch batch."""
        if not batch_data:
            return [], []

        def _collate(values: list[Any]) -> Any:
            first = values[0]
            if isinstance(first, torch.Tensor):
                return torch.cat(values, dim=0)
            if isinstance(first, Mapping):
                return {
                    key: _collate([value[key] for value in values])
                    if isinstance(first[key], (torch.Tensor, Mapping))
                    else first[key]
                    for key in first
                }
            return first

        batched = {
            key: _collate([sample[key] for sample in batch_data])
            if isinstance(value, (torch.Tensor, Mapping)) and key in {
                "fields",
                "coords",
                "flow_conditions",
                "wall_distance",
                "wall_segment_mask",
            }
            else value
            for key, value in batch_data[0].items()
        }
        score, result = self.compute_residual(**batched)
        batch_size = len(batch_data)
        score_tensor = torch.as_tensor(score).detach().reshape(-1)
        if score_tensor.numel() != batch_size:
            raise ValueError(
                "Batched residual backend returned "
                f"{score_tensor.numel()} scores for {batch_size} samples"
            )

        def _sample_value(value: Any, sample_index: int) -> Any:
            if isinstance(value, torch.Tensor):
                if value.ndim > 0 and value.shape[0] == batch_size:
                    return value[sample_index]
                return value
            if isinstance(value, Mapping):
                return {
                    key: _sample_value(item, sample_index)
                    for key, item in value.items()
                }
            return value

        scores = [float(value) for value in score_tensor.cpu().tolist()]
        results = [
            {
                key: _sample_value(value, sample_index)
                for key, value in result.items()
            }
            for sample_index in range(batch_size)
        ]
        return scores, results

    def __call__(self, *args: Any, **kwargs: Any):
        return self.compute_residual(*args, **kwargs)


__all__ = ["PDEResidualCalculator"]
