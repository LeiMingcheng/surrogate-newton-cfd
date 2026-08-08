"""Serving predictor adapters for direct and FSB backends."""

from __future__ import annotations

from typing import Any, Optional

import torch

from surrogate.data import UniformFlowInitializer
from surrogate.inference.backends import DirectPredictorBackend, FSBPredictorBackend
from surrogate.serving._tensors import expand_geometry, expand_spatial
from surrogate.serving.contracts import PredictionRequest, PredictionResponse


def _tensor_request(request: PredictionRequest, *, device: torch.device) -> PredictionRequest:
    flow_conditions = torch.as_tensor(
        request.flow_conditions,
        dtype=torch.float32,
        device=device,
    )
    if flow_conditions.ndim == 1:
        flow_conditions = flow_conditions.unsqueeze(0)
    if flow_conditions.ndim != 2 or int(flow_conditions.shape[1]) != 3:
        raise ValueError(
            "flow_conditions must have shape (3,) or (N,3); "
            f"got {tuple(flow_conditions.shape)}"
        )
    count = int(flow_conditions.shape[0])
    initial_field = None
    if request.initial_field is not None:
        initial_field = expand_spatial(
            request.initial_field,
            count=count,
            device=device,
            dtype=torch.float32,
            name="initial_field",
        )
    return PredictionRequest(
        geometry=expand_geometry(
            request.geometry,
            count=count,
            device=device,
        ),
        flow_conditions=flow_conditions,
        coords=expand_spatial(
            request.coords,
            count=count,
            device=device,
            dtype=torch.float32,
            name="coords",
        ),
        initial_field=initial_field,
        metadata=request.metadata,
    )


class DirectServingPredictor:
    """Transport-neutral serving wrapper around a direct predictor backend."""

    def __init__(self, backend: DirectPredictorBackend) -> None:
        self.backend = backend

    @torch.no_grad()
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        request = _tensor_request(request, device=self.backend.device)
        fields = self.backend.predict(
            geometry=request.geometry,
            flow_conditions=request.flow_conditions,
            coords=request.coords,
            initial_field=request.initial_field,
            inverse_transform=True,
        )
        return PredictionResponse(fields=fields, metadata=dict(request.metadata))


class FSBServingPredictor:
    """Transport-neutral serving wrapper around an FSB predictor backend."""

    def __init__(
        self,
        backend: FSBPredictorBackend,
        *,
        uniform_initializer: Optional[UniformFlowInitializer] = None,
    ) -> None:
        self.backend = backend
        self.uniform_initializer = uniform_initializer or UniformFlowInitializer(
            normalizer=backend.normalizer,
            device=backend.device,
        )

    def _initial_field_from(self, request: PredictionRequest) -> Any:
        if request.initial_field is not None:
            return request.initial_field
        coords = request.coords
        return self.uniform_initializer.generate_uniform_field(
            flow_conditions=request.flow_conditions,
            spatial_shape=(int(coords.shape[-2]), int(coords.shape[-1])),
            coords=coords,
        )

    @torch.no_grad()
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        request = _tensor_request(request, device=self.backend.device)
        timesteps = None
        if request.metadata.get("n_inference_steps") is not None:
            n_inference_steps = int(request.metadata["n_inference_steps"])
            if n_inference_steps < 1 or n_inference_steps > 20:
                raise ValueError("n_inference_steps must be between 1 and 20")
            timesteps = self.backend.engine.i2sb_scheduler.get_timesteps(
                n_inference_steps,
                self.backend.device,
            )
        fields = self.backend.predict(
            initial_field=self._initial_field_from(request),
            geometry=request.geometry,
            flow_conditions=request.flow_conditions,
            coords=request.coords,
            timesteps=timesteps,
            inverse_transform=True,
        )
        return PredictionResponse(fields=fields, metadata=dict(request.metadata))


__all__ = [
    "DirectServingPredictor",
    "FSBServingPredictor",
]
