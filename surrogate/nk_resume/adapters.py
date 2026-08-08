"""Predictor adapters for ADflow resume workflows."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from surrogate.data import UniformFlowInitializer
from surrogate.inference.backends import DirectPredictorBackend, FSBPredictorBackend
from surrogate.nk_resume.contracts import ResumePrediction, ResumeRequest


class DirectResumePredictorAdapter:
    """Adapt a direct predictor backend to the resume request contract."""

    def __init__(self, backend: DirectPredictorBackend) -> None:
        self.backend = backend

    @torch.no_grad()
    def predict(self, request: ResumeRequest) -> ResumePrediction:
        fields = self.backend.predict(
            geometry=request.geometry,
            flow_conditions=request.flow_conditions,
            coords=request.coords,
            initial_field=request.initial_field,
            inverse_transform=True,
        )
        return ResumePrediction(fields=fields, metadata=dict(request.metadata))


class FSBResumePredictorAdapter:
    """Adapt an FSB predictor backend to the resume request contract."""

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

    def _initial_field_from(self, request: ResumeRequest) -> Any:
        if request.initial_field is not None:
            return request.initial_field
        coords = request.coords
        return self.uniform_initializer.generate_uniform_field(
            flow_conditions=request.flow_conditions,
            spatial_shape=(int(coords.shape[-2]), int(coords.shape[-1])),
            coords=coords,
        )

    @torch.no_grad()
    def predict(self, request: ResumeRequest) -> ResumePrediction:
        fields = self.backend.predict(
            initial_field=self._initial_field_from(request),
            geometry=request.geometry,
            flow_conditions=request.flow_conditions,
            coords=request.coords,
            inverse_transform=True,
        )
        return ResumePrediction(fields=fields, metadata=dict(request.metadata))


__all__ = [
    "DirectResumePredictorAdapter",
    "FSBResumePredictorAdapter",
]
