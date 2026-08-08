"""Flow-state-bridge validation runner."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import torch

from surrogate.data import UniformFlowInitializer
from surrogate.evaluation.runners import (
    ValidationOptions,
    ValidationResult,
    coords_from_batch,
    evaluate_prediction_batches,
)
from surrogate.evaluation.reports import EvaluationReport
from surrogate.inference.backends import FSBPredictorBackend
from surrogate.inference.contracts import FSBPredictorConfig


class FSBValidationRunner:
    """Run training-time or offline validation for an FSB predictor backend."""

    def __init__(
        self,
        backend: FSBPredictorBackend,
        *,
        options: Optional[ValidationOptions] = None,
        uniform_initializer: Optional[UniformFlowInitializer] = None,
        residual_calculator: Any = None,
    ) -> None:
        self.backend = backend
        self.options = options or ValidationOptions()
        self.uniform_initializer = uniform_initializer or UniformFlowInitializer(
            normalizer=backend.normalizer,
            device=backend.device,
        )
        self.residual_calculator = residual_calculator

    def _initial_field_from(self, batch: Mapping[str, Any], coords: torch.Tensor) -> torch.Tensor:
        if "initial_field" in batch:
            return batch["initial_field"]
        if "x1" in batch:
            return batch["x1"]
        return self.uniform_initializer.generate_uniform_field(
            flow_conditions=batch["flow_conditions"],
            spatial_shape=(int(coords.shape[-2]), int(coords.shape[-1])),
            coords=coords,
        )

    def _predict_batch(self, batch: Mapping[str, Any]) -> torch.Tensor:
        coords = coords_from_batch(batch)
        initial_field = self._initial_field_from(batch, coords)
        return self.backend.predict(
            initial_field=initial_field,
            geometry=batch["geometry"],
            flow_conditions=batch["flow_conditions"],
            coords=coords,
            inverse_transform=False,
        )

    def evaluate(self, dataloader: Iterable[Mapping[str, Any]]) -> ValidationResult:
        return evaluate_prediction_batches(
            dataloader,
            predict_batch=self._predict_batch,
            device=self.backend.device,
            options=self.options,
            normalizer=self.backend.normalizer,
            residual_calculator=self.residual_calculator,
        )

    def evaluate_report(
        self,
        dataloader: Iterable[Mapping[str, Any]],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EvaluationReport:
        result = self.evaluate(dataloader)
        return EvaluationReport(
            result=result,
            model_family="fsb",
            model_key=self.backend.config.model.get_public_model_key(),
            metadata={
                "config": self.backend.config.to_dict(),
                **dict(metadata or {}),
            },
        )


def create_fsb_validation_runner(
    config_path: str,
    *,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    use_ema: bool = True,
    n_inference_steps: Optional[int] = None,
    custom_timesteps: Optional[list[int]] = None,
    eta: float = 0.0,
    noise_mode: str = "zeros",
    options: Optional[ValidationOptions] = None,
    residual_calculator: Any = None,
) -> FSBValidationRunner:
    """Create an FSB validation runner from config/checkpoint paths."""
    backend = FSBPredictorBackend(
        FSBPredictorConfig(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            use_ema=use_ema,
            n_inference_steps=n_inference_steps,
            custom_timesteps=custom_timesteps,
            eta=eta,
            noise_mode=noise_mode,
        )
    )
    return FSBValidationRunner(
        backend,
        options=options,
        residual_calculator=residual_calculator,
    )


__all__ = [
    "FSBValidationRunner",
    "create_fsb_validation_runner",
]
