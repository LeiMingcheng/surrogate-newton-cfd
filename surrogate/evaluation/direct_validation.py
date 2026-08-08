"""Direct-model validation runner."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Any

import torch

from surrogate.evaluation.runners import (
    ValidationOptions,
    ValidationResult,
    coords_from_batch,
    evaluate_prediction_batches,
)
from surrogate.evaluation.reports import EvaluationReport
from surrogate.inference.backends import DirectPredictorBackend
from surrogate.inference.contracts import DirectPredictorConfig


class DirectValidationRunner:
    """Run training-time or offline validation for a direct predictor backend."""

    def __init__(
        self,
        backend: DirectPredictorBackend,
        *,
        options: Optional[ValidationOptions] = None,
        residual_calculator: Any = None,
    ) -> None:
        self.backend = backend
        self.options = options or ValidationOptions()
        self.residual_calculator = residual_calculator

    def _predict_batch(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.backend.predict(
            geometry=batch["geometry"],
            flow_conditions=batch["flow_conditions"],
            coords=coords_from_batch(batch),
            initial_field=batch.get("initial_field"),
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
            model_family="direct",
            model_key=self.backend.config.model.get_public_model_key(),
            metadata={
                "config": self.backend.config.to_dict(),
                **dict(metadata or {}),
            },
        )


def create_direct_validation_runner(
    config_path: str,
    *,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    use_ema: bool = True,
    options: Optional[ValidationOptions] = None,
    residual_calculator: Any = None,
) -> DirectValidationRunner:
    """Create a direct validation runner from config/checkpoint paths."""
    backend = DirectPredictorBackend(
        DirectPredictorConfig(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            use_ema=use_ema,
        )
    )
    return DirectValidationRunner(
        backend,
        options=options,
        residual_calculator=residual_calculator,
    )


__all__ = [
    "DirectValidationRunner",
    "create_direct_validation_runner",
]
