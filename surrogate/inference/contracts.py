"""Inference contracts for surrogate runtime backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PredictorConfig:
    """Common predictor loading config."""

    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
    use_ema: bool = True


@dataclass(frozen=True)
class DirectPredictorConfig(PredictorConfig):
    """Direct-model predictor config."""


@dataclass(frozen=True)
class FSBPredictorConfig(PredictorConfig):
    """FSB predictor config."""

    n_inference_steps: Optional[int] = None
    custom_timesteps: Optional[list[int]] = None
    eta: float = 0.0
    noise_mode: str = "zeros"


@dataclass
class PredictionBatch:
    """Tensor batch accepted by direct and FSB predictors."""

    geometry: Any
    flow_conditions: Any
    coords: Any
    initial_field: Any = None


__all__ = [
    "DirectPredictorConfig",
    "FSBPredictorConfig",
    "PredictionBatch",
    "PredictorConfig",
]
