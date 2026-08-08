"""Model loading helpers for inference backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from surrogate.common.checkpointing import load_checkpoint_payload, load_model_checkpoint
from surrogate.configs import ExperimentConfig, load_config
from surrogate.data import create_normalizer
from surrogate.models import create_model


def load_experiment_config(config_or_path: ExperimentConfig | str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from an object or path."""
    if isinstance(config_or_path, ExperimentConfig):
        return config_or_path
    return load_config(config_or_path)


def create_loaded_model(
    config: ExperimentConfig,
    *,
    checkpoint_path: Optional[str | Path] = None,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate a canonical surrogate model and optionally load checkpoint weights."""
    torch_device = torch.device(device)
    model = create_model(config.model).to(torch_device)
    checkpoint_payload: dict[str, Any] = {}

    resolved_checkpoint = checkpoint_path or config.runtime.checkpoint
    if resolved_checkpoint is not None:
        checkpoint_payload = load_model_checkpoint(
            model,
            resolved_checkpoint,
            torch_device,
            use_ema=use_ema,
            context=f"inference checkpoint {resolved_checkpoint}",
        )

    model.eval()
    return model, checkpoint_payload


def create_normalizer_from_config(config: ExperimentConfig):
    """Create a field normalizer from config.data when enabled."""
    if not bool(config.data.normalize) and not bool(config.data.scale_turbulent):
        return None
    if config.data.stats_path is None:
        raise ValueError("data.stats_path is required when normalization is enabled")
    return create_normalizer(
        stats_path=config.data.stats_path,
        scale_turbulent=bool(config.data.scale_turbulent),
        normalize=bool(config.data.normalize),
    )


__all__ = [
    "create_loaded_model",
    "create_normalizer_from_config",
    "load_checkpoint_payload",
    "load_experiment_config",
]
