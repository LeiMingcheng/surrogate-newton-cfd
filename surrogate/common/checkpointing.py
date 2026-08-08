"""Checkpoint loading helpers shared by training and inference entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch

from surrogate.common.components import load_state_dict_with_stability_head_compat
from surrogate.common.ema import EMAModel


def load_checkpoint_payload(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a checkpoint payload and return model state plus raw checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, checkpoint
    raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint)}")


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    use_ema: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    context: str = "checkpoint",
) -> dict[str, Any]:
    """Load checkpoint weights into an already-created model."""
    state_dict, checkpoint_payload = load_checkpoint_payload(checkpoint_path, device)
    load_state_dict_with_stability_head_compat(
        model,
        state_dict,
        logger=logger,
        context=context,
    )
    if use_ema and "ema_state_dict" in checkpoint_payload:
        ema = EMAModel(model, decay=float(checkpoint_payload["ema_state_dict"].get("decay", 0.999)))
        ema.load_state_dict(checkpoint_payload["ema_state_dict"])
        ema.apply_shadow()
    return checkpoint_payload


__all__ = [
    "load_checkpoint_payload",
    "load_model_checkpoint",
]
