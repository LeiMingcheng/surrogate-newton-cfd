"""Tensor utility helpers used across runtime adapters."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def detach_to_cpu(value: Any) -> Any:
    """Detach tensors to CPU while leaving non-tensors unchanged."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    return value


def move_mapping_to_device(values: Mapping[str, Any], device: str | torch.device) -> dict[str, Any]:
    """Move tensor values in a mapping to device."""
    torch_device = torch.device(device)
    return {
        key: value.to(torch_device) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


__all__ = [
    "detach_to_cpu",
    "move_mapping_to_device",
]
