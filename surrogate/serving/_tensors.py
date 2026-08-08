"""Internal tensor-shape helpers shared by serving adapters."""

from __future__ import annotations

from typing import Any

import torch


def as_1d_tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
    if int(tensor.numel()) <= 0:
        raise ValueError("Expected at least one scalar value")
    return tensor


def expand_geometry(
    geometry: Any,
    *,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(geometry, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0).expand(int(count), -1).contiguous()
    if tensor.ndim == 2 and int(tensor.shape[0]) == int(count):
        return tensor.contiguous()
    if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
        return tensor.expand(int(count), -1).contiguous()
    raise ValueError(
        f"geometry must have shape (G,), (1,G), or ({count},G); "
        f"got {tuple(tensor.shape)}"
    )


def expand_spatial(
    value: Any,
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0).expand(int(count), -1, -1, -1).contiguous()
    if tensor.ndim == 4 and int(tensor.shape[0]) == int(count):
        return tensor.contiguous()
    if tensor.ndim == 4 and int(tensor.shape[0]) == 1:
        return tensor.expand(int(count), -1, -1, -1).contiguous()
    raise ValueError(
        f"{name} must have shape (C,H,W), (1,C,H,W), or ({count},C,H,W); "
        f"got {tuple(tensor.shape)}"
    )


__all__ = [
    "as_1d_tensor",
    "expand_geometry",
    "expand_spatial",
]
