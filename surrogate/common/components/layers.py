"""Small shared neural-network layers used by multiple surrogate backbones."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels for BCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for bridge timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -scale)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        if embeddings.shape[-1] < self.dim:
            embeddings = F.pad(embeddings, (0, self.dim - embeddings.shape[-1]))
        return embeddings


__all__ = [
    "ChannelLayerNorm2d",
    "SinusoidalPositionEmbeddings",
]
