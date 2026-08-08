"""Shared convolution blocks used by canonical surrogate models."""

import torch
import torch.nn as nn
from typing import Optional

from .conditioning import FiLM


class ResBlock(nn.Module):
    """
    Residual block with optional FiLM conditioning.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 activation: str = "relu",
                 norm_type: str = "batch",
                 use_norm: bool = True,
                 condition_dim: Optional[int] = None,
                 film_impl: str = "film",
                 dropout: float = 0.0,
                 stability_config: Optional[dict] = None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if padding is None:
            padding = kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_shortcut = (in_channels != out_channels or stride != 1)

        # Main branch layers
        layers = []

        # First convolution
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride, padding, bias=not use_norm))
        if use_norm:
            layers.append(self._get_norm_layer(norm_type, out_channels))
        layers.append(self._get_activation(activation))

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Second convolution
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size,
                                1, padding, bias=not use_norm))
        if use_norm:
            layers.append(self._get_norm_layer(norm_type, out_channels))

        self.main_branch = nn.Sequential(*layers)

        # Shortcut connection
        if self.use_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride)

        # Conditioning
        self.conditioning = None
        if condition_dim is not None:
            if film_impl == "film":
                self.conditioning = FiLM(
                    condition_dim, out_channels,
                    use_layer_norm=(norm_type == "layer"),
                    use_gate=False,
                    stability_config=stability_config,
                )
            else:
                raise ValueError(f"Unsupported FiLM implementation: {film_impl}")

    def _get_norm_layer(self, norm_type: str, num_channels: int) -> nn.Module:
        """Get normalization layer by type"""
        if norm_type == "batch":
            return nn.BatchNorm2d(num_channels)
        elif norm_type == "instance":
            return nn.InstanceNorm2d(num_channels)
        elif norm_type == "group":
            return nn.GroupNorm(8, num_channels)
        elif norm_type == "layer":
            return nn.GroupNorm(1, num_channels)  # Equivalent to LayerNorm for 4D
        elif norm_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        if activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU(inplace=True)
        elif activation.lower() == "swish":
            return nn.SiLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self,
                x: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (B, C, H, W) input tensor
            condition: (B, condition_dim) optional condition vector

        Returns:
            (B, out_channels, H', W') output tensor
        """
        residual = x

        # Main branch
        out = self.main_branch(x)

        # Apply conditioning if available
        if self.conditioning is not None and condition is not None:
            out = self.conditioning(out, condition)

        # Shortcut connection
        if self.use_shortcut:
            residual = self.shortcut(residual)

        return out + residual
