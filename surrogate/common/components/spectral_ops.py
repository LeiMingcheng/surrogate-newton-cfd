"""Shared spectral operations used by canonical FNO models."""

import torch
import torch.nn as nn


class FourierSpectralConv2d(nn.Module):
    """FNO spectral convolution with separate positive/negative frequency weights."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        op_real = torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 0]) - (
            torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 1])
        )
        op_imag = torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 1]) + (
            torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 0])
        )
        return torch.stack([op_real, op_imag], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        height, width = x.shape[-2:]

        x_ft = torch.fft.rfft2(x)
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            2,
            device=x.device,
            dtype=x.dtype,
        )

        out_ft[:, :, :self.modes1, :self.modes2, :] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2, :],
            self.weights1,
        )
        out_ft[:, :, -self.modes1:, :self.modes2, :] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2, :],
            self.weights2,
        )

        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        return torch.fft.irfft2(out_ft_complex, s=(height, width))
