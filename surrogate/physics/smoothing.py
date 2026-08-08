"""Differentiable smoothing helpers for residual-field training objectives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor


def dct_type2(x: Tensor, dim: int = -2, norm: str = "ortho") -> Tensor:
    """Vectorized DCT-II implemented with a mirrored FFT."""

    n = x.shape[dim]
    device = x.device
    dtype = x.dtype
    x = x.transpose(dim, -1)
    original_shape = x.shape[:-1]
    x_flat = x.reshape(-1, n)
    x_ext = torch.cat([x_flat, x_flat.flip(dims=[-1])], dim=-1)
    spectrum = torch.fft.fft(x_ext, dim=-1)
    k = torch.arange(n, device=device, dtype=dtype)
    phase = torch.exp((-1j * math.pi * k / (2 * n)).to(torch.complex64))
    coeffs = (spectrum[..., :n] * phase).real
    if norm == "ortho":
        scale = torch.full((n,), math.sqrt(2.0 / n), device=device, dtype=dtype)
        scale[0] = math.sqrt(1.0 / n)
        coeffs = coeffs * scale
    coeffs = coeffs.reshape(*original_shape, n)
    return coeffs.transpose(-1, dim)


def idct_type2(x: Tensor, dim: int = -2, norm: str = "ortho") -> Tensor:
    """Inverse of :func:`dct_type2` for the orthonormal mode used here."""

    n = x.shape[dim]
    device = x.device
    dtype = x.dtype
    x = x.transpose(dim, -1)
    original_shape = x.shape[:-1]
    x_flat = x.reshape(-1, n)
    if norm == "ortho":
        scale = torch.full((n,), math.sqrt(n / 2.0), device=device, dtype=dtype)
        scale[0] = math.sqrt(float(n))
        x_flat = x_flat * scale
    k = torch.arange(n, device=device, dtype=dtype)
    phase = torch.exp((1j * math.pi * k / (2 * n)).to(torch.complex64))
    spectrum = torch.zeros(x_flat.shape[0], 2 * n, dtype=torch.complex64, device=device)
    spectrum[..., :n] = x_flat.to(torch.complex64) * phase
    spectrum[..., n + 1 :] = spectrum[..., 1:n].flip(dims=[-1]).conj()
    recovered = torch.fft.ifft(spectrum, dim=-1).real[..., :n]
    recovered = recovered.reshape(*original_shape, n)
    return recovered.transpose(-1, dim)


@dataclass(frozen=True)
class SobolevSmoothingConfig:
    """Fixed-parameter Sobolev smoothing for signed residual fields."""

    lambda_eta: float = 4.0
    lambda_xi: float = 1.0


class SobolevResidualSmoother:
    """Apply ``(I - lambda_eta Delta_eta - lambda_xi Delta_xi)^-1`` to a field."""

    def __init__(self, config: Optional[SobolevSmoothingConfig] = None) -> None:
        self.config = config or SobolevSmoothingConfig()
        self._eig_cache: dict[tuple[str, str, int, int], tuple[Tensor, Tensor]] = {}

    def _get_eigs(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        key = (str(device), str(dtype), int(height), int(width))
        cached = self._eig_cache.get(key)
        if cached is not None:
            return cached
        k_eta = torch.arange(height, device=device, dtype=dtype)
        eig_eta = 4.0 * torch.sin(math.pi * k_eta / (2.0 * height)).pow(2)
        k_xi = torch.arange(width // 2 + 1, device=device, dtype=dtype)
        eig_xi = 4.0 * torch.sin(math.pi * k_xi / float(width)).pow(2)
        self._eig_cache[key] = (eig_eta, eig_xi)
        return eig_eta, eig_xi

    def apply(
        self,
        field: Tensor,
        *,
        lambda_eta: Optional[float] = None,
        lambda_xi: Optional[float] = None,
        wall_layers: Optional[int] = None,
    ) -> Tensor:
        if field.ndim != 4:
            raise ValueError(f"Expected field shape (B, C, H, W), got {tuple(field.shape)}")
        if wall_layers is None or int(wall_layers) >= int(field.shape[-2]):
            return self._apply_block(
                field,
                lambda_eta=float(self.config.lambda_eta if lambda_eta is None else lambda_eta),
                lambda_xi=float(self.config.lambda_xi if lambda_xi is None else lambda_xi),
            )
        layers = max(1, min(int(wall_layers), int(field.shape[-2])))
        out = field.clone()
        out[:, :, :layers, :] = self._apply_block(
            out[:, :, :layers, :],
            lambda_eta=float(self.config.lambda_eta if lambda_eta is None else lambda_eta),
            lambda_xi=float(self.config.lambda_xi if lambda_xi is None else lambda_xi),
        )
        return out

    def _apply_block(self, field: Tensor, *, lambda_eta: float, lambda_xi: float) -> Tensor:
        batch, _, height, width = field.shape
        eig_eta, eig_xi = self._get_eigs(height, width, device=field.device, dtype=field.dtype)
        lambda_eta_t = torch.full((batch,), float(lambda_eta), device=field.device, dtype=field.dtype)
        lambda_xi_t = torch.full((batch,), float(lambda_xi), device=field.device, dtype=field.dtype)
        denom = (
            1.0
            + lambda_eta_t[:, None, None] * eig_eta[None, :, None]
            + lambda_xi_t[:, None, None] * eig_xi[None, None, :]
        )
        field_dct = dct_type2(field, dim=-2, norm="ortho")
        field_hat = torch.fft.rfft(field_dct, dim=-1)
        field_hat = field_hat / denom.unsqueeze(1)
        field_ifft = torch.fft.irfft(field_hat, n=width, dim=-1)
        return idct_type2(field_ifft, dim=-2, norm="ortho")


__all__ = [
    "SobolevResidualSmoother",
    "SobolevSmoothingConfig",
    "dct_type2",
    "idct_type2",
]
