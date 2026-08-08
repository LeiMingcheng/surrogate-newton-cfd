"""Uniform-flow initializer for FSB training and inference."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch


class UniformFlowInitializer:
    """Generate ADflow-style uniform initial fields from [Mach, AoA, Re]."""

    def __init__(
        self,
        gamma: float = 1.4,
        cv1: float = 7.1,
        eddy_vis_ratio: float = 0.009,
        normalizer: Any = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.gamma = float(gamma)
        self.cv1 = float(cv1)
        self.eddy_vis_ratio = float(eddy_vis_ratio)
        self.normalizer = normalizer
        self.device = device
        self.sqrt_gamma = math.sqrt(self.gamma)
        self.cv1_cubed = self.cv1 ** 3

    def _solve_sa_nu_tilde_batch(
        self,
        eddy_ratio: float,
        nu_lam: torch.Tensor,
        max_iter: int = 50,
        tol: float = 1.0e-10,
    ) -> torch.Tensor:
        if eddy_ratio <= 0:
            return torch.zeros_like(nu_lam)

        if eddy_ratio < 1.0e-4:
            chi = torch.full_like(nu_lam, 0.5)
        elif eddy_ratio < 1.0:
            chi = torch.full_like(nu_lam, 5.0)
        elif eddy_ratio < 10.0:
            chi = torch.full_like(nu_lam, 10.0)
        else:
            chi = torch.full_like(nu_lam, eddy_ratio)

        for _ in range(max_iter):
            chi2 = chi * chi
            chi3 = chi * chi2
            chi4 = chi * chi3
            f = chi4 - eddy_ratio * (chi3 + self.cv1_cubed)
            df = 4.0 * chi3 - 3.0 * eddy_ratio * chi2
            df = torch.where(df.abs() < 1.0e-20, torch.full_like(df, 1.0e-20), df)
            dchi = f / df
            chi = chi - dchi
            if (dchi.abs() < tol * chi.abs().clamp_min(1.0e-20)).all():
                break

        return nu_lam * chi

    def generate_uniform_field(
        self,
        flow_conditions: torch.Tensor,
        spatial_shape: Tuple[int, int],
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del coords
        batch_size = int(flow_conditions.shape[0])
        height, width = int(spatial_shape[0]), int(spatial_shape[1])
        device = flow_conditions.device

        mach = flow_conditions[:, 0]
        aoa_rad = flow_conditions[:, 1] * (math.pi / 180.0)
        reynolds = flow_conditions[:, 2]

        u_inf = mach * self.sqrt_gamma
        u = u_inf * torch.cos(aoa_rad)
        v = u_inf * torch.sin(aoa_rad)
        rho = torch.ones(batch_size, device=device, dtype=flow_conditions.dtype)
        p = torch.ones(batch_size, device=device, dtype=flow_conditions.dtype)
        nu_lam = u_inf / reynolds
        nu_tilde = self._solve_sa_nu_tilde_batch(self.eddy_vis_ratio, nu_lam)

        fields = torch.zeros(batch_size, 5, height, width, device=device, dtype=flow_conditions.dtype)
        fields[:, 0, :, :] = rho.view(batch_size, 1, 1)
        fields[:, 1, :, :] = u.view(batch_size, 1, 1)
        fields[:, 2, :, :] = v.view(batch_size, 1, 1)
        fields[:, 3, :, :] = p.view(batch_size, 1, 1)
        fields[:, 4, :, :] = nu_tilde.view(batch_size, 1, 1)

        if self.normalizer is not None:
            fields = self.normalizer.transform(fields)
        return fields

    def __repr__(self) -> str:
        return (
            "UniformFlowInitializer("
            f"gamma={self.gamma}, cv1={self.cv1}, eddy_vis_ratio={self.eddy_vis_ratio}, "
            f"normalizer={'yes' if self.normalizer is not None else 'no'})"
        )


def create_uniform_flow_initializer(
    config: Optional[dict[str, Any]] = None,
    normalizer: Any = None,
    device: str | torch.device = "cuda",
) -> UniformFlowInitializer:
    config = dict(config or {})
    return UniformFlowInitializer(
        gamma=float(config.get("gamma", 1.4)),
        cv1=float(config.get("cv1", 7.1)),
        eddy_vis_ratio=float(config.get("eddy_vis_ratio", 0.009)),
        normalizer=normalizer,
        device=device,
    )


__all__ = [
    "UniformFlowInitializer",
    "create_uniform_flow_initializer",
]
