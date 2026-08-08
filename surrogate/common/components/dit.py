"""Shared DiT building blocks for direct and FSB models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from surrogate.common.components.conditioning import AdaLNZero
from surrogate.common.components.stability import resolve_stability_clamp_head_config


COMPAT_MODE_LEGACY_HARD_ADALN = "legacy_hard_adaln_clamp"
COMPAT_MODE_MODERN_MODULATION = "modern_modulation_clamp"


def resolve_modulation_stability_config(
    stability_config: Optional[dict],
) -> dict:
    """Resolve modulation-clamp config without reinterpreting hard-clamp configs."""
    stability_dict = dict(stability_config or {})
    if "clamp_head" in stability_dict:
        return resolve_stability_clamp_head_config(stability_dict)

    return resolve_stability_clamp_head_config({
        "enabled": False,
        "clean_output": stability_dict.get("clean_output", True),
    })


def infer_dit_compatibility_mode(
    stability_config: Optional[dict],
) -> str:
    """Infer DiT modulation compatibility mode from model config."""
    stability_dict = dict(stability_config or {})
    explicit_mode = stability_dict.get("compatibility_mode")
    if explicit_mode is not None:
        mode = str(explicit_mode).lower()
        if mode not in {COMPAT_MODE_LEGACY_HARD_ADALN, COMPAT_MODE_MODERN_MODULATION}:
            raise ValueError(
                f"Unsupported stability.compatibility_mode={explicit_mode!r}. "
                f"Expected one of: {COMPAT_MODE_LEGACY_HARD_ADALN}, {COMPAT_MODE_MODERN_MODULATION}."
            )
        return mode

    if "clamp_head" in stability_dict:
        return COMPAT_MODE_MODERN_MODULATION
    if bool(stability_dict.get("adaln_clamp", False)):
        return COMPAT_MODE_LEGACY_HARD_ADALN
    return COMPAT_MODE_MODERN_MODULATION


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        condition_dim: int,
        mlp_ratio: float = 4.0,
        adaln_clamp: bool = False,
        compatibility_mode: str = COMPAT_MODE_MODERN_MODULATION,
        stability_config: Optional[dict] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.adaln_clamp = adaln_clamp
        self.compatibility_mode = COMPAT_MODE_MODERN_MODULATION

        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
            dropout=0.0,
        )

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

        self.adaln_zero = AdaLNZero(
            condition_dim,
            hidden_dim,
            stability_config=stability_config,
            enable_stability_clamp=False,
        )
        self.set_compatibility_mode(compatibility_mode)

    def set_compatibility_mode(self, mode: str) -> None:
        if mode not in {COMPAT_MODE_LEGACY_HARD_ADALN, COMPAT_MODE_MODERN_MODULATION}:
            raise ValueError(
                f"Unsupported DiTBlock compatibility mode: {mode!r}. "
                f"Expected one of: {COMPAT_MODE_LEGACY_HARD_ADALN}, {COMPAT_MODE_MODERN_MODULATION}."
            )
        self.compatibility_mode = mode
        self.use_legacy_hard_adaln_clamp = (
            mode == COMPAT_MODE_LEGACY_HARD_ADALN and bool(self.adaln_clamp)
        )
        self.adaln_zero.set_use_stability_clamp(
            mode == COMPAT_MODE_MODERN_MODULATION
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.adaln_zero(x, condition)

        if self.use_legacy_hard_adaln_clamp:
            gate_msa = torch.tanh(gate_msa)
            gate_mlp = torch.tanh(gate_mlp)
            scale_msa = torch.clamp(scale_msa, -2.0, 2.0)
            scale_mlp = torch.clamp(scale_mlp, -2.0, 2.0)
            shift_msa = torch.clamp(shift_msa, -5.0, 5.0)
            shift_mlp = torch.clamp(shift_mlp, -5.0, 5.0)

        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa * attn_out

        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp * mlp_out

        return x


__all__ = [
    "COMPAT_MODE_LEGACY_HARD_ADALN",
    "COMPAT_MODE_MODERN_MODULATION",
    "DiTBlock",
    "infer_dit_compatibility_mode",
    "resolve_modulation_stability_config",
]
