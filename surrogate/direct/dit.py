"""
Single-step DiT surrogate for direct flow-field prediction.

This keeps the surrogate training interface unchanged:
    geometry + flow_conditions + coords -> fields
while replacing the UNet/FNO backbone with a DiT-style patch transformer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from surrogate.common import BaseDirectModel
from surrogate.common.components import GeometryConditionEncoder
from surrogate.common.components.dit import (
    COMPAT_MODE_MODERN_MODULATION,
    DiTBlock,
    infer_dit_compatibility_mode,
    resolve_modulation_stability_config,
)


class DirectDiT(BaseDirectModel):
    """Direct-regression DiT surrogate on the O-grid."""

    def __init__(
        self,
        geometry_dim: int = 27,
        flow_condition_dim: int = 3,
        coord_channels: int = 4,
        out_channels: int = 5,
        hidden_dim: int = 640,
        num_layers: int = 12,
        num_heads: int = 8,
        patch_size: int = 8,
        mlp_ratio: float = 4.0,
        condition_dim: Optional[int] = None,
        spatial_shape: Optional[Tuple[int, int]] = (88, 304),
        use_gradient_checkpointing: bool = True,
        geometry_condition: Optional[dict] = None,
        use_initial_field: bool = False,
        stability: Optional[dict] = None,
    ):
        super().__init__(geometry_dim, flow_condition_dim, coord_channels, out_channels)

        if spatial_shape is None:
            raise ValueError("DirectDiT requires model.spatial_shape to be set")
        if len(spatial_shape) != 2:
            raise ValueError(f"spatial_shape must be (H, W), got {spatial_shape}")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.patch_size = int(patch_size)
        self.mlp_ratio = float(mlp_ratio)
        self.condition_dim = int(condition_dim or hidden_dim)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_initial_field = bool(use_initial_field)

        height, width = self.spatial_shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"spatial_shape={self.spatial_shape} must be divisible by patch_size={self.patch_size}"
            )

        self.h_patches = height // self.patch_size
        self.w_patches = width // self.patch_size
        self.num_patches = self.h_patches * self.w_patches

        stability_config = dict(stability or {})
        compatibility_mode = infer_dit_compatibility_mode(stability_config)
        modulation_stability_config = resolve_modulation_stability_config(stability_config)
        if compatibility_mode != COMPAT_MODE_MODERN_MODULATION and not stability_config.get("adaln_clamp", False):
            compatibility_mode = COMPAT_MODE_MODERN_MODULATION

        self.geometry_condition_encoder = GeometryConditionEncoder(
            geometry_dim=geometry_dim,
            output_dim=geometry_dim,
            coord_channels=coord_channels,
            config=geometry_condition,
        )

        self.condition_encoder = nn.Sequential(
            nn.Linear(geometry_dim + flow_condition_dim, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )

        self.patch_embed = nn.Conv2d(
            coord_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.constant_(self.patch_embed.bias, 0)
        self.initial_field_patch_embed: Optional[nn.Module] = None
        if self.use_initial_field:
            self.initial_field_patch_embed = nn.Conv2d(
                out_channels,
                self.hidden_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            nn.init.normal_(self.initial_field_patch_embed.weight, std=0.02)
            nn.init.constant_(self.initial_field_patch_embed.bias, 0)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    condition_dim=self.condition_dim,
                    mlp_ratio=self.mlp_ratio,
                    adaln_clamp=bool(stability_config.get("adaln_clamp", False)),
                    compatibility_mode=compatibility_mode,
                    stability_config=modulation_stability_config,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.hidden_dim * 2),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.final_linear = nn.Linear(
            self.hidden_dim,
            self.patch_size * self.patch_size * out_channels,
        )
        nn.init.normal_(self.final_linear.weight, std=0.02)
        nn.init.zeros_(self.final_linear.bias)

    @staticmethod
    def _normalize_checkpoint_keys(state_dict):
        return {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }

    def configure_legacy_compat_from_state_dict(
        self,
        state_dict,
        *,
        logger_fn=None,
        context: str = "checkpoint",
    ):
        normalized_state = self._normalize_checkpoint_keys(state_dict)
        has_initial_field_branch = any(
            key.startswith("initial_field_patch_embed.")
            or ".initial_field_patch_embed." in key
            for key in normalized_state
        )
        if not has_initial_field_branch and self.initial_field_patch_embed is not None:
            self.initial_field_patch_embed = None
            self.use_initial_field = False
            if logger_fn is not None:
                logger_fn(
                    "Disabled DirectDiT initial-field branch while loading "
                    f"{context}; checkpoint does not contain "
                    "initial_field_patch_embed.*"
                )
        return "no_initial_field" if not has_initial_field_branch else "initial_field"

    def _get_pos_embed_for_patch_grid(
        self,
        actual_h_patches: int,
        actual_w_patches: int,
    ) -> torch.Tensor:
        actual_num_patches = actual_h_patches * actual_w_patches
        if actual_num_patches == self.num_patches:
            return self.pos_embed

        pos_embed_reshaped = rearrange(
            self.pos_embed,
            "1 (h w) d -> 1 d h w",
            h=self.h_patches,
            w=self.w_patches,
        )
        pos_embed_interpolated = F.interpolate(
            pos_embed_reshaped,
            size=(actual_h_patches, actual_w_patches),
            mode="bilinear",
            align_corners=False,
        )
        return rearrange(pos_embed_interpolated, "1 d h w -> 1 (h w) d")

    def _unpatchify(
        self,
        patch_tokens: torch.Tensor,
        actual_h_patches: Optional[int] = None,
        actual_w_patches: Optional[int] = None,
    ) -> torch.Tensor:
        h_patches = self.h_patches if actual_h_patches is None else int(actual_h_patches)
        w_patches = self.w_patches if actual_w_patches is None else int(actual_w_patches)
        return rearrange(
            patch_tokens,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h_patches,
            w=w_patches,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )

    def forward(
        self,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        initial_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.validate_inputs(geometry, flow_conditions, coords):
            raise ValueError("Invalid input shapes for DirectDiT")

        initial_field = self.resolve_initial_field(flow_conditions, coords, initial_field)
        flow_conditions = self.preprocess_flow_conditions(flow_conditions)
        geometry_condition = self.geometry_condition_encoder(geometry, coords)
        condition = self.condition_encoder(torch.cat([geometry_condition, flow_conditions], dim=1))

        input_height, input_width = coords.shape[-2:]
        padded_height = ((input_height + self.patch_size - 1) // self.patch_size) * self.patch_size
        padded_width = ((input_width + self.patch_size - 1) // self.patch_size) * self.patch_size
        pad_h = padded_height - input_height
        pad_w = padded_width - input_width
        if pad_h > 0 or pad_w > 0:
            coords = F.pad(coords, (0, pad_w, 0, pad_h), mode="replicate")
            if initial_field is not None:
                initial_field = F.pad(initial_field, (0, pad_w, 0, pad_h), mode="replicate")

        x = self.patch_embed(coords)
        if self.initial_field_patch_embed is not None:
            if initial_field is None:
                raise ValueError("DirectDiT requires initial_field when use_initial_field=True")
            x = x + self.initial_field_patch_embed(initial_field)
        actual_h_patches, actual_w_patches = x.shape[-2:]
        x = rearrange(x, "b d h w -> b (h w) d")
        x = x + self._get_pos_embed_for_patch_grid(actual_h_patches, actual_w_patches)

        use_ckpt = self.use_gradient_checkpointing and self.training and torch.is_grad_enabled()
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint

        for block in self.blocks:
            if use_ckpt:
                x = checkpoint(block, x, condition, use_reentrant=False)
            else:
                x = block(x, condition)

        scale, shift = self.final_adaLN(condition).chunk(2, dim=1)
        x = self.final_norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_linear(x)

        x = self._unpatchify(x, actual_h_patches, actual_w_patches)
        return x[:, :, :input_height, :input_width]
