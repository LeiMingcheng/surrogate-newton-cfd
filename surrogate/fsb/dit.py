"""DiT backbone for the retained flow-state-bridge path."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from surrogate.common import BaseBridgeModel
from surrogate.common.components import GeometryConditionEncoder, MultimodalConditionFusion
from surrogate.common.components.dit import (
    COMPAT_MODE_LEGACY_HARD_ADALN,
    COMPAT_MODE_MODERN_MODULATION,
    DiTBlock,
    infer_dit_compatibility_mode as _infer_fsb_dit_compatibility_mode,
    resolve_modulation_stability_config as _resolve_modulation_stability_config,
)
from surrogate.common.components.layers import SinusoidalPositionEmbeddings

logger = logging.getLogger(__name__)


class PatchEmbed2d(nn.Module):
    """Patch embedding with checkpoint-compatible weights and selectable execution path."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        patch_size: int,
        mode: str = "conv2d",
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.mode = str(mode).lower()
        if self.mode not in {"conv2d", "serial_conv2d", "patchify_linear"}:
            raise ValueError(
                f"Unsupported patch embed mode={mode!r}. "
                "Expected one of: conv2d, serial_conv2d, patchify_linear."
            )

        self.weight = nn.Parameter(
            torch.empty(self.hidden_dim, self.input_channels, self.patch_size, self.patch_size)
        )
        self.bias = nn.Parameter(torch.empty(self.hidden_dim))
        self._reset_parameters_like_legacy_conv2d()

    def _reset_parameters_like_legacy_conv2d(self) -> None:
        # Preserve the legacy nn.Conv2d constructor RNG stream before checkpoint load.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "conv2d":
            return F.conv2d(x, self.weight, self.bias, stride=self.patch_size)
        if self.mode == "serial_conv2d":
            return self._forward_serial_conv2d(x)
        return self._forward_patchify_linear(x)

    def _forward_serial_conv2d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape {tuple(x.shape)}")
        if int(x.shape[0]) <= 1:
            return F.conv2d(x, self.weight, self.bias, stride=self.patch_size)
        return torch.cat(
            [
                F.conv2d(x[index: index + 1], self.weight, self.bias, stride=self.patch_size)
                for index in range(int(x.shape[0]))
            ],
            dim=0,
        )

    def _forward_patchify_linear(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape {tuple(x.shape)}")

        batch, channels, height, width = x.shape
        if channels != self.input_channels:
            raise ValueError(
                f"PatchEmbed2d expected {self.input_channels} channels, got {channels}"
            )

        patch = self.patch_size
        h_patches = height // patch
        w_patches = width // patch
        if h_patches == 0 or w_patches == 0:
            raise ValueError(
                f"PatchEmbed2d input is smaller than patch_size={patch}: got {(height, width)}"
            )

        if height % patch != 0 or width % patch != 0:
            x = x[:, :, : h_patches * patch, : w_patches * patch]
        patches = x.reshape(batch, channels, h_patches, patch, w_patches, patch)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, h_patches, w_patches, -1)
        weight = self.weight.reshape(self.hidden_dim, -1)
        output = torch.matmul(patches, weight.transpose(0, 1))
        output = output.permute(0, 3, 1, 2).contiguous()
        return output + self.bias.view(1, -1, 1, 1)


class IntegratedMultiBasisHead(nn.Module):
    """Optional backbone-integrated basis generator for operator-guided updates."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        condition_dim: int,
        field_channels: int,
        patch_size: int,
        num_basis: int,
        num_heads: int,
        num_blocks: int,
        mlp_ratio: float,
        output_init: str,
        compatibility_mode: str,
        stability_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.field_channels = int(field_channels)
        self.patch_size = int(patch_size)
        self.num_basis = int(num_basis)
        self.output_init = str(output_init).lower()

        if self.num_basis <= 0:
            raise ValueError(f"num_basis must be positive, got {num_basis}")

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    condition_dim=condition_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_clamp=False,
                    compatibility_mode=compatibility_mode,
                    stability_config=stability_config,
                )
                for _ in range(max(1, int(num_blocks)))
            ]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, patch_size * patch_size * field_channels * self.num_basis),
        )
        self._init_output_projection()

    def _init_output_projection(self) -> None:
        head = self.output_proj[-1]
        if self.output_init == "zero":
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            return
        if self.output_init == "usual":
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
            return
        if self.output_init != "default":
            raise ValueError(
                f"Unsupported multibasis output_init={self.output_init!r}. "
                "Expected one of: default, zero, usual."
            )
        nn.init.normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        combined_conditions: torch.Tensor,
        actual_h_patches: int,
        actual_w_patches: int,
    ) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x, combined_conditions)

        x = self.output_proj(x)
        return rearrange(
            x,
            "b (h w) (k p1 p2 c) -> b k c (h p1) (w p2)",
            h=actual_h_patches,
            w=actual_w_patches,
            k=self.num_basis,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.field_channels,
        )


class FSBDiT(BaseBridgeModel):
    """Residual-free DiT backbone for FSB training and inference."""

    def __init__(
        self,
        field_channels: int = 5,
        coord_channels: int = 4,
        wall_layers: Optional[int] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
        circumferential_points: int = 304,
        geometry_dim: int = 27,
        flow_condition_dim: int = 3,
        hidden_dim: int = 640,
        num_layers: int = 12,
        num_heads: int = 8,
        patch_size: int = 2,
        condition_fusion_method: str = "attention",
        use_gradient_checkpointing: bool = True,
        use_torch_compile: bool = False,
        stability_config: Optional[dict] = None,
        patch_embed_config: Optional[Dict[str, Any]] = None,
        geometry_condition: Optional[Dict[str, Any]] = None,
        multibasis_head_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(geometry_dim, flow_condition_dim, coord_channels, field_channels)

        if spatial_shape is not None:
            self.spatial_shape = tuple(int(v) for v in spatial_shape)
            self.wall_layers = wall_layers if wall_layers is not None else self.spatial_shape[0]
        elif wall_layers is not None:
            self.wall_layers = int(wall_layers)
            self.spatial_shape = (self.wall_layers, int(circumferential_points))
        else:
            self.wall_layers = 20
            self.spatial_shape = (20, int(circumferential_points))

        self.field_channels = int(field_channels)
        self.coord_channels = int(coord_channels)
        self.input_channels_backbone = self.field_channels + self.coord_channels
        self.geometry_dim = int(geometry_dim)
        self.flow_condition_dim = int(flow_condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.patch_size = int(patch_size)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_torch_compile = bool(use_torch_compile)

        self.stability_config = dict(stability_config or {})
        self.compatibility_mode = _infer_fsb_dit_compatibility_mode(self.stability_config)
        self.modulation_stability_config = _resolve_modulation_stability_config(
            self.stability_config
        )
        self.adaln_clamp = bool(self.stability_config.get("adaln_clamp", False))
        self.clip_noise_pred = self.stability_config.get("clip_noise_pred", None)
        self.clean_output = bool(self.stability_config.get("clean_output", True))

        self.patch_embed_config = dict(patch_embed_config or {})
        self.patch_embed_mode = str(self.patch_embed_config.get("mode", "conv2d")).lower()
        if self.patch_embed_mode not in {"conv2d", "serial_conv2d", "patchify_linear"}:
            raise ValueError(
                f"Unsupported patch_embed.mode={self.patch_embed_mode!r}. "
                "Expected one of: conv2d, serial_conv2d, patchify_linear."
            )

        self.h_patches = self.spatial_shape[0] // self.patch_size
        self.w_patches = self.spatial_shape[1] // self.patch_size
        self.num_patches = self.h_patches * self.w_patches
        if self.num_patches <= 0:
            raise ValueError(
                f"spatial_shape={self.spatial_shape} is too small for patch_size={self.patch_size}"
            )

        self.geometry_condition_encoder = GeometryConditionEncoder(
            geometry_dim=geometry_dim,
            output_dim=geometry_dim,
            coord_channels=coord_channels,
            config=geometry_condition,
        )
        self.geometry_embed = nn.Linear(geometry_dim, hidden_dim // 2)
        self.flow_embed = nn.Linear(flow_condition_dim, hidden_dim // 2)
        self.condition_fusion = MultimodalConditionFusion(
            {"geometry": hidden_dim // 2, "flow": hidden_dim // 2},
            hidden_dim,
            fusion_method=condition_fusion_method,
        )
        self.time_embed = SinusoidalPositionEmbeddings(hidden_dim)

        self.patch_embed_backbone = PatchEmbed2d(
            input_channels=self.input_channels_backbone,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            mode=self.patch_embed_mode,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)
        self.backbone_blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim,
                    num_heads,
                    hidden_dim,
                    adaln_clamp=self.adaln_clamp,
                    compatibility_mode=self.compatibility_mode,
                    stability_config=self.modulation_stability_config,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, patch_size * patch_size * field_channels),
        )
        self._init_weights()

        self.multibasis_head_config = dict(multibasis_head_config or {})
        self.multibasis_head_enabled = bool(self.multibasis_head_config.get("enabled", False))
        self.multibasis_head_source_layer = str(
            self.multibasis_head_config.get("source_layer", "pre_final")
        )
        if self.multibasis_head_source_layer not in {"patch_embed", "mid_block", "pre_final"}:
            raise ValueError(
                f"Unsupported multibasis_head.source_layer={self.multibasis_head_source_layer!r}. "
                "Expected one of: patch_embed, mid_block, pre_final."
            )
        if self.multibasis_head_enabled:
            self.multibasis_head = IntegratedMultiBasisHead(
                hidden_dim=hidden_dim,
                condition_dim=hidden_dim,
                field_channels=self.field_channels,
                patch_size=patch_size,
                num_basis=int(self.multibasis_head_config.get("num_basis", 4)),
                num_heads=num_heads,
                num_blocks=int(self.multibasis_head_config.get("num_blocks", 1)),
                mlp_ratio=float(self.multibasis_head_config.get("mlp_ratio", 2.0)),
                output_init=str(self.multibasis_head_config.get("output_init", "default")),
                compatibility_mode=self.compatibility_mode,
                stability_config=self.modulation_stability_config,
            )
        else:
            self.multibasis_head = None

        logger.info(
            "FSBDiT initialized: backbone_in=%s (field=%s, coord=%s), "
            "wall_layers=%s, spatial=%s, patches=(%s,%s)=%s, "
            "patch_embed_mode=%s, multibasis=%s",
            self.input_channels_backbone,
            field_channels,
            coord_channels,
            self.wall_layers,
            self.spatial_shape,
            self.h_patches,
            self.w_patches,
            self.num_patches,
            self.patch_embed_mode,
            self.multibasis_head_enabled,
        )

    @staticmethod
    def _normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }

    def _set_compatibility_mode(self, mode: str) -> None:
        if mode not in {COMPAT_MODE_LEGACY_HARD_ADALN, COMPAT_MODE_MODERN_MODULATION}:
            raise ValueError(
                f"Unsupported FSBDiT compatibility mode: {mode!r}. "
                f"Expected one of: {COMPAT_MODE_LEGACY_HARD_ADALN}, {COMPAT_MODE_MODERN_MODULATION}."
            )
        self.compatibility_mode = mode
        for block in self.backbone_blocks:
            block.set_compatibility_mode(mode)
        if self.multibasis_head is not None:
            for block in self.multibasis_head.blocks:
                block.set_compatibility_mode(mode)

    def _detect_compatibility_mode_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> str:
        normalized_state = self._normalize_checkpoint_keys(state_dict)
        has_stability_head = any(
            key.startswith("stability_head.") or ".stability_head." in key
            for key in normalized_state
        )
        has_modulation_clamp = any(
            ".scale_clamp." in key
            or ".shift_clamp." in key
            or ".gate_clamp." in key
            or key.startswith("scale_clamp.")
            or key.startswith("shift_clamp.")
            or key.startswith("gate_clamp.")
            for key in normalized_state
        )
        if has_modulation_clamp:
            return COMPAT_MODE_MODERN_MODULATION
        if has_stability_head:
            return self.compatibility_mode
        if self.adaln_clamp:
            return COMPAT_MODE_LEGACY_HARD_ADALN
        return self.compatibility_mode

    def configure_legacy_compat_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        logger_fn=None,
        context: str = "checkpoint",
    ) -> str:
        target_mode = self._detect_compatibility_mode_from_state_dict(state_dict)
        if target_mode != self.compatibility_mode:
            self._set_compatibility_mode(target_mode)
            if logger_fn is not None:
                logger_fn(
                    f"Switched FSBDiT compatibility mode to "
                    f"{target_mode} while loading {context}"
                )
        return self.compatibility_mode

    def load_state_dict(self, state_dict, strict: bool = True):
        if isinstance(state_dict, dict):
            self.configure_legacy_compat_from_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def _init_weights(self) -> None:
        nn.init.normal_(self.patch_embed_backbone.weight, std=0.02)
        nn.init.constant_(self.patch_embed_backbone.bias, 0)
        nn.init.constant_(self.output_proj[-1].weight, 0)
        nn.init.constant_(self.output_proj[-1].bias, 0)

    def embed_conditions(
        self,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed geometry and flow conditions."""
        flow_conditions = self.preprocess_flow_conditions(flow_conditions)
        geometry_condition = self.geometry_condition_encoder(geometry, coords)
        return self.condition_fusion(
            {
                "geometry": self.geometry_embed(geometry_condition),
                "flow": self.flow_embed(flow_conditions),
            }
        )

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

    @staticmethod
    def _pad_spatial_tensor(tensor: torch.Tensor, pad_w: int, pad_h: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return tensor
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    def _collect_backbone_context(
        self,
        noisy_fields: torch.Tensor,
        timesteps: torch.Tensor,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
    ) -> Dict[str, Any]:
        orig_h, orig_w = noisy_fields.shape[2:]
        pad_h = (self.patch_size - orig_h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - orig_w % self.patch_size) % self.patch_size

        noisy_fields = self._pad_spatial_tensor(noisy_fields, pad_w, pad_h)
        coords = self._pad_spatial_tensor(coords, pad_w, pad_h)

        backbone_input = torch.cat([noisy_fields, coords], dim=1)
        conditions = self.embed_conditions(geometry, flow_conditions, coords=coords)
        combined_conditions = conditions + self.time_embed(timesteps)

        x = self.patch_embed_backbone(backbone_input)
        x = rearrange(x, "b c h w -> b (h w) c")

        actual_h_patches = backbone_input.shape[2] // self.patch_size
        actual_w_patches = backbone_input.shape[3] // self.patch_size
        x = x + self._get_pos_embed_for_patch_grid(actual_h_patches, actual_w_patches)

        return {
            "x": x,
            "combined_conditions": combined_conditions,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "actual_h_patches": actual_h_patches,
            "actual_w_patches": actual_w_patches,
        }

    def _run_backbone_to_layer(
        self,
        x: torch.Tensor,
        combined_conditions: torch.Tensor,
        return_layer: str,
    ) -> torch.Tensor:
        if return_layer == "patch_embed":
            return x

        use_ckpt = self.use_gradient_checkpointing and self.training and torch.is_grad_enabled()
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint

        mid_block_idx = self.num_layers // 2
        for index, block in enumerate(self.backbone_blocks):
            if use_ckpt:
                x = checkpoint(block, x, combined_conditions, use_reentrant=False)
            else:
                x = block(x, combined_conditions)

            if return_layer == "mid_block" and index == mid_block_idx - 1:
                return x

        if return_layer == "pre_final":
            return x

        raise ValueError(
            f"Unknown return_layer: {return_layer}. Expected 'patch_embed', 'mid_block', or 'pre_final'"
        )

    def extract_backbone_tokens(
        self,
        noisy_fields: torch.Tensor,
        timesteps: torch.Tensor,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        return_layer: str = "mid_block",
    ) -> torch.Tensor:
        context = self._collect_backbone_context(
            noisy_fields=noisy_fields,
            timesteps=timesteps,
            geometry=geometry,
            flow_conditions=flow_conditions,
            coords=coords,
        )
        return self._run_backbone_to_layer(
            x=context["x"],
            combined_conditions=context["combined_conditions"],
            return_layer=return_layer,
        )

    def predict_multibasis(
        self,
        noisy_fields: torch.Tensor,
        timesteps: torch.Tensor,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        return_layer: Optional[str] = None,
    ) -> torch.Tensor:
        if not self.multibasis_head_enabled or self.multibasis_head is None:
            raise RuntimeError("multibasis_head is disabled for this FSBDiT")

        context = self._collect_backbone_context(
            noisy_fields=noisy_fields,
            timesteps=timesteps,
            geometry=geometry,
            flow_conditions=flow_conditions,
            coords=coords,
        )
        tokens = self._run_backbone_to_layer(
            x=context["x"],
            combined_conditions=context["combined_conditions"],
            return_layer=return_layer or self.multibasis_head_source_layer,
        )
        basis_fields = self.multibasis_head(
            tokens=tokens,
            combined_conditions=context["combined_conditions"],
            actual_h_patches=context["actual_h_patches"],
            actual_w_patches=context["actual_w_patches"],
        )

        if context["pad_h"] > 0 or context["pad_w"] > 0:
            basis_fields = basis_fields[:, :, :, : context["orig_h"], : context["orig_w"]]
        return basis_fields

    def forward(
        self,
        noisy_fields: torch.Tensor,
        timesteps: torch.Tensor,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for residual-free FSB prediction."""
        if coords is None:
            raise ValueError("coords must be provided")

        context = self._collect_backbone_context(
            noisy_fields=noisy_fields,
            timesteps=timesteps,
            geometry=geometry,
            flow_conditions=flow_conditions,
            coords=coords,
        )

        x = self._run_backbone_to_layer(
            x=context["x"],
            combined_conditions=context["combined_conditions"],
            return_layer="pre_final",
        )
        x = self.output_proj(x)
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=context["actual_h_patches"],
            w=context["actual_w_patches"],
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.field_channels,
        )
        if self.clean_output:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if context["pad_h"] > 0 or context["pad_w"] > 0:
            x = x[:, :, : context["orig_h"], : context["orig_w"]]
        return x

    def extract_features(
        self,
        noisy_fields: torch.Tensor,
        timesteps: torch.Tensor,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        return_layer: str = "mid_block",
    ) -> torch.Tensor:
        """Extract pooled intermediate backbone features."""
        tokens = self.extract_backbone_tokens(
            noisy_fields=noisy_fields,
            timesteps=timesteps,
            geometry=geometry,
            flow_conditions=flow_conditions,
            coords=coords,
            return_layer=return_layer,
        )
        return tokens.mean(dim=1)

    def get_model_info(self) -> dict:
        """Get model information."""
        info = super().get_model_info() if hasattr(super(), "get_model_info") else {}
        info.update(
            {
                "model_type": "FSBDiT",
                "backbone_input_channels": self.input_channels_backbone,
                "field_channels": self.field_channels,
                "coord_channels": self.coord_channels,
                "wall_layers": self.wall_layers,
                "spatial_shape": self.spatial_shape,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_patches": self.num_patches,
                "patch_size": self.patch_size,
                "condition_modalities": 2,
                "multibasis_head_enabled": self.multibasis_head_enabled,
                "multibasis_head_config": self.multibasis_head_config,
                "compatibility_mode": self.compatibility_mode,
                "modulation_stability_config": self.modulation_stability_config,
            }
        )

        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        info.update(
            {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            }
        )
        return info
