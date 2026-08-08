"""
FNO-based Multi-Field Surrogate Model

Fourier Neural Operator for flow field prediction
- Global receptive field (frequency domain convolution)
- Resolution independent (generalizes to different grids)
- Parameter efficient (frequency domain truncation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from surrogate.common import BaseDirectModel
from surrogate.common.components.conditioning import (
    AdaLNZero,
    FiLM,
    GeometryConditionEncoder,
    MultimodalConditionFusion,
)
from surrogate.common.components.convolution import ResBlock
from surrogate.common.components.layers import ChannelLayerNorm2d
from surrogate.common.components.spectral_ops import FourierSpectralConv2d as SpectralConv2d


class ConditionedFourierBlock(nn.Module):
    """DiT-style gated modulation around spectral, local, and channel-mixing updates."""

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        condition_dim: int,
        local_kernel_size: int = 3,
        stability_config: Optional[dict] = None,
    ):
        super().__init__()
        self.norm1 = ChannelLayerNorm2d(width)
        self.norm2 = ChannelLayerNorm2d(width)
        self.modulation = AdaLNZero(
            condition_dim,
            width,
            stability_config=stability_config,
            enable_stability_clamp=True,
        )
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.skip_conv = nn.Conv2d(width, width, 1)
        padding = max(local_kernel_size // 2, 0)
        self.local_conv = nn.Conv2d(width, width, local_kernel_size, padding=padding)
        self.mlp = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 1),
        )

    @staticmethod
    def _to_spatial(param: torch.Tensor) -> torch.Tensor:
        return param[:, 0, :].unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        dummy_tokens = x.new_zeros((x.shape[0], 1, x.shape[1]))
        scale_mix, shift_mix, gate_mix, scale_mlp, shift_mlp, gate_mlp = self.modulation(
            dummy_tokens, condition
        )

        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + self._to_spatial(scale_mix)) + self._to_spatial(shift_mix)
        mixed = self.spectral_conv(x_mod) + self.skip_conv(x_mod) + self.local_conv(x_mod)
        x = x + self._to_spatial(gate_mix) * mixed

        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + self._to_spatial(scale_mlp)) + self._to_spatial(shift_mlp)
        x = x + self._to_spatial(gate_mlp) * self.mlp(x_mod)
        return x


class FourierLayer(nn.Module):
    """
    Single Fourier layer

    Structure: SpectralConv + Skip connection + FiLM modulation + Activation
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        condition_dim: int,
        stability_config: Optional[dict] = None,
    ):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.skip_conv = nn.Conv2d(width, width, 1)  # Skip connection 1x1 conv
        self.film = FiLM(condition_dim, width, stability_config=stability_config)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, width, H, W)
            condition: (batch, condition_dim)

        Returns:
            output: (batch, width, H, W)
        """
        # Fourier branch
        x1 = self.spectral_conv(x)

        # Skip connection branch
        x2 = self.skip_conv(x)

        # Addition
        out = x1 + x2

        # FiLM modulation
        out = self.film(out, condition)

        # Activation
        out = F.gelu(out)

        return out


class DirectFNO(BaseDirectModel):
    """
    FNO-based multi-field surrogate model

    Architecture:
        - Lifting layer: coords (2 channels) → hidden (width channels)
        - N Fourier layers: frequency domain convolution + skip + condition modulation
        - Projection layer: hidden → output (5 channels)
    """

    def __init__(
        self,
        geometry_dim: int = 27,
        flow_condition_dim: int = 3,
        coord_channels: int = 4,
        out_channels: int = 5,  # Default 5 channels [ρ, u, v, p, ν~]
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        n_layers: int = 4,
        padding: int = 9,
        condition_dim: int = 128,
        condition_fusion_method: str = "attention",
        local_kernel_size: int = 3,
        use_conditioned_coord_bias: bool = False,
        backbone_variant: str = "legacy",
        geometry_condition_mode: str = "normal",
        geometry_condition_fixed: Optional[list[float]] = None,
        geometry_condition: Optional[dict] = None,
        use_initial_field: bool = False,
        stability: Optional[dict] = None,
    ):
        """
        Args:
            geometry_dim: Geometry parameter dimension
            flow_condition_dim: Flow condition dimension
            coord_channels: Number of coordinate channels (default: 4, [x, y, i_norm, j_norm])
            out_channels: Number of output channels
            modes1: Number of frequency modes (x direction)
            modes2: Number of frequency modes (y direction)
            width: Hidden layer width
            n_layers: Number of Fourier layers
            padding: Padding size (for boundary handling)
        """
        super().__init__(geometry_dim, flow_condition_dim, coord_channels, out_channels)

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.padding = padding
        self.condition_dim = int(condition_dim)
        self.condition_fusion_method = str(condition_fusion_method)
        self.local_kernel_size = int(local_kernel_size)
        self.use_conditioned_coord_bias = bool(use_conditioned_coord_bias)
        self.use_initial_field = bool(use_initial_field)
        self.backbone_variant = str(backbone_variant).lower()
        if self.backbone_variant not in {"legacy", "modern"}:
            raise ValueError(
                f"Unsupported backbone_variant={backbone_variant!r}. "
                "Expected one of: legacy, modern."
            )
        self.geometry_condition_mode = str(geometry_condition_mode).lower()
        if self.geometry_condition_mode not in {"normal", "fixed", "zero", "tanh"}:
            raise ValueError(
                f"Unsupported geometry_condition_mode={geometry_condition_mode!r}. "
                "Expected one of: normal, fixed, zero, tanh."
            )
        if geometry_condition_fixed is not None and len(geometry_condition_fixed) != geometry_dim:
            raise ValueError(
                f"geometry_condition_fixed must have length {geometry_dim}, "
                f"got {len(geometry_condition_fixed)}"
            )
        if self.geometry_condition_mode == "fixed" and geometry_condition_fixed is None:
            raise ValueError("geometry_condition_fixed must be provided when geometry_condition_mode='fixed'")
        self.geometry_condition_fixed = (
            tuple(float(value) for value in geometry_condition_fixed)
            if geometry_condition_fixed is not None else None
        )
        self.stability_config = dict(stability or {})
        self.geometry_condition_encoder = GeometryConditionEncoder(
            geometry_dim=geometry_dim,
            output_dim=geometry_dim,
            coord_channels=coord_channels,
            config=geometry_condition,
            geometry_condition_mode=self.geometry_condition_mode,
            geometry_condition_fixed=self.geometry_condition_fixed,
        )

        # Lifting layer: lift coordinates to hidden space
        self.lifting = nn.Conv2d(coord_channels, self.width, 1)
        self.condition_encoder: Optional[nn.Module] = None
        self.geometry_embed: Optional[nn.Module] = None
        self.flow_embed: Optional[nn.Module] = None
        self.condition_fusion: Optional[nn.Module] = None
        self.condition_post: Optional[nn.Module] = None
        self.coord_bias_proj: Optional[nn.Module] = None
        self.coord_bias_gate: Optional[nn.Module] = None
        self.initial_field_lifting: Optional[nn.Module] = None

        if self.use_initial_field:
            self.initial_field_lifting = nn.Sequential(
                nn.Conv2d(out_channels, self.width, 1),
                nn.GELU(),
                nn.Conv2d(self.width, self.width, 1),
            )

        if self.backbone_variant == "legacy":
            legacy_input_dim = geometry_dim + flow_condition_dim
            self.condition_encoder = nn.Sequential(
                nn.Linear(legacy_input_dim, self.condition_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.condition_dim, self.condition_dim),
                nn.ReLU(inplace=True),
            )
            self.fourier_layers = nn.ModuleList([
                FourierLayer(
                    self.width,
                    self.modes1,
                    self.modes2,
                    self.condition_dim,
                    stability_config=self.stability_config,
                )
                for _ in range(self.n_layers)
            ])
        else:
            condition_branch_dim = max(self.condition_dim // 2, 32)
            self.geometry_embed = nn.Linear(geometry_dim, condition_branch_dim)
            self.flow_embed = nn.Linear(flow_condition_dim, condition_branch_dim)
            self.condition_fusion = MultimodalConditionFusion(
                {
                    "geometry": condition_branch_dim,
                    "flow": condition_branch_dim,
                },
                self.condition_dim,
                fusion_method=self.condition_fusion_method,
            )
            self.condition_post = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.condition_dim, self.condition_dim),
            )
            if self.use_conditioned_coord_bias:
                self.coord_bias_proj = nn.Sequential(
                    nn.Conv2d(coord_channels, self.width, 1),
                    nn.GELU(),
                    nn.Conv2d(self.width, self.width, 1),
                )
                self.coord_bias_gate = nn.Sequential(
                    nn.Linear(self.condition_dim, self.width),
                    nn.Tanh(),
                )
                nn.init.zeros_(self.coord_bias_gate[0].weight)
                nn.init.zeros_(self.coord_bias_gate[0].bias)
            self.fourier_layers = nn.ModuleList([
                ConditionedFourierBlock(
                    self.width,
                    self.modes1,
                    self.modes2,
                    self.condition_dim,
                    local_kernel_size=self.local_kernel_size,
                    stability_config=self.stability_config,
                )
                for _ in range(self.n_layers)
            ])

        # Projection layer: hidden space → output
        self.projection = nn.Sequential(
            nn.Conv2d(self.width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def _prepare_geometry_condition(
        self,
        geometry: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.geometry_condition_encoder(geometry, coords)

    def _embed_conditions(
        self,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        flow_conditions = self.preprocess_flow_conditions(flow_conditions)
        geometry_condition = self._prepare_geometry_condition(geometry, coords)

        if self.backbone_variant == "legacy":
            if self.condition_encoder is None:
                raise RuntimeError("Legacy condition_encoder is not initialized")
            condition = torch.cat([geometry_condition, flow_conditions], dim=1)
            return self.condition_encoder(condition)

        if self.geometry_embed is None or self.flow_embed is None:
            raise RuntimeError("Modern geometry/flow condition branches are not initialized")
        if self.condition_fusion is None or self.condition_post is None:
            raise RuntimeError("Modern condition fusion modules are not initialized")

        fused_condition = self.condition_fusion(
            {
                "geometry": self.geometry_embed(geometry_condition),
                "flow": self.flow_embed(flow_conditions),
            }
        )
        return self.condition_post(fused_condition)

    def forward(
        self,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        initial_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            geometry: (batch, 27) - Enhanced CST parameters
            flow_conditions: (batch, 3) - [Ma, AoA, Re]
            coords: (batch, 4, H, W) - [x, y, i_norm, j_norm] coordinates

        Returns:
            fields: (batch, 5, H, W) - [ρ, u, v, p, ν~]
        """
        condition_emb = self._embed_conditions(geometry, flow_conditions, coords)
        initial_field = self.resolve_initial_field(flow_conditions, coords, initial_field)

        # Lifting: coords → hidden
        x = self.lifting(coords)  # (batch, width, H, W)
        if self.initial_field_lifting is not None:
            if initial_field is None:
                raise ValueError("DirectFNO requires initial_field when use_initial_field=True")
            x = x + self.initial_field_lifting(initial_field)
        if self.coord_bias_proj is not None and self.coord_bias_gate is not None:
            coord_bias = self.coord_bias_proj(coords)
            coord_gate = self.coord_bias_gate(condition_emb).unsqueeze(-1).unsqueeze(-1)
            x = x + coord_bias * coord_gate

        # Padding (handle boundary effects)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # Fourier layers
        for layer in self.fourier_layers:
            x = layer(x, condition_emb)

        # Remove padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        # Projection: hidden → output
        fields = self.projection(x)  # (batch, out_channels, H, W)

        return fields
