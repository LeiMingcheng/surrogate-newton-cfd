"""FNO backbone for the retained flow-state-bridge path."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from surrogate.common import BaseBridgeModel
from surrogate.common.components import AdaLNZero, GeometryConditionEncoder, MultimodalConditionFusion
from surrogate.common.components.layers import ChannelLayerNorm2d, SinusoidalPositionEmbeddings
from surrogate.common.components.spectral_ops import FourierSpectralConv2d as SpectralConv2d


class FSBFourierBlock(nn.Module):
    """DiT-style gated modulation wrapped around an FNO block."""

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


class FSBFNO(BaseBridgeModel):
    """
    FNO backbone for flow-state-bridge prediction.

    Input:
      - noisy_fields x_t
      - coords
      - geometry
      - flow conditions
      - timesteps

    Output:
      - bridge prediction with the same shape as x_t
    """

    def __init__(
        self,
        geometry_dim: int = 27,
        flow_condition_dim: int = 3,
        coord_channels: int = 4,
        out_channels: int = 5,
        modes1: int = 16,
        modes2: int = 24,
        width: int = 128,
        n_layers: int = 6,
        padding: int = 9,
        condition_dim: int = 128,
        time_embed_dim: int = 128,
        condition_fusion_method: str = "attention",
        local_kernel_size: int = 3,
        use_conditioned_coord_bias: bool = True,
        geometry_condition_mode: str = "normal",
        geometry_condition_fixed: Optional[list[float]] = None,
        geometry_condition: Optional[dict] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
        stability: Optional[dict] = None,
    ):
        super().__init__(geometry_dim, flow_condition_dim, coord_channels, out_channels)

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.padding = padding
        self.condition_dim = condition_dim
        self.time_embed_dim = time_embed_dim
        self.condition_fusion_method = condition_fusion_method
        self.local_kernel_size = local_kernel_size
        self.use_conditioned_coord_bias = use_conditioned_coord_bias
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
        self.geometry_condition_encoder = GeometryConditionEncoder(
            geometry_dim=geometry_dim,
            output_dim=geometry_dim,
            coord_channels=coord_channels,
            config=geometry_condition,
            geometry_condition_mode=self.geometry_condition_mode,
            geometry_condition_fixed=self.geometry_condition_fixed,
        )
        self.field_channels = out_channels
        self.input_channels_backbone = out_channels + coord_channels
        self.spatial_shape = spatial_shape
        self.stability_config = dict(stability or {})

        self.time_embed = SinusoidalPositionEmbeddings(time_embed_dim)
        condition_branch_dim = max(condition_dim // 2, 32)
        self.geometry_embed = nn.Linear(geometry_dim, condition_branch_dim)
        self.flow_embed = nn.Linear(flow_condition_dim, condition_branch_dim)
        self.condition_fusion = MultimodalConditionFusion(
            {
                "geometry": condition_branch_dim,
                "flow": condition_branch_dim,
            },
            condition_dim,
            fusion_method=condition_fusion_method,
        )
        self.time_condition_proj = nn.Sequential(
            nn.Linear(time_embed_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.condition_post = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        self.lifting = nn.Conv2d(self.input_channels_backbone, self.width, 1)
        if self.use_conditioned_coord_bias:
            self.coord_bias_proj = nn.Sequential(
                nn.Conv2d(coord_channels, self.width, 1),
                nn.GELU(),
                nn.Conv2d(self.width, self.width, 1),
            )
            self.coord_bias_gate = nn.Sequential(
                nn.Linear(condition_dim, self.width),
                nn.Tanh(),
            )
            nn.init.zeros_(self.coord_bias_gate[0].weight)
            nn.init.zeros_(self.coord_bias_gate[0].bias)
        else:
            self.coord_bias_proj = None
            self.coord_bias_gate = None
        self.fourier_layers = nn.ModuleList([
            FSBFourierBlock(
                self.width,
                self.modes1,
                self.modes2,
                self.condition_dim,
                local_kernel_size=self.local_kernel_size,
                stability_config=self.stability_config,
            )
            for _ in range(self.n_layers)
        ])
        self.projection = nn.Sequential(
            nn.Conv2d(self.width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1),
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
        timesteps: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        geometry_condition = self._prepare_geometry_condition(geometry, coords)
        flow_conditions = self.preprocess_flow_conditions(flow_conditions)
        fused_condition = self.condition_fusion(
            {
                "geometry": self.geometry_embed(geometry_condition),
                "flow": self.flow_embed(flow_conditions),
            }
        )
        time_condition = self.time_condition_proj(self.time_embed(timesteps.float()))
        return self.condition_post(fused_condition + time_condition)

    def forward(
        self,
        noisy_fields: torch.Tensor,
        timesteps: torch.Tensor,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if coords is None:
            raise ValueError("coords must be provided")

        condition_emb = self._embed_conditions(geometry, flow_conditions, timesteps, coords)

        x = torch.cat([noisy_fields, coords], dim=1)
        x = self.lifting(x)
        if self.coord_bias_proj is not None and self.coord_bias_gate is not None:
            coord_bias = self.coord_bias_proj(coords)
            coord_gate = self.coord_bias_gate(condition_emb).unsqueeze(-1).unsqueeze(-1)
            x = x + coord_bias * coord_gate

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.fourier_layers:
            x = layer(x, condition_emb)

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        x = self.projection(x)
        return x
