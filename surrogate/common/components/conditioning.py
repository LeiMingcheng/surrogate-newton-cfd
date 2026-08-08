"""
Unified conditioning mechanisms for model architectures

This module provides standardized conditioning implementations that can be reused
across different model types (UNet, FNO, DiT, etc.) to eliminate code duplication.

Features:
- FiLM (Feature-wise Linear Modulation)
- AdaLN-Zero (Adaptive LayerNorm with Zero initialization)
- Multimodal condition fusion
- Flexible condition embedding strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
from abc import ABC, abstractmethod

from .stability import StabilityClampHead, resolve_stability_clamp_head_config


class BaseConditioning(nn.Module, ABC):
    """Abstract base class for conditioning mechanisms"""

    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.condition_dim = condition_dim
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply conditioning to features"""
        pass


class GeometryConditionEncoder(nn.Module):
    """
    Encode geometry from CST coefficients or a fixed wall-coordinate trace.

    Surrogate-Newton CFD production configurations use ``source="wall_coords"``. CST
    conditioning is retained only for legacy checkpoints and explicit
    ablations because fitted coefficients carry parameterization-dependent
    representation bias.

    ``wall_coords`` defaults to the legacy pointwise Conv1d + pool encoder for
    checkpoint compatibility. A flattened-trace MLP path remains available via
    ``geometry_condition.encoder_type=flat_mlp`` for explicit experiments.
    ``cst`` and ``hybrid`` are kept for backward compatibility with older
    checkpoints and experiments.
    """

    def __init__(
        self,
        geometry_dim: int,
        output_dim: int,
        coord_channels: int = 4,
        config: Optional[Dict[str, Any]] = None,
        geometry_condition_mode: str = "normal",
        geometry_condition_fixed: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        cfg = dict(config or {})
        self.geometry_dim = int(geometry_dim)
        self.output_dim = int(output_dim)
        self.coord_channels = int(coord_channels)
        self.source = str(cfg.get("source", "wall_coords")).lower()
        self.pool = str(cfg.get("pool", "mean")).lower()
        self.wall_row_index = int(cfg.get("wall_row_index", 0))
        self.normalize_wall_coords = bool(cfg.get("normalize_wall_coords", True))
        self.passthrough = bool(cfg.get("passthrough", False))
        self.geometry_condition_mode = str(geometry_condition_mode).lower()
        self.geometry_condition_fixed = geometry_condition_fixed
        self.wall_encoder_type = str(cfg.get("encoder_type", "pointwise_pool")).strip().lower()

        if self.source not in {"cst", "wall_coords", "hybrid"}:
            raise ValueError(
                f"Unsupported geometry_condition.source={self.source!r}. "
                "Expected one of: cst, wall_coords, hybrid."
            )
        if self.wall_encoder_type not in {"flat_mlp", "pointwise_pool"}:
            raise ValueError(
                f"Unsupported geometry_condition.encoder_type={self.wall_encoder_type!r}. "
                "Expected one of: flat_mlp, pointwise_pool."
            )
        if self.wall_encoder_type == "pointwise_pool" and self.pool not in {"mean", "max"}:
            raise ValueError(
                f"Unsupported geometry_condition.pool={self.pool!r}. "
                "Expected one of: mean, max."
            )
        if self.geometry_condition_mode not in {"normal", "fixed", "zero", "tanh"}:
            raise ValueError(
                f"Unsupported geometry_condition_mode={self.geometry_condition_mode!r}. "
                "Expected one of: normal, fixed, zero, tanh."
            )
        if self.passthrough and self.source not in {"cst", "hybrid"}:
            raise ValueError(
                "geometry_condition.passthrough=true is only supported for "
                "geometry_condition.source in {'cst', 'hybrid'}"
            )
        if self.passthrough and self.output_dim != self.geometry_dim:
            raise ValueError(
                "geometry_condition.passthrough=true requires output_dim == geometry_dim. "
                f"Got output_dim={self.output_dim}, geometry_dim={self.geometry_dim}."
            )
        if self.geometry_condition_mode == "fixed" and self.geometry_condition_fixed is None:
            raise ValueError(
                "geometry_condition_fixed must be provided when geometry_condition_mode='fixed'"
            )
        if (
            self.geometry_condition_fixed is not None
            and len(self.geometry_condition_fixed) != self.geometry_dim
        ):
            raise ValueError(
                f"geometry_condition_fixed must have length {self.geometry_dim}, "
                f"got {len(self.geometry_condition_fixed)}"
            )

        hidden_dim = int(cfg.get("hidden_dim", max(64, output_dim)))
        wall_hidden_dim = int(cfg.get("wall_hidden_dim", max(32, output_dim // 2)))

        self.cst_encoder = None
        if self.source in {"cst", "hybrid"} and not self.passthrough:
            self.cst_encoder = nn.Sequential(
                nn.Linear(self.geometry_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.wall_point_encoder = None
        self.wall_mlp_encoder = None
        self.wall_output = None
        if self.source in {"wall_coords", "hybrid"}:
            if self.wall_encoder_type == "flat_mlp":
                self.wall_mlp_encoder = nn.Sequential(
                    nn.LazyLinear(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
            else:
                self.wall_point_encoder = nn.Sequential(
                    nn.Conv1d(2, wall_hidden_dim, kernel_size=1),
                    nn.SiLU(),
                    nn.Conv1d(wall_hidden_dim, wall_hidden_dim, kernel_size=1),
                    nn.SiLU(),
                )
                self.wall_output = nn.Sequential(
                    nn.Linear(wall_hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, output_dim),
                )

        self.hybrid_fusion = None
        if self.source == "hybrid":
            self.hybrid_fusion = nn.Sequential(
                nn.Linear(output_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def _flatten_wall_coords(self, wall_coords: torch.Tensor) -> torch.Tensor:
        # Interleave (x, y) per point so the MLP sees the ordered airfoil trace.
        return wall_coords.transpose(1, 2).reshape(wall_coords.shape[0], -1)

    def _prepare_cst_geometry(self, geometry: torch.Tensor) -> torch.Tensor:
        if self.geometry_condition_mode == "normal":
            return geometry
        if self.geometry_condition_mode == "zero":
            return torch.zeros_like(geometry)
        if self.geometry_condition_mode == "tanh":
            return torch.tanh(geometry)
        fixed = geometry.new_tensor(self.geometry_condition_fixed).unsqueeze(0)
        return fixed.expand(geometry.shape[0], -1)

    def _extract_wall_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if coords is None:
            raise ValueError("coords must be provided when geometry_condition.source uses wall_coords")
        if coords.ndim != 4 or coords.shape[1] < 2:
            raise ValueError(
                f"coords must have shape (B, C>=2, H, W) for wall-coordinate conditioning, "
                f"got {tuple(coords.shape)}"
            )

        row_idx = min(max(self.wall_row_index, 0), coords.shape[2] - 1)
        wall_coords = coords[:, :2, row_idx, :]

        if self.normalize_wall_coords:
            mean = wall_coords.mean(dim=-1, keepdim=True)
            scale = wall_coords.std(dim=-1, keepdim=True).clamp_min(1e-6)
            wall_coords = (wall_coords - mean) / scale

        return wall_coords

    def _encode_wall_coords(self, coords: torch.Tensor) -> torch.Tensor:
        wall_coords = self._extract_wall_coords(coords)
        if self.wall_encoder_type == "flat_mlp":
            return self.wall_mlp_encoder(self._flatten_wall_coords(wall_coords))

        wall_features = self.wall_point_encoder(wall_coords)
        if self.pool == "max":
            pooled = wall_features.max(dim=-1).values
        else:
            pooled = wall_features.mean(dim=-1)
        return self.wall_output(pooled)

    def forward(
        self,
        geometry: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cst_embedding: Optional[torch.Tensor] = None
        if self.source in {"cst", "hybrid"}:
            cst_embedding = self._prepare_cst_geometry(geometry)
            if not self.passthrough:
                cst_embedding = self.cst_encoder(cst_embedding)

        if self.source == "cst":
            return cst_embedding

        wall_embedding = self._encode_wall_coords(coords)
        if self.source == "wall_coords":
            return wall_embedding

        return self.hybrid_fusion(torch.cat([cst_embedding, wall_embedding], dim=1))


class FiLM(BaseConditioning):
    """
    Feature-wise Linear Modulation

    Universal FiLM implementation that replaces 3 current duplicate versions
    in UNet, FNO, and PolicyNetwork modules.

    Formula: FiLM(x, c) = (1 + γ(c)) * x + β(c)
    where γ and β are generated from condition c.

    Args:
        condition_dim: Dimension of condition vector
        feature_dim: Number of feature channels to modulate
        use_layer_norm: Apply LayerNorm before FiLM for stability
        use_gate: Add learnable gate for additional control
        hidden_dim: Hidden dimension in parameter generation network
        activation: Activation function in parameter generation
    """

    def __init__(self,
                 condition_dim: int,
                 feature_dim: int,
                 use_layer_norm: bool = False,
                 use_gate: bool = False,
                 hidden_dim: Optional[int] = None,
                 activation: str = "relu",
                 stability_config: Optional[Dict[str, Any]] = None):
        super().__init__(condition_dim, feature_dim)

        self.use_layer_norm = use_layer_norm
        self.use_gate = use_gate

        # Parameter generation network dimension
        param_dim = feature_dim * 3 if use_gate else feature_dim * 2
        hidden_dim = hidden_dim or max(condition_dim, feature_dim)

        # FiLM parameter generation network
        layers = [
            nn.Linear(condition_dim, hidden_dim),
            self._get_activation(activation)
        ]

        # Optional hidden layer for complex transformations
        if hidden_dim > max(condition_dim, feature_dim) * 2:
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim // 2),
                self._get_activation(activation)
            ])
            hidden_dim = hidden_dim // 2

        layers.append(nn.Linear(hidden_dim, param_dim))

        self.param_generator = nn.Sequential(*layers)

        # Optional layer normalization for stability
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(feature_dim)

        self.gamma_clamp = StabilityClampHead(
            out_channels=feature_dim,
            config=resolve_stability_clamp_head_config(stability_config, role="gamma"),
        )
        self.beta_clamp = StabilityClampHead(
            out_channels=feature_dim,
            config=resolve_stability_clamp_head_config(stability_config, role="beta"),
        )
        self.gate_clamp = None
        if use_gate:
            self.gate_clamp = StabilityClampHead(
                out_channels=feature_dim,
                config=resolve_stability_clamp_head_config(stability_config, role="gate"),
            )

        # Initialize weights for training stability
        nn.init.zeros_(self.param_generator[-1].weight)
        nn.init.zeros_(self.param_generator[-1].bias)

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
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning to features

        Args:
            x: (B, C, ...) input features (can be 2D, 3D, or 4D)
            condition: (B, condition_dim) condition vector

        Returns:
            Conditioned features with same shape as input x
        """
        # Generate FiLM parameters
        params = self.param_generator(condition)  # (B, param_dim)

        if self.use_gate:
            gamma, beta, gate = params.chunk(3, dim=1)  # (B, C) each
        else:
            gamma, beta = params.chunk(2, dim=1)  # (B, C) each
            gate = None

        gamma = self.gamma_clamp(gamma, feature_axis=1)
        beta = self.beta_clamp(beta, feature_axis=1)
        if gate is not None and self.gate_clamp is not None:
            gate = self.gate_clamp(gate, feature_axis=1)

        # Expand parameters to match input dimensions
        if len(x.shape) == 4:  # (B, C, H, W)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            if gate is not None:
                gate = gate.unsqueeze(-1).unsqueeze(-1)
        elif len(x.shape) == 3:  # (B, N, C)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            if gate is not None:
                gate = gate.unsqueeze(1)
        elif len(x.shape) == 2:  # (B, C)
            # No expansion needed
            pass
        else:
            raise ValueError(f"Unsupported input dimension: {len(x.shape)}")

        # Optional layer normalization
        if self.use_layer_norm:
            x = self.layer_norm(x)

        # Apply FiLM transformation
        x = (1 + gamma) * x + beta

        # Optional gating
        if self.use_gate and gate is not None:
            x = x * torch.sigmoid(gate)

        return x


class AdaLNZero(BaseConditioning):
    """
    Adaptive LayerNorm with Zero initialization

    Used in DiT (Diffusion Transformer) blocks for conditioning.
    Generates scale, shift, and gate parameters for both attention and MLP components.

    Formula:
        y = gate * (LN(x) * (1 + scale) + shift)

    Args:
        condition_dim: Dimension of condition vector
        hidden_dim: Hidden dimension of the features being conditioned
        use_bias: Whether to use bias in parameter generation
    """

    def __init__(self,
                 condition_dim: int,
                 hidden_dim: int,
                 use_bias: bool = True,
                 stability_config: Optional[Dict[str, Any]] = None,
                 enable_stability_clamp: bool = True):
        super().__init__(condition_dim, hidden_dim)

        self.hidden_dim = hidden_dim

        # Parameter generation network
        # Generates 6 parameters: scale, shift, gate for attention and MLP
        self.param_generator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_dim * 6, bias=use_bias)
        )

        # Zero initialization for training stability
        nn.init.zeros_(self.param_generator[-1].weight)
        if use_bias:
            nn.init.zeros_(self.param_generator[-1].bias)

        self.scale_clamp = None
        self.shift_clamp = None
        self.gate_clamp = None
        self.use_stability_clamp = bool(enable_stability_clamp)
        if stability_config is not None:
            self.scale_clamp = StabilityClampHead(
                out_channels=hidden_dim,
                config=resolve_stability_clamp_head_config(stability_config, role="scale"),
            )
            self.shift_clamp = StabilityClampHead(
                out_channels=hidden_dim,
                config=resolve_stability_clamp_head_config(stability_config, role="shift"),
            )
            self.gate_clamp = StabilityClampHead(
                out_channels=hidden_dim,
                config=resolve_stability_clamp_head_config(stability_config, role="gate"),
            )

    def set_use_stability_clamp(self, enabled: bool) -> None:
        """Toggle shared modulation clamps without rebuilding the module."""
        self.use_stability_clamp = bool(enabled)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Generate AdaLN-Zero parameters

        Args:
            x: (B, N, hidden_dim) input features
            condition: (B, condition_dim) condition vector

        Returns:
            Tuple of 6 parameter tensors:
            (scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp)
        """
        # Generate parameters from condition
        params = self.param_generator(condition)  # (B, hidden_dim * 6)

        # Reshape for broadcasting
        # Split into 6 parameter sets
        scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = params.chunk(6, dim=1)

        if self.use_stability_clamp and self.scale_clamp is not None:
            scale_attn = self.scale_clamp(scale_attn, feature_axis=1)
            scale_mlp = self.scale_clamp(scale_mlp, feature_axis=1)
        if self.use_stability_clamp and self.shift_clamp is not None:
            shift_attn = self.shift_clamp(shift_attn, feature_axis=1)
            shift_mlp = self.shift_clamp(shift_mlp, feature_axis=1)
        if self.use_stability_clamp and self.gate_clamp is not None:
            gate_attn = self.gate_clamp(gate_attn, feature_axis=1)
            gate_mlp = self.gate_clamp(gate_mlp, feature_axis=1)

        scale_attn = scale_attn.unsqueeze(1)
        shift_attn = shift_attn.unsqueeze(1)
        gate_attn = gate_attn.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)

        return scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp


class MultimodalConditionFusion(nn.Module):
    """
    Multimodal condition fusion mechanism

    Combines different types of conditions (geometry, flow, PDE residual, etc.)
    into a unified representation that can be used for model conditioning.

    Args:
        condition_dims: Dictionary mapping condition names to their dimensions
        fusion_dim: Output dimension for fused condition
        fusion_method: Method for combining conditions
            - "concat": Simple concatenation followed by linear projection
            - "attention": Self-attention fusion
            - "weighted": Learnable weighted sum
    """

    def __init__(self,
                 condition_dims: Dict[str, int],
                 fusion_dim: int,
                 fusion_method: str = "attention",
                 num_heads: int = 8):
        super().__init__()

        self.condition_dims = condition_dims
        self.fusion_dim = fusion_dim
        self.fusion_method = fusion_method
        self.condition_names = list(condition_dims.keys())

        # Individual condition embedders
        self.embedders = nn.ModuleDict()
        embed_dim_per_condition = fusion_dim // len(condition_dims)

        for name, dim in condition_dims.items():
            self.embedders[name] = nn.Sequential(
                nn.Linear(dim, embed_dim_per_condition),
                nn.SiLU(),
                nn.Linear(embed_dim_per_condition, embed_dim_per_condition)
            )

        # Fusion mechanism
        if fusion_method == "attention":
            # MHA embed_dim must match input last dimension (embed_dim_per_condition)
            # Ensure num_heads is valid: at least 1, and divides embed_dim_per_condition
            adjusted_num_heads = max(1, min(num_heads, embed_dim_per_condition // 64))
            # Ensure embed_dim_per_condition is divisible by num_heads
            while embed_dim_per_condition % adjusted_num_heads != 0 and adjusted_num_heads > 1:
                adjusted_num_heads -= 1

            self.fusion = nn.MultiheadAttention(
                embed_dim=embed_dim_per_condition,
                num_heads=adjusted_num_heads,
                batch_first=True,
                dropout=0.0
            )
            # Output projection to fusion_dim
            self.output_proj_attn = nn.Linear(embed_dim_per_condition, fusion_dim)
        elif fusion_method == "concat":
            total_embed_dim = embed_dim_per_condition * len(condition_dims)
            self.fusion = nn.Sequential(
                nn.Linear(total_embed_dim, fusion_dim),
                nn.SiLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
        elif fusion_method == "weighted":
            self.condition_weights = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(1.0))
                for name in condition_dims.keys()
            })
            self.output_proj = nn.Linear(embed_dim_per_condition, fusion_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def forward(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple condition tensors

        Args:
            conditions: Dictionary mapping condition names to tensors

        Returns:
            Fused condition tensor of shape (B, fusion_dim)
        """
        if not conditions:
            return torch.zeros(1, self.fusion_dim, device=next(iter(self.parameters())).device)

        batch_size = len(next(iter(conditions.values())))

        # Embed each condition
        embedded_conditions = []
        for name in self.condition_names:
            if name in conditions and conditions[name] is not None:
                cond = conditions[name]
                if cond.dim() == 1:
                    cond = cond.unsqueeze(0).expand(batch_size, -1)
                embedded = self.embedders[name](cond)
                embedded_conditions.append(embedded)
            else:
                # Missing condition - use zeros
                embed_dim = self.fusion_dim // len(self.condition_names)
                embedded_conditions.append(torch.zeros(
                    batch_size, embed_dim,
                    device=next(iter(self.parameters())).device
                ))

        if not embedded_conditions:
            return torch.zeros(batch_size, self.fusion_dim,
                             device=next(iter(self.parameters())).device)

        # Fusion
        if self.fusion_method == "attention":
            # Stack for attention
            fused = torch.stack(embedded_conditions, dim=1)  # (B, num_conditions, embed_dim_per_condition)

            # Apply self-attention
            fused, _ = self.fusion(fused, fused, fused)  # (B, num_conditions, embed_dim_per_condition)

            # Global pooling
            fused = fused.mean(dim=1)  # (B, embed_dim_per_condition)

            # Project to fusion_dim
            fused = self.output_proj_attn(fused)  # (B, fusion_dim)

        elif self.fusion_method == "concat":
            # Concatenate and project
            fused = torch.cat(embedded_conditions, dim=-1)  # (B, total_embed_dim)
            fused = self.fusion(fused)  # (B, fusion_dim)

        elif self.fusion_method == "weighted":
            # Weighted sum
            weighted_sum = None
            for i, (name, embedded) in enumerate(zip(self.condition_names, embedded_conditions)):
                weight = torch.sigmoid(self.condition_weights[name])
                if i == 0:
                    weighted_sum = weight * embedded
                else:
                    weighted_sum += weight * embedded

            fused = self.output_proj(weighted_sum)  # (B, fusion_dim)

        return fused


class ConditionalWrapper(nn.Module):
    """
    Wrapper to add conditioning to any module

    Allows applying FiLM conditioning to existing modules without modification.
    """

    def __init__(self,
                 module: nn.Module,
                 condition_dim: int,
                 feature_dim: int,
                 conditioning_type: str = "film",
                 **conditioning_kwargs):
        super().__init__()

        self.module = module
        self.conditioning_type = conditioning_type

        # Create conditioning layer
        if conditioning_type == "film":
            self.conditioning = FiLM(condition_dim, feature_dim, **conditioning_kwargs)
        elif conditioning_type == "adalin":
            self.conditioning = AdaLNZero(condition_dim, feature_dim, **conditioning_kwargs)
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning_type}")

    def forward(self, x: torch.Tensor, condition: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with conditioning"""
        x = self.module(x, *args, **kwargs)
        x = self.conditioning(x, condition)
        return x
