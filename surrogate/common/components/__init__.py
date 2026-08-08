"""Canonical shared components used by direct and FSB model architectures."""

from .conditioning import (
    AdaLNZero,
    MultimodalConditionFusion,
    GeometryConditionEncoder,
)

from .convolution import (
    ResBlock,
)

from .dit import (
    COMPAT_MODE_LEGACY_HARD_ADALN,
    COMPAT_MODE_MODERN_MODULATION,
    DiTBlock,
    infer_dit_compatibility_mode,
    resolve_modulation_stability_config,
)

from .layers import (
    ChannelLayerNorm2d,
    SinusoidalPositionEmbeddings,
)

from .spectral_ops import (
    FourierSpectralConv2d,
)

from .stability import (
    StabilityClampHead,
    load_state_dict_with_stability_head_compat,
    resolve_stability_clamp_head_config,
)

__all__ = [
    # Conditioning
    "AdaLNZero",
    "MultimodalConditionFusion",
    "GeometryConditionEncoder",

    # Convolution
    "ResBlock",

    # DiT blocks
    "COMPAT_MODE_LEGACY_HARD_ADALN",
    "COMPAT_MODE_MODERN_MODULATION",
    "DiTBlock",
    "infer_dit_compatibility_mode",
    "resolve_modulation_stability_config",

    # Small shared layers
    "ChannelLayerNorm2d",
    "SinusoidalPositionEmbeddings",

    # Spectral operations
    "FourierSpectralConv2d",

    # Stability
    "StabilityClampHead",
    "load_state_dict_with_stability_head_compat",
    "resolve_stability_clamp_head_config",
]
