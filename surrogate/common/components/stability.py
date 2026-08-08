"""Shared stability components for modulation clamps and checkpoint loading."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_CLAMP_HEAD_CONFIG = {
    "enabled": True,
    "clean_output": True,
    "clamp_mode": "tanh",
    "learnable_scale": True,
    "per_channel": True,
    "init_scale": 12.0,
    "min_scale": 1.0e-3,
}

_DEFAULT_ROLE_INIT_SCALES = {
    "gamma": 2.0,
    "beta": 5.0,
    "scale": 2.0,
    "shift": 5.0,
    "gate": 1.0,
}


def _is_stability_head_key(key: str) -> bool:
    return key.startswith("stability_head.") or ".stability_head." in key


def _is_geometry_condition_key(key: str) -> bool:
    return (
        key.startswith("geometry_condition_encoder.")
        or ".geometry_condition_encoder." in key
    )


def _is_modulation_clamp_key(key: str) -> bool:
    modulation_tokens = (
        ".gamma_clamp.",
        ".beta_clamp.",
        ".gate_clamp.",
        ".scale_clamp.",
        ".shift_clamp.",
    )
    return (
        key.startswith("gamma_clamp.")
        or key.startswith("beta_clamp.")
        or key.startswith("gate_clamp.")
        or key.startswith("scale_clamp.")
        or key.startswith("shift_clamp.")
        or any(token in key for token in modulation_tokens)
    )


def _is_multibasis_head_key(key: str) -> bool:
    return key.startswith("multibasis_head.") or ".multibasis_head." in key


def _is_compat_optional_key(key: str) -> bool:
    return (
        _is_stability_head_key(key)
        or _is_geometry_condition_key(key)
        or _is_modulation_clamp_key(key)
        or _is_multibasis_head_key(key)
    )


def _log_message(logger: Optional[Callable[[str], None]], message: str) -> None:
    if logger is not None:
        logger(message)


def load_state_dict_with_stability_head_compat(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    logger: Optional[Callable[[str], None]] = None,
    context: str = "checkpoint",
) -> tuple[list[str], list[str]]:
    """Load state dict while tolerating optional compatibility modules."""
    normalized_state = OrderedDict()
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized_state[normalized_key] = value

    target_module = module.module if hasattr(module, "module") else module
    compat_hook = getattr(target_module, "configure_legacy_compat_from_state_dict", None)
    if callable(compat_hook):
        compat_hook(
            normalized_state,
            logger_fn=logger,
            context=context,
        )

    missing_keys, unexpected_keys = module.load_state_dict(normalized_state, strict=False)

    bad_missing = [key for key in missing_keys if not _is_compat_optional_key(key)]
    bad_unexpected = [key for key in unexpected_keys if not _is_compat_optional_key(key)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"State-dict mismatch while loading {context}. "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )

    allowed_missing = [key for key in missing_keys if _is_compat_optional_key(key)]
    allowed_unexpected = [key for key in unexpected_keys if _is_compat_optional_key(key)]
    if allowed_missing:
        _log_message(
            logger,
            f"Missing optional compatibility keys while loading {context}; using module defaults: {allowed_missing}",
        )
    if allowed_unexpected:
        _log_message(
            logger,
            f"Ignoring unexpected optional compatibility keys while loading {context}: {allowed_unexpected}",
        )

    return list(missing_keys), list(unexpected_keys)


class StabilityClampHead(nn.Module):
    """Learnable soft clamp for modulation parameters or other feature tensors."""

    def __init__(
        self,
        out_channels: int,
        config: Optional[Mapping[str, object]] = None,
    ):
        super().__init__()

        cfg = dict(_DEFAULT_CLAMP_HEAD_CONFIG)
        if config is not None:
            cfg.update(dict(config))

        self.enabled = bool(cfg.get("enabled", True))
        self.clean_output = bool(cfg.get("clean_output", True))
        self.clamp_mode = str(cfg.get("clamp_mode", "tanh")).lower()
        self.learnable_scale = bool(cfg.get("learnable_scale", True))
        self.register_scale_parameter = bool(cfg.get("register_scale_parameter", True))
        self.per_channel = bool(cfg.get("per_channel", True))
        self.min_scale = float(cfg.get("min_scale", 1.0e-3))
        init_scale = float(cfg.get("init_scale", 12.0))

        if init_scale <= 0.0:
            raise ValueError("stability clamp init_scale must be positive")
        if self.min_scale <= 0.0:
            raise ValueError("stability clamp min_scale must be positive")
        if self.clamp_mode not in {"tanh", "softsign", "identity"}:
            raise ValueError(
                f"Unsupported clamp_mode={self.clamp_mode!r}. "
                "Expected one of: tanh, softsign, identity."
            )

        scale_shape = (out_channels,) if self.per_channel else (1,)
        init_unconstrained = torch.full(scale_shape, self._inverse_softplus(init_scale - self.min_scale))
        if not self.enabled:
            self.register_parameter("log_scale", None)
            self.register_buffer("_fixed_log_scale", None, persistent=False)
        elif self.register_scale_parameter:
            self.log_scale = nn.Parameter(
                init_unconstrained,
                requires_grad=self.learnable_scale,
            )
            self.register_buffer("_fixed_log_scale", None, persistent=False)
        else:
            self.register_parameter("log_scale", None)
            self.register_buffer("_fixed_log_scale", init_unconstrained, persistent=False)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor)).item()

    def _scale_source(self) -> Optional[torch.Tensor]:
        if self.log_scale is not None:
            return self.log_scale
        return self._fixed_log_scale

    def _get_scale(self, x: torch.Tensor, feature_axis: int) -> torch.Tensor:
        scale_source = self._scale_source()
        if scale_source is None:
            raise RuntimeError("StabilityClampHead scale requested while clamp is disabled")
        scale = F.softplus(scale_source) + self.min_scale
        feature_axis = feature_axis if feature_axis >= 0 else x.dim() + feature_axis
        if feature_axis < 0 or feature_axis >= x.dim():
            raise ValueError(
                f"feature_axis must index a valid dimension for input shape {tuple(x.shape)}, "
                f"got {feature_axis}"
            )
        if self.per_channel:
            view_shape = [1] * x.dim()
            view_shape[feature_axis] = scale.shape[0]
            return scale.to(device=x.device, dtype=x.dtype).view(*view_shape)
        return scale.to(device=x.device, dtype=x.dtype).view(*([1] * x.dim()))

    def _apply_clamp(self, x: torch.Tensor, feature_axis: int) -> torch.Tensor:
        if self.clamp_mode == "identity":
            return x
        scale = self._get_scale(x, feature_axis=feature_axis)
        if self.clamp_mode == "tanh":
            return torch.tanh(x / scale) * scale
        return scale * (x / (scale + torch.abs(x)))

    def forward(self, x: torch.Tensor, feature_axis: int = 1) -> torch.Tensor:
        if self.clean_output:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.enabled:
            return x
        x = self._apply_clamp(x, feature_axis=feature_axis)
        if self.clean_output:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def get_config(self) -> dict:
        scale_source = self._scale_source()
        scale_value: object
        if scale_source is None:
            scale_value = None
        else:
            scale = F.softplus(scale_source.detach()) + self.min_scale
            if self.per_channel:
                scale_value = [float(v) for v in scale.cpu().tolist()]
            else:
                scale_value = float(scale.item())
        return {
            "enabled": self.enabled,
            "clean_output": self.clean_output,
            "clamp_mode": self.clamp_mode,
            "learnable_scale": self.learnable_scale,
            "register_scale_parameter": self.register_scale_parameter,
            "per_channel": self.per_channel,
            "min_scale": self.min_scale,
            "current_scale": scale_value,
        }


def resolve_stability_clamp_head_config(
    stability_config: Optional[Mapping[str, object]],
    *,
    role: Optional[str] = None,
) -> dict:
    """Resolve clamp config from a model stability dict."""
    if stability_config is None:
        config = dict(_DEFAULT_CLAMP_HEAD_CONFIG)
        if role in _DEFAULT_ROLE_INIT_SCALES:
            config["init_scale"] = _DEFAULT_ROLE_INIT_SCALES[role]
        return config

    stability_dict = dict(stability_config)
    nested = stability_dict.get("clamp_head")
    source_dict = dict(nested) if isinstance(nested, Mapping) else stability_dict

    config = dict(_DEFAULT_CLAMP_HEAD_CONFIG)
    explicit_init_scale = False
    for key in _DEFAULT_CLAMP_HEAD_CONFIG:
        if key in source_dict:
            config[key] = source_dict[key]
            if key == "init_scale":
                explicit_init_scale = True
    if "register_scale_parameter" in source_dict:
        config["register_scale_parameter"] = source_dict["register_scale_parameter"]

    if role and not explicit_init_scale and role in _DEFAULT_ROLE_INIT_SCALES:
        config["init_scale"] = _DEFAULT_ROLE_INIT_SCALES[role]

    if role:
        role_cfg = source_dict.get(role)
        if isinstance(role_cfg, Mapping):
            config.update(dict(role_cfg))

    return config
