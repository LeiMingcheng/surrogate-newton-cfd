"""Field normalization helpers for surrogate data pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class FieldNormalizer(nn.Module):
    """Flow-field normalizer with optional turbulent-channel scaling and z-score."""

    def __init__(
        self,
        stats_path: str | Path,
        scale_turbulent: bool = False,
        normalize: bool = False,
        turbulent_channel_idx: int = 4,
        eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        self.scale_turbulent = bool(scale_turbulent)
        self.normalize = bool(normalize)
        self.turbulent_channel_idx = int(turbulent_channel_idx)
        self.eps = float(eps)

        if self.normalize:
            self.scale_turbulent = True

        self._load_stats(stats_path)

        if self.scale_turbulent:
            self.register_buffer("k", torch.tensor(self.k_value, dtype=torch.float32))
        if self.normalize:
            self.register_buffer("means", self.means_tensor)
            self.register_buffer("stds", self.stds_tensor)

    def _load_stats(self, stats_path: str | Path) -> None:
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Field statistics file not found: {stats_path}")

        with open(stats_path, "r", encoding="utf-8") as file:
            stats = json.load(file)

        version = stats.get("version", "unknown")
        if not str(version).startswith("turbulent-scale"):
            raise ValueError(f"Unsupported field statistics version: {version}")

        turb_stats = stats["turbulent_channel"]
        if int(turb_stats["index"]) != self.turbulent_channel_idx:
            raise ValueError(
                "Turbulent-channel index mismatch: "
                f"stats={turb_stats['index']}, requested={self.turbulent_channel_idx}"
            )

        self.k_value = float(turb_stats["scale_factor"])
        self.original_mean = turb_stats["original_mean"]
        self.original_std = turb_stats["original_std"]
        self.log_mean = turb_stats["log_mean"]
        self.log_std = turb_stats["log_std"]

        if self.normalize:
            other_means = stats["other_channels"]["means"]
            other_stds = stats["other_channels"]["stds"]
            means_list = other_means + [self.log_mean]
            stds_list = other_stds + [self.log_std]
            self.means_tensor = torch.tensor(means_list, dtype=torch.float32)
            self.stds_tensor = torch.clamp(
                torch.tensor(stds_list, dtype=torch.float32),
                min=self.eps,
            )

    def transform(self, fields: torch.Tensor) -> torch.Tensor:
        """Transform physical fields into model space."""
        if not self.scale_turbulent and not self.normalize:
            return fields

        fields = fields.clone()
        if self.scale_turbulent:
            nu_tilde = fields[:, self.turbulent_channel_idx : self.turbulent_channel_idx + 1]
            k = self.k.to(fields.device) if hasattr(self, "k") else self.k_value
            fields[:, self.turbulent_channel_idx : self.turbulent_channel_idx + 1] = torch.log1p(k * nu_tilde)

        if self.normalize:
            means = self.means.to(fields.device).view(1, -1, 1, 1)
            stds = self.stds.to(fields.device).view(1, -1, 1, 1)
            fields = (fields - means) / stds
        return fields

    def inverse_transform(self, fields: torch.Tensor) -> torch.Tensor:
        """Transform model-space fields back into physical fields."""
        if not self.scale_turbulent and not self.normalize:
            return fields

        fields = fields.clone()
        if self.normalize:
            means = self.means.to(fields.device).view(1, -1, 1, 1)
            stds = self.stds.to(fields.device).view(1, -1, 1, 1)
            fields = fields * stds + means

        if self.scale_turbulent:
            log_nu = fields[:, self.turbulent_channel_idx : self.turbulent_channel_idx + 1]
            k = self.k.to(fields.device) if hasattr(self, "k") else self.k_value
            fields[:, self.turbulent_channel_idx : self.turbulent_channel_idx + 1] = torch.expm1(log_nu) / k
        return fields

    def normalize_single_channel(self, nu_tilde: torch.Tensor) -> torch.Tensor:
        if not self.scale_turbulent:
            return nu_tilde
        k = self.k.to(nu_tilde.device) if hasattr(self, "k") else self.k_value
        nu_scaled = torch.log1p(k * nu_tilde)
        if self.normalize:
            mean = self.means[self.turbulent_channel_idx].to(nu_tilde.device)
            std = self.stds[self.turbulent_channel_idx].to(nu_tilde.device)
            nu_scaled = (nu_scaled - mean) / std
        return nu_scaled

    def denormalize_single_channel(self, normalized: torch.Tensor) -> torch.Tensor:
        if not self.scale_turbulent:
            return normalized
        nu_scaled = normalized
        if self.normalize:
            mean = self.means[self.turbulent_channel_idx].to(normalized.device)
            std = self.stds[self.turbulent_channel_idx].to(normalized.device)
            nu_scaled = nu_scaled * std + mean
        k = self.k.to(nu_scaled.device) if hasattr(self, "k") else self.k_value
        return torch.expm1(nu_scaled) / k

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.transform(fields)

    def get_config(self) -> dict[str, Any]:
        return {
            "scale_turbulent": self.scale_turbulent,
            "normalize": self.normalize,
            "turbulent_channel_idx": self.turbulent_channel_idx,
            "scale_factor": self.k_value if self.scale_turbulent else None,
        }


def create_normalizer(
    stats_path: str | Path | None,
    scale_turbulent: bool = False,
    normalize: bool = False,
    **kwargs: Any,
) -> Optional[FieldNormalizer]:
    """Create a FieldNormalizer, returning None when normalization is disabled."""
    if not scale_turbulent and not normalize:
        return None
    if stats_path is None:
        raise ValueError("stats_path is required when field normalization is enabled")
    return FieldNormalizer(
        stats_path=stats_path,
        scale_turbulent=scale_turbulent,
        normalize=normalize,
        **kwargs,
    )


def _get_nested_value(config: Any, key: str) -> Any:
    value = config
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            return None
        if value is None:
            return None
    return value


def _unwrap_dataset(dataset: Any) -> Any:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def load_normalizer_from_config_or_dataset(
    config: Any,
    train_loader: Any,
    config_keys: Optional[list[str]] = None,
    scale_turbulent: bool = True,
    normalize: bool = True,
) -> Tuple[Optional[FieldNormalizer], str]:
    """Load a normalizer with priority: config path, then dataset normalizer."""
    config_keys = config_keys or ["stats_path", "data.stats_path"]
    normalizer = None
    source = "none"

    for key in config_keys:
        stats_file = _get_nested_value(config, key)
        if stats_file and Path(stats_file).exists():
            normalizer = create_normalizer(stats_file, scale_turbulent, normalize)
            source = f"config: {stats_file}"
            break

    if normalizer is None:
        base_dataset = _unwrap_dataset(train_loader.dataset)
        get_normalizer = getattr(base_dataset, "get_normalizer", None)
        if callable(get_normalizer):
            normalizer = get_normalizer()
            if normalizer is not None:
                source = f"dataset: {base_dataset.__class__.__name__}"

    return normalizer, source


__all__ = [
    "FieldNormalizer",
    "create_normalizer",
    "load_normalizer_from_config_or_dataset",
]
