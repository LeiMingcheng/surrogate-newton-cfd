"""Local base classes for the canonical surrogate package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class BaseSurrogate(nn.Module):
    """Shared base for models implemented inside ``surrogate``."""

    def __init__(
        self,
        geometry_dim: int = 27,
        flow_condition_dim: int = 3,
        coord_channels: int = 4,
        out_channels: int = 5,
        **params: Any,
    ) -> None:
        super().__init__()
        self.geometry_dim = int(geometry_dim)
        self.flow_condition_dim = int(flow_condition_dim)
        self.coord_channels = int(coord_channels)
        self.out_channels = int(out_channels)
        self.params = dict(params)
        self.field_names = [
            "Density",
            "VelocityX",
            "VelocityY",
            "Pressure",
            "TurbulentSANuTilde",
        ][: self.out_channels]
        self._model_info = {
            "geometry_dim": self.geometry_dim,
            "flow_condition_dim": self.flow_condition_dim,
            "coord_channels": self.coord_channels,
            "out_channels": self.out_channels,
            "model_type": self.__class__.__name__,
        }

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def count_parameters(self) -> tuple[int, int]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return total, trainable

    def resolve_initial_field(
        self,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
        initial_field: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return an explicitly supplied initial field, if this model consumes one."""
        del flow_conditions, coords
        return initial_field

    def preprocess_flow_conditions(self, flow_conditions: torch.Tensor) -> torch.Tensor:
        """Preprocess [Mach, AoA, Re] as [Mach, radians(AoA), log10(Re)]."""
        mach = flow_conditions[:, 0]
        aoa = flow_conditions[:, 1]
        reynolds = flow_conditions[:, 2]

        already_processed = (reynolds < 10.0).all()
        if already_processed:
            return flow_conditions

        log_re = torch.log10(torch.clamp(reynolds, min=1e3))
        aoa_rad = torch.deg2rad(aoa)
        return torch.stack([mach, aoa_rad, log_re], dim=1)

    def validate_inputs(
        self,
        geometry: torch.Tensor,
        flow_conditions: torch.Tensor,
        coords: torch.Tensor,
    ) -> bool:
        """Validate batch and channel dimensions used by local surrogate models."""
        batch_size = geometry.shape[0]

        if not (geometry.shape[0] == flow_conditions.shape[0] == coords.shape[0] == batch_size):
            print(
                "Batch size mismatch: "
                f"geo={geometry.shape[0]}, flow={flow_conditions.shape[0]}, coords={coords.shape[0]}"
            )
            return False

        if geometry.shape[1] != self.geometry_dim:
            print(f"Geometry dimension mismatch: expected {self.geometry_dim}, got {geometry.shape[1]}")
            return False

        if flow_conditions.shape[1] != self.flow_condition_dim:
            print(
                "Flow conditions dimension mismatch: "
                f"expected {self.flow_condition_dim}, got {flow_conditions.shape[1]}"
            )
            return False

        if coords.shape[1] != self.coord_channels:
            print(f"Coord channels mismatch: expected {self.coord_channels}, got {coords.shape[1]}")
            return False

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and metadata."""
        self._model_info.update({
            "num_parameters": sum(param.numel() for param in self.parameters()),
            "num_trainable_parameters": sum(param.numel() for param in self.parameters() if param.requires_grad),
            "field_names": self.field_names,
        })
        return self._model_info.copy()

    def print_model_info(self) -> None:
        """Print detailed model information."""
        info = self.get_model_info()

        print("=" * 60)
        print(f"Model: {info['model_type']}")
        print("=" * 60)
        print(f"Geometry dimension: {info['geometry_dim']}")
        print(f"Flow condition dimension: {info['flow_condition_dim']}")
        print(f"Coordinate channels: {info['coord_channels']}")
        print(f"Output channels: {info['out_channels']}")
        print(f"Field names: {info['field_names']}")
        print(f"Total parameters: {info['num_parameters']:,}")
        print(f"Trainable parameters: {info['num_trainable_parameters']:,}")
        print("=" * 60)

    def save_checkpoint(self, path: Union[str, Path], **kwargs: Any) -> None:
        """Save model checkpoint with metadata."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_info": self.get_model_info(),
            **kwargs,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_checkpoint(self, path: Union[str, Path], map_location: Optional[str] = None):
        """Load model checkpoint."""
        if map_location is None:
            map_location = str(self.device)

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {path}")
        return checkpoint


class BaseDirectModel(BaseSurrogate):
    """Base class for single-step direct surrogate models."""

    family = "direct"


class BaseBridgeModel(BaseSurrogate):
    """Base class for multi-step flow-state-bridge models."""

    family = "fsb"


__all__ = [
    "BaseSurrogate",
    "BaseDirectModel",
    "BaseBridgeModel",
]
