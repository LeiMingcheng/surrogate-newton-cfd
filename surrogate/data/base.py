"""Base dataset interface for surrogate field datasets."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


class BaseFieldDataset(torch.utils.data.Dataset):
    """Base class for field datasets used by direct and FSB workflows."""

    def __init__(self) -> None:
        self.field_names = [
            "Density",
            "VelocityX",
            "VelocityY",
            "Pressure",
            "TurbulentSANuTilde",
        ]
        self.geometry_names = ["Enhanced CST"]
        self.flow_names = ["Mach", "AoA", "Reynolds"]
        self.n_fields = 5
        self.geometry_dim = 27
        self.flow_dim = 3
        self.image_shape = (84, 304)
        self.normalization_stats = None

    def get_shape_info(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "fields": (self.n_fields,) + self.image_shape,
            "coords_center": (4,) + self.image_shape,
            "coords_center_pde": (2,) + self.image_shape,
            "coords_vertex": (2, self.image_shape[0] + 1, self.image_shape[1] + 1),
            "geometry": (self.geometry_dim,),
            "flow_conditions": (self.flow_dim,),
        }

    def get_normalization_stats(self) -> Optional[Dict[str, Any]]:
        return self.normalization_stats

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        required_keys = ["coords_center", "coords_vertex", "geometry", "flow_conditions"]
        if not all(key in sample for key in required_keys):
            return False
        for key in required_keys:
            if not isinstance(sample[key], torch.Tensor):
                return False
        if "fields" in sample and not isinstance(sample["fields"], torch.Tensor):
            return False

        expected_shapes = self.get_shape_info()
        for key in required_keys:
            if tuple(sample[key].shape) != expected_shapes[key]:
                return False
        if "fields" in sample and tuple(sample["fields"].shape) != expected_shapes["fields"]:
            return False
        return True

    def get_field_names(self) -> list[str]:
        return self.field_names.copy()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_type": self.__class__.__name__,
            "n_fields": self.n_fields,
            "field_names": self.field_names,
            "geometry_dim": self.geometry_dim,
            "flow_dim": self.flow_dim,
            "image_shape": self.image_shape,
            "normalization": self.normalization_stats is not None,
        }
