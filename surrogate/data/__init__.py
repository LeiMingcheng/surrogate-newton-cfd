"""Canonical data utilities for surrogate workflows."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BaseFieldDataset",
    "FieldNormalizer",
    "H5MultiFieldDataset",
    "UniformFlowInitializer",
    "collect_sample_ordinals",
    "create_dataloaders",
    "create_dataloaders_from_config",
    "create_normalizer",
    "create_uniform_flow_initializer",
    "get_base_dataset",
    "load_normalizer_from_config_or_dataset",
    "prefer_geometry_orig_batch",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "BaseFieldDataset": ("surrogate.data.base", "BaseFieldDataset"),
    "FieldNormalizer": ("surrogate.data.normalizers", "FieldNormalizer"),
    "H5MultiFieldDataset": ("surrogate.data.h5_dataset", "H5MultiFieldDataset"),
    "UniformFlowInitializer": ("surrogate.data.uniform_flow_initializer", "UniformFlowInitializer"),
    "collect_sample_ordinals": ("surrogate.data.dataset_utils", "collect_sample_ordinals"),
    "create_dataloaders": ("surrogate.data.loaders", "create_dataloaders"),
    "create_dataloaders_from_config": ("surrogate.data.loaders", "create_dataloaders_from_config"),
    "create_normalizer": ("surrogate.data.normalizers", "create_normalizer"),
    "create_uniform_flow_initializer": ("surrogate.data.uniform_flow_initializer", "create_uniform_flow_initializer"),
    "get_base_dataset": ("surrogate.data.dataset_utils", "get_base_dataset"),
    "load_normalizer_from_config_or_dataset": ("surrogate.data.normalizers", "load_normalizer_from_config_or_dataset"),
    "prefer_geometry_orig_batch": ("surrogate.data.dataset_utils", "prefer_geometry_orig_batch"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module 'surrogate.data' has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr
