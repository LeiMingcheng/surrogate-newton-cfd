"""Canonical Surrogate-Newton CFD surrogate core package."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "1.0.0"

__all__ = [
    "create_model",
    "ModelRegistry",
    "ModelConfig",
    "DataConfig",
    "TaskConfig",
    "TrainingLossConfig",
    "TrainingConfig",
    "FSBConfig",
    "RuntimeConfig",
    "EvaluationConfig",
    "NKResumeConfig",
    "ExperimentConfig",
    "ConfigManager",
    "load_config",
    "run_experiment",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "create_model": ("surrogate.models", "create_model"),
    "ModelRegistry": ("surrogate.models", "ModelRegistry"),
    "ModelConfig": ("surrogate.configs", "ModelConfig"),
    "DataConfig": ("surrogate.configs", "DataConfig"),
    "TaskConfig": ("surrogate.configs", "TaskConfig"),
    "TrainingLossConfig": ("surrogate.configs", "TrainingLossConfig"),
    "TrainingConfig": ("surrogate.configs", "TrainingConfig"),
    "FSBConfig": ("surrogate.configs", "FSBConfig"),
    "RuntimeConfig": ("surrogate.configs", "RuntimeConfig"),
    "EvaluationConfig": ("surrogate.configs", "EvaluationConfig"),
    "NKResumeConfig": ("surrogate.configs", "NKResumeConfig"),
    "ExperimentConfig": ("surrogate.configs", "ExperimentConfig"),
    "ConfigManager": ("surrogate.configs", "ConfigManager"),
    "load_config": ("surrogate.configs", "load_config"),
    "run_experiment": ("surrogate.entrypoints", "run_experiment"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module 'surrogate' has no attribute {name!r}")

    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr
