"""PDE residual calculation and reporting utilities."""

from __future__ import annotations

import importlib
import json
from typing import Any

__all__ = [
    "PDEResidualCalculator",
    "TorchResidualBackend",
    "get_residual_calculator",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "PDEResidualCalculator": ("surrogate.physics.residual.calculator", "PDEResidualCalculator"),
    "TorchResidualBackend": ("surrogate.physics.residual.torch_backend", "TorchResidualBackend"),
}

_GLOBAL_RESIDUAL_CALCULATORS: dict[tuple[str, str], Any] = {}


def _normalize_backend_name(backend: str) -> str:
    backend_name = str(backend).lower()
    return "torch" if backend_name == "pytorch" else backend_name


def _cache_payload(kwargs: dict[str, Any]) -> str:
    try:
        return json.dumps(kwargs, sort_keys=True, default=str)
    except TypeError:
        return repr(sorted((key, repr(value)) for key, value in kwargs.items()))


def get_residual_calculator(
    backend: str = "torch",
    *,
    force_new: bool = False,
    **kwargs: Any,
) -> Any:
    """Create or reuse a residual calculator for a backend/config pair."""
    backend_name = _normalize_backend_name(backend)
    cache_key = (backend_name, _cache_payload(kwargs))
    if force_new or cache_key not in _GLOBAL_RESIDUAL_CALCULATORS:
        calculator_cls = __getattr__("PDEResidualCalculator")
        _GLOBAL_RESIDUAL_CALCULATORS[cache_key] = calculator_cls(backend=backend_name, **kwargs)
    return _GLOBAL_RESIDUAL_CALCULATORS[cache_key]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module 'surrogate.physics.residual' has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr
