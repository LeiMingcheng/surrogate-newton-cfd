"""Unified airfoil optimization framework."""

from optimization.config import OptimizationConfig, load_optimization_config
from optimization.contracts import CandidateEvaluation, OperatingPointResult

__all__ = [
    "CandidateEvaluation",
    "OperatingPointResult",
    "OptimizationConfig",
    "load_optimization_config",
]
