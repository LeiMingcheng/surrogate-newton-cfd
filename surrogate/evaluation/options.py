"""Build offline validation options from clean surrogate configs."""

from __future__ import annotations

from typing import Any, Optional

from surrogate.evaluation.metrics import FieldMetricsConfig
from surrogate.evaluation.physics import ForceMetricsConfig, ResidualMetricsConfig
from surrogate.evaluation.runners import ValidationOptions


def _field_metrics_from_config(config: Any) -> FieldMetricsConfig:
    values = config if isinstance(config, dict) else getattr(config, "__dict__", {})
    return FieldMetricsConfig(**dict(values or {}))


def _force_metrics_from_config(config: Any) -> ForceMetricsConfig:
    values = config if isinstance(config, dict) else getattr(config, "__dict__", {})
    return ForceMetricsConfig(**dict(values or {}))


def _residual_metrics_from_config(config: Any) -> ResidualMetricsConfig:
    values = config if isinstance(config, dict) else getattr(config, "__dict__", {})
    return ResidualMetricsConfig(**dict(values or {}))


def _override_bool(value: Optional[bool], fallback: bool) -> bool:
    return bool(fallback) if value is None else bool(value)


def _override_int(value: Optional[int], fallback: Optional[int]) -> Optional[int]:
    return fallback if value is None else int(value)


def build_validation_options_from_config(
    config: Any,
    *,
    max_batches: Optional[int] = None,
    record_samples: Optional[bool] = None,
    compute_physical_field_metrics: Optional[bool] = None,
    compute_forces: Optional[bool] = None,
    compute_residuals: Optional[bool] = None,
) -> ValidationOptions:
    """Create offline validation options from ``config.evaluation.rich``.

    CLI arguments pass explicit overrides as non-None values. ``None`` means
    the YAML config remains authoritative.
    """

    rich = getattr(getattr(config, "evaluation"), "rich")
    return ValidationOptions(
        field_metrics=_field_metrics_from_config(rich.field_metrics),
        compute_physical_field_metrics=_override_bool(
            compute_physical_field_metrics,
            rich.compute_physical_field_metrics,
        ),
        physical_field_metrics=_field_metrics_from_config(rich.physical_field_metrics),
        compute_forces=_override_bool(compute_forces, rich.compute_forces),
        force_metrics=_force_metrics_from_config(rich.force_metrics),
        compute_residuals=_override_bool(compute_residuals, rich.compute_residuals),
        residual_metrics=_residual_metrics_from_config(rich.residual_metrics),
        record_samples=_override_bool(record_samples, rich.record_samples),
        max_batches=_override_int(max_batches, rich.max_batches),
        inverse_transform_for_physics=bool(rich.inverse_transform_for_physics),
    )


__all__ = ["build_validation_options_from_config"]
