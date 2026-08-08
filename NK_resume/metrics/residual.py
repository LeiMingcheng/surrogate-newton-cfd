"""Residual-history metrics independent from ADflow internals."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

from ..exceptions import ContractError
from .trajectory import hitting_time


RESIDUAL_METRICS_SCHEMA = "residual_metrics_v1"


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): v for k, v in dict(value or {}).items()}


def _raw_tuple(values: Iterable[Any] | Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(values.values()) if isinstance(values, Mapping) else tuple(values)


def _series(raw_values: tuple[Any, ...]) -> tuple[float, ...]:
    out: list[float] = []
    for value in raw_values:
        if hasattr(value, "item"):
            value = value.item()
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ContractError("residual values must be numeric") from exc
        if math.isfinite(number):
            out.append(number)
    if not out:
        raise ContractError("residual_metrics requires at least one finite value")
    return tuple(out)


def _nonfinite_count(raw_values: tuple[Any, ...]) -> int:
    count = 0
    for value in raw_values:
        if hasattr(value, "item"):
            value = value.item()
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ContractError("residual values must be numeric") from exc
        if not math.isfinite(number):
            count += 1
    return count


def _budget_series(budgets: Iterable[Any] | None, length: int) -> tuple[int, ...]:
    if budgets is None:
        return tuple(range(length))
    out = tuple(int(v) for v in budgets)
    if len(out) != length:
        raise ContractError("budgets length must match residual values length")
    return out


@dataclass(frozen=True)
class ResidualMetric:
    """Residual summary for one stage or one projection trajectory."""

    values: tuple[float, ...]
    budgets: tuple[int, ...] = ()
    threshold: float | None = None
    nonfinite_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in self.values)
        if not values:
            raise ContractError("ResidualMetric.values must not be empty")
        if any(not math.isfinite(v) for v in values):
            raise ContractError("ResidualMetric.values must be finite")
        budgets = tuple(int(v) for v in self.budgets) if self.budgets else tuple(range(len(values)))
        if len(budgets) != len(values):
            raise ContractError("ResidualMetric.budgets length must match values")
        threshold = None if self.threshold is None else float(self.threshold)
        if threshold is not None and threshold <= 0.0:
            raise ContractError("ResidualMetric.threshold must be positive when set")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "nonfinite_count", int(self.nonfinite_count))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        initial = self.values[0]
        final = self.values[-1]
        reduction = initial - final
        relative_reduction = reduction / abs(initial) if initial else None
        log10_final = math.log10(final) if final > 0.0 else None
        hit_budget = None
        if self.threshold is not None:
            hit_budget = hitting_time(self.values, self.budgets, self.threshold)
        return {
            "schema_version": RESIDUAL_METRICS_SCHEMA,
            "values": list(self.values),
            "budgets": list(self.budgets),
            "threshold": self.threshold,
            "summary": {
                "count": len(self.values),
                "initial": initial,
                "final": final,
                "min": min(self.values),
                "max": max(self.values),
                "mean": sum(self.values) / len(self.values),
                "reduction": reduction,
                "relative_reduction": relative_reduction,
                "log10_final": log10_final,
                "hit_budget": hit_budget,
                "nonfinite_count": self.nonfinite_count,
            },
            "metadata": dict(self.metadata),
        }


def residual_metrics(
    values: Iterable[Any] | Mapping[str, Any],
    *,
    budgets: Iterable[Any] | None = None,
    threshold: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute residual-history summary metrics."""

    raw_values = _raw_tuple(values)
    series = _series(raw_values)
    return ResidualMetric(
        values=series,
        budgets=_budget_series(budgets, len(series)),
        threshold=threshold,
        nonfinite_count=_nonfinite_count(raw_values),
        metadata=_metadata(metadata),
    ).to_dict()
