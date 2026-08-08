"""Trajectory metrics that are independent from solver internals."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..exceptions import ContractError


TRAJECTORY_METRICS_SCHEMA = "trajectory_metrics_v1"


def hitting_time(values: Iterable[float], budgets: Iterable[int], threshold: float) -> int | None:
    """Return first budget whose value is at or below `threshold`."""

    threshold = float(threshold)
    for value, budget in zip(values, budgets):
        if float(value) <= threshold:
            return int(budget)
    return None


def _float_tuple(values: Iterable[Any]) -> tuple[float, ...]:
    out = tuple(float(v.item() if hasattr(v, "item") else v) for v in values)
    if not out:
        raise ContractError("trajectory values must not be empty")
    return out


def _budget_tuple(budgets: Iterable[Any] | None, length: int) -> tuple[int, ...]:
    if budgets is None:
        return tuple(range(length))
    out = tuple(int(v) for v in budgets)
    if len(out) != length:
        raise ContractError("trajectory budgets length must match values length")
    return out


def trajectory_summary(
    values: Iterable[Any],
    *,
    budgets: Iterable[Any] | None = None,
    thresholds: Iterable[float] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize a scalar metric trajectory."""

    value_tuple = _float_tuple(values)
    budget_tuple = _budget_tuple(budgets, len(value_tuple))
    threshold_tuple = tuple(float(v) for v in thresholds)
    hits = {
        str(threshold): hitting_time(value_tuple, budget_tuple, threshold)
        for threshold in threshold_tuple
    }
    best_index = min(range(len(value_tuple)), key=value_tuple.__getitem__)
    return {
        "schema_version": TRAJECTORY_METRICS_SCHEMA,
        "values": list(value_tuple),
        "budgets": list(budget_tuple),
        "summary": {
            "count": len(value_tuple),
            "initial": value_tuple[0],
            "final": value_tuple[-1],
            "best": value_tuple[best_index],
            "best_budget": budget_tuple[best_index],
            "hit_budgets": hits,
        },
        "metadata": {str(k): v for k, v in dict(metadata or {}).items()},
    }
