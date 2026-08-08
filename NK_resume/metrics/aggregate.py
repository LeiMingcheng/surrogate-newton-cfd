"""Aggregation over clean replay/projection result dictionaries."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping

from ..exceptions import ContractError


AGGREGATE_METRICS_SCHEMA = "aggregate_metrics_v1"


def _result_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return {str(k): v for k, v in result.items()}
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return {str(k): v for k, v in vars(result).items()}
    raise ContractError(f"Unsupported result object: {type(result).__name__}")


def _stage_dict(stage: Any) -> dict[str, Any]:
    if isinstance(stage, Mapping):
        return {str(k): v for k, v in stage.items()}
    if is_dataclass(stage):
        return asdict(stage)
    raise ContractError(f"Unsupported stage object: {type(stage).__name__}")


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float | None:
    value_tuple = tuple(values)
    if not value_tuple:
        return None
    return sum(value_tuple) / len(value_tuple)


def _stage_metrics(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = stage.get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _force_max_abs(stage: Mapping[str, Any]) -> float | None:
    metrics = _stage_metrics(stage)
    force = metrics.get("force") or stage.get("force")
    if not isinstance(force, Mapping):
        return None
    return _number(
        _nested(force, "summary", "max_abs_delta")
        if _nested(force, "summary", "max_abs_delta") is not None
        else force.get("max_abs_delta")
    )


def _residual_final(stage: Mapping[str, Any]) -> float | None:
    metrics = _stage_metrics(stage)
    residual = metrics.get("residual") or stage.get("residual")
    if not isinstance(residual, Mapping):
        return None
    return _number(
        _nested(residual, "summary", "final")
        if _nested(residual, "summary", "final") is not None
        else residual.get("final")
    )


def _residual_hit(stage: Mapping[str, Any]) -> bool | None:
    metrics = _stage_metrics(stage)
    residual = metrics.get("residual") or stage.get("residual")
    if not isinstance(residual, Mapping):
        return None
    hit_budget = _nested(residual, "summary", "hit_budget")
    if hit_budget is None:
        return None
    return True


def _field_metric(stage: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metrics = _stage_metrics(stage)
    field = metrics.get("field") or stage.get("field")
    return field if isinstance(field, Mapping) else None


def _field_compare(stage: Mapping[str, Any]) -> Mapping[str, Any] | None:
    field = _field_metric(stage)
    if field is None:
        return None
    compare = field.get("candidate_vs_reference")
    return compare if isinstance(compare, Mapping) else None


def _field_summary_value(stage: Mapping[str, Any], key: str) -> float | None:
    compare = _field_compare(stage)
    if compare is None:
        return None
    return _number(_nested(compare, "summary", key))


def _field_improvement_value(stage: Mapping[str, Any], key: str) -> float | None:
    field = _field_metric(stage)
    if field is None:
        return None
    improvement = field.get("improvement")
    if not isinstance(improvement, Mapping):
        return None
    return _number(improvement.get(key))


def aggregate_results(results: Iterable[Any]) -> dict[str, Any]:
    """Aggregate clean projection result dicts by stage name."""

    result_tuple = tuple(_result_dict(result) for result in results)
    if not result_tuple:
        raise ContractError("aggregate_results requires at least one result")

    case_ids: list[str] = []
    by_stage: dict[str, dict[str, Any]] = {}
    for result in result_tuple:
        case_id = str(result.get("case_id", ""))
        if case_id:
            case_ids.append(case_id)
        stages = result.get("stages", ())
        if not isinstance(stages, (list, tuple)):
            raise ContractError("result.stages must be a list or tuple")
        for stage_raw in stages:
            stage = _stage_dict(stage_raw)
            name = str(stage.get("name") or stage.get("stage_name") or "final")
            bucket = by_stage.setdefault(
                name,
                {
                    "count": 0,
                    "force_max_abs_delta_values": [],
                    "residual_final_values": [],
                    "residual_hit_count": 0,
                    "field_mse_values": [],
                    "field_rmse_values": [],
                    "field_relative_mse_reduction_values": [],
                },
            )
            bucket["count"] += 1
            force_value = _force_max_abs(stage)
            if force_value is not None:
                bucket["force_max_abs_delta_values"].append(force_value)
            residual_value = _residual_final(stage)
            if residual_value is not None:
                bucket["residual_final_values"].append(residual_value)
            residual_hit = _residual_hit(stage)
            if residual_hit:
                bucket["residual_hit_count"] += 1
            field_mse = _field_summary_value(stage, "mse")
            if field_mse is not None:
                bucket["field_mse_values"].append(field_mse)
            field_rmse = _field_summary_value(stage, "rmse")
            if field_rmse is not None:
                bucket["field_rmse_values"].append(field_rmse)
            field_mse_reduction = _field_improvement_value(
                stage,
                "relative_mse_reduction",
            )
            if field_mse_reduction is not None:
                bucket["field_relative_mse_reduction_values"].append(field_mse_reduction)

    stage_summary: dict[str, dict[str, Any]] = {}
    for name, bucket in by_stage.items():
        force_values = tuple(bucket["force_max_abs_delta_values"])
        residual_values = tuple(bucket["residual_final_values"])
        field_mse_values = tuple(bucket["field_mse_values"])
        field_rmse_values = tuple(bucket["field_rmse_values"])
        field_mse_reduction_values = tuple(bucket["field_relative_mse_reduction_values"])
        stage_summary[name] = {
            "count": bucket["count"],
            "force_max_abs_delta_mean": _mean(force_values),
            "force_max_abs_delta_max": max(force_values) if force_values else None,
            "residual_final_mean": _mean(residual_values),
            "residual_final_min": min(residual_values) if residual_values else None,
            "residual_hit_count": bucket["residual_hit_count"],
            "field_mse_mean": _mean(field_mse_values),
            "field_mse_min": min(field_mse_values) if field_mse_values else None,
            "field_rmse_mean": _mean(field_rmse_values),
            "field_rmse_min": min(field_rmse_values) if field_rmse_values else None,
            "field_relative_mse_reduction_mean": _mean(field_mse_reduction_values),
        }

    return {
        "schema_version": AGGREGATE_METRICS_SCHEMA,
        "case_count": len(set(case_ids)) if case_ids else len(result_tuple),
        "result_count": len(result_tuple),
        "case_ids": case_ids,
        "by_stage": stage_summary,
    }
