"""Field-level error metrics for clean NK_resume results."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..exceptions import ContractError


FIELD_METRICS_SCHEMA = "field_metrics_v1"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): _jsonable(v) for k, v in dict(value or {}).items()}


def _array(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise ContractError(f"{name} is required")
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - numpy owns exact exception type.
        raise ContractError(f"{name} cannot be converted to a numeric array") from exc
    if array.size == 0:
        raise ContractError(f"{name} must not be empty")
    return array


def _optional_array(value: Any, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _array(value, name=name)


def _require_same_shape(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape != right.shape:
        raise ContractError(
            f"{left_name} shape {left.shape} must match {right_name} shape {right.shape}"
        )


def _mask_array(mask: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    try:
        mask_array = np.asarray(mask, dtype=bool)
    except Exception as exc:  # pragma: no cover - numpy owns exact exception type.
        raise ContractError("mask cannot be converted to a boolean array") from exc
    try:
        return np.broadcast_to(mask_array, shape)
    except ValueError as exc:
        raise ContractError(f"mask shape {mask_array.shape} cannot broadcast to {shape}") from exc


def _finite_pair_mask(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    finite = np.isfinite(left) & np.isfinite(right)
    if mask is not None:
        finite = finite & mask
    return finite


def _comparison(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_name: str,
    right_name: str,
    mask: np.ndarray | None,
) -> dict[str, Any]:
    _require_same_shape(left, right, left_name=left_name, right_name=right_name)
    valid = _finite_pair_mask(left, right, mask)
    valid_count = int(valid.sum())
    if valid_count == 0:
        raise ContractError(f"{left_name} and {right_name} have no comparable finite values")

    diff = left[valid] - right[valid]
    abs_diff = np.abs(diff)
    squared = diff * diff
    mse = float(np.mean(squared))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff))
    mean_error = float(np.mean(diff))

    return {
        "left": left_name,
        "right": right_name,
        "shape": list(left.shape),
        "summary": {
            "valid_count": valid_count,
            "total_count": int(left.size),
            "masked_count": int(left.size - valid_count),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "max_abs": max_abs,
            "mean_error": mean_error,
        },
    }


def _axis(axis: int | None, ndim: int) -> int | None:
    if axis is None:
        return None
    out = int(axis)
    if out < 0:
        out += ndim
    if out < 0 or out >= ndim:
        raise ContractError(f"channel_axis {axis} is out of bounds for ndim={ndim}")
    return out


def _channel_name(index: int, names: Sequence[str]) -> str:
    if index < len(names) and str(names[index]).strip():
        return str(names[index])
    return f"channel_{index}"


def _per_channel(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_name: str,
    right_name: str,
    channel_axis: int | None,
    channel_names: Sequence[str],
    mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    axis = _axis(channel_axis, left.ndim)
    if axis is None:
        return []
    _require_same_shape(left, right, left_name=left_name, right_name=right_name)
    out: list[dict[str, Any]] = []
    for index in range(left.shape[axis]):
        left_slice = np.take(left, index, axis=axis)
        right_slice = np.take(right, index, axis=axis)
        mask_slice = None if mask is None else np.take(mask, index, axis=axis)
        summary = _comparison(
            left_slice,
            right_slice,
            left_name=left_name,
            right_name=right_name,
            mask=mask_slice,
        )
        summary["name"] = _channel_name(index, channel_names)
        summary["index"] = index
        out.append(summary)
    return out


def _summary_value(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return None
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def _improvement(
    candidate_vs_reference: dict[str, Any] | None,
    initial_vs_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    if not candidate_vs_reference or not initial_vs_reference:
        return {}
    candidate_mse = _summary_value(candidate_vs_reference, "mse")
    initial_mse = _summary_value(initial_vs_reference, "mse")
    candidate_rmse = _summary_value(candidate_vs_reference, "rmse")
    initial_rmse = _summary_value(initial_vs_reference, "rmse")
    def reduction(initial: float | None, candidate: float | None) -> float | None:
        if initial is None or candidate is None or initial <= 0.0:
            return None
        return (initial - candidate) / initial

    return {
        "mse_delta": None
        if initial_mse is None or candidate_mse is None
        else candidate_mse - initial_mse,
        "rmse_delta": None
        if initial_rmse is None or candidate_rmse is None
        else candidate_rmse - initial_rmse,
        "relative_mse_reduction": reduction(initial_mse, candidate_mse),
        "relative_rmse_reduction": reduction(initial_rmse, candidate_rmse),
    }


@dataclass(frozen=True)
class FieldMetric:
    """Field metric result for projected, reference, and optional initial fields."""

    candidate: Any
    reference: Any | None = None
    initial: Any | None = None
    mask: Any | None = None
    channel_axis: int | None = None
    channel_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        candidate = _array(self.candidate, name="candidate")
        reference = _optional_array(self.reference, name="reference")
        initial = _optional_array(self.initial, name="initial")
        mask = _mask_array(self.mask, candidate.shape)

        candidate_vs_reference = None
        initial_vs_reference = None
        candidate_vs_initial = None
        channels: dict[str, list[dict[str, Any]]] = {}

        if reference is not None:
            candidate_vs_reference = _comparison(
                candidate,
                reference,
                left_name="candidate",
                right_name="reference",
                mask=mask,
            )
            per_channel = _per_channel(
                candidate,
                reference,
                left_name="candidate",
                right_name="reference",
                channel_axis=self.channel_axis,
                channel_names=self.channel_names,
                mask=mask,
            )
            if per_channel:
                channels["candidate_vs_reference"] = per_channel

        if initial is not None:
            candidate_vs_initial = _comparison(
                candidate,
                initial,
                left_name="candidate",
                right_name="initial",
                mask=mask,
            )
            per_channel = _per_channel(
                candidate,
                initial,
                left_name="candidate",
                right_name="initial",
                channel_axis=self.channel_axis,
                channel_names=self.channel_names,
                mask=mask,
            )
            if per_channel:
                channels["candidate_vs_initial"] = per_channel

        if reference is not None and initial is not None:
            initial_vs_reference = _comparison(
                initial,
                reference,
                left_name="initial",
                right_name="reference",
                mask=mask,
            )
            per_channel = _per_channel(
                initial,
                reference,
                left_name="initial",
                right_name="reference",
                channel_axis=self.channel_axis,
                channel_names=self.channel_names,
                mask=mask,
            )
            if per_channel:
                channels["initial_vs_reference"] = per_channel

        return {
            "schema_version": FIELD_METRICS_SCHEMA,
            "shape": list(candidate.shape),
            "channel_axis": self.channel_axis,
            "channel_names": list(self.channel_names),
            "candidate_vs_reference": candidate_vs_reference,
            "initial_vs_reference": initial_vs_reference,
            "candidate_vs_initial": candidate_vs_initial,
            "improvement": _improvement(candidate_vs_reference, initial_vs_reference),
            "per_channel": channels,
            "metadata": _metadata(self.metadata),
        }


def field_metrics(
    candidate: Any,
    reference: Any | None = None,
    *,
    initial: Any | None = None,
    mask: Any | None = None,
    channel_axis: int | None = None,
    channel_names: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute field-level MSE and descriptive pointwise error metrics."""

    return FieldMetric(
        candidate=candidate,
        reference=reference,
        initial=initial,
        mask=mask,
        channel_axis=channel_axis,
        channel_names=tuple(str(name) for name in channel_names),
        metadata=_metadata(metadata),
    ).to_dict()
