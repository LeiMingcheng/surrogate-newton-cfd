"""Reusable validation-loop helpers for training-time and offline evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import torch

from surrogate.evaluation.metrics import FieldMetricsConfig, compute_field_metrics
from surrogate.evaluation.physics import (
    ForceMetricsConfig,
    ResidualMetricsConfig,
    compute_force_metrics,
    compute_residual_batch_evaluation,
    compute_residual_pair_metrics,
)


BatchPredictFn = Callable[[Mapping[str, Any]], torch.Tensor]


@dataclass
class ValidationOptions:
    """Shared controls for direct and FSB validation/benchmark loops."""

    field_metrics: FieldMetricsConfig = field(default_factory=FieldMetricsConfig)
    compute_physical_field_metrics: bool = False
    physical_field_metrics: FieldMetricsConfig = field(default_factory=FieldMetricsConfig)
    compute_forces: bool = False
    force_metrics: ForceMetricsConfig = field(default_factory=ForceMetricsConfig)
    compute_residuals: bool = False
    residual_metrics: ResidualMetricsConfig = field(default_factory=ResidualMetricsConfig)
    record_samples: bool = False
    max_batches: Optional[int] = None
    inverse_transform_for_physics: bool = True


@dataclass
class ValidationResult:
    """Aggregated validation metrics and loop metadata."""

    metrics: Dict[str, float]
    n_samples: int
    n_batches: int
    sample_records: list[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "n_samples": int(self.n_samples),
            "n_batches": int(self.n_batches),
            "sample_records": [dict(record) for record in self.sample_records],
        }


def tensor_to_device(value: Any, device: torch.device) -> Any:
    """Move tensor values to a device while leaving metadata untouched."""
    return value.to(device) if isinstance(value, torch.Tensor) else value


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor entries in a dataloader batch to device."""
    return {key: tensor_to_device(value, device) for key, value in batch.items()}


def coords_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    if "coords" in batch:
        return batch["coords"]
    if "coords_center" in batch:
        return batch["coords_center"]
    raise KeyError("Validation batch requires 'coords' or 'coords_center'")


def target_from_batch(batch: Mapping[str, Any]) -> torch.Tensor:
    if "target" in batch:
        return batch["target"]
    if "fields" in batch:
        return batch["fields"]
    raise KeyError("Validation batch requires 'target' or 'fields'")


def volumes_from_batch(batch: Mapping[str, Any]) -> Optional[torch.Tensor]:
    volumes = batch.get("cell_volumes")
    if volumes is None:
        volumes = batch.get("volumes")
    return volumes if isinstance(volumes, torch.Tensor) else None


def sample_metadata_from_batch(
    batch: Mapping[str, Any],
    sample_index: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Extract stable sample identifiers without assuming a dataset schema."""
    metadata: Dict[str, Any] = {}
    for key in (
        "sample_id",
        "case_id",
        "case_name",
        "airfoil_id",
        "airfoil",
        "filename",
        "path",
        "cgns_basename",
        "source_name",
        "source_kind",
        "source_chunk",
        "source_index_path",
        "source_shard_root",
        "source_shard_path",
        "index",
        "global_id",
    ):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == batch_size:
                item = value[sample_index]
            else:
                item = value
            metadata[key] = item.detach().cpu().item() if item.numel() == 1 else item.detach().cpu().tolist()
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            metadata[key] = value[sample_index]
        else:
            metadata[key] = value
    return metadata


def maybe_inverse_transform(normalizer: Any, fields: torch.Tensor) -> torch.Tensor:
    if normalizer is None:
        return fields
    return normalizer.inverse_transform(fields)


def _slice_tensor(value: torch.Tensor, sample_index: int, batch_size: int, *, keep_batch: bool) -> torch.Tensor:
    if value.ndim > 0 and int(value.shape[0]) == batch_size:
        return value[sample_index:sample_index + 1] if keep_batch else value[sample_index]
    return value


def _slice_optional_tensor(
    value: Any,
    sample_index: int,
    batch_size: int,
    *,
    keep_batch: bool,
) -> Any:
    if isinstance(value, torch.Tensor):
        return _slice_tensor(value, sample_index, batch_size, keep_batch=keep_batch)
    return value


def _sample_batch(batch: Mapping[str, Any], sample_index: int, batch_size: int) -> Dict[str, Any]:
    return {
        key: _slice_optional_tensor(value, sample_index, batch_size, keep_batch=True)
        for key, value in batch.items()
    }


def _prefix_metrics(prefix: str, metrics: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _merge_record_updates(
    records: list[Dict[str, Any]],
    updates: list[Mapping[str, Any]],
) -> None:
    for record, update in zip(records, updates):
        record.update(dict(update))


def _normalize_residual_targets(config: ResidualMetricsConfig) -> tuple[str, ...]:
    targets = tuple(str(target).lower() for target in config.targets)
    if not targets:
        raise ValueError("residual_metrics.targets must contain at least one target")
    unknown = sorted(set(targets) - {"pred", "target"})
    if unknown:
        raise ValueError(f"Unknown residual target(s): {unknown}")
    return targets


def _residual_samples_from_batch(
    fields: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    config: ResidualMetricsConfig,
) -> list[Dict[str, Any]]:
    batch_size = int(fields.shape[0])
    if "coords_vertex" not in batch:
        raise KeyError("Residual evaluation requires 'coords_vertex' in validation batch")
    if "coords_center_pde" in batch:
        coords_center = batch["coords_center_pde"]
    else:
        coords_center = coords_from_batch(batch)[:, :2]
    coords_vertex = batch["coords_vertex"]
    flow_conditions = batch["flow_conditions"]
    samples: list[Dict[str, Any]] = []
    for sample_index in range(batch_size):
        sample: Dict[str, Any] = {
            "fields": _slice_tensor(fields, sample_index, batch_size, keep_batch=True),
            "coords": {
                "center": _slice_tensor(coords_center, sample_index, batch_size, keep_batch=True),
                "vertex": _slice_tensor(coords_vertex, sample_index, batch_size, keep_batch=True),
            },
            "flow_conditions": _slice_tensor(flow_conditions, sample_index, batch_size, keep_batch=True),
            "weights": dict(config.weights or {}),
            "periodic_xi": bool(config.periodic_xi),
            "preserve_residual_dtype": bool(config.preserve_residual_dtype),
            "state_is_adflow_consistent": bool(config.state_is_adflow_consistent),
            "state_is_adflow_mixed": bool(config.state_is_adflow_mixed),
            "return_components": bool(config.return_components),
        }
        if config.wall_layers is not None:
            sample["wall_layers"] = int(config.wall_layers)
        if config.spatial_wall_layers is not None:
            sample["spatial_wall_layers"] = int(config.spatial_wall_layers)
        if config.dtype is not None:
            sample["dtype"] = config.dtype
        if "wall_distance" in batch:
            sample["wall_distance"] = _slice_optional_tensor(
                batch["wall_distance"],
                sample_index,
                batch_size,
                keep_batch=True,
            )
        samples.append(sample)
    return samples


def _accumulate_weighted(
    totals: Dict[str, float],
    metrics: Mapping[str, float],
    batch_size: int,
) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value) * float(batch_size)


def _finalize_weighted(totals: Mapping[str, float], n_samples: int) -> Dict[str, float]:
    if n_samples <= 0:
        raise ValueError("Validation requires at least one sample")
    return {key: float(value) / float(n_samples) for key, value in totals.items()}


def evaluate_prediction_batches(
    dataloader: Iterable[Mapping[str, Any]],
    *,
    predict_batch: BatchPredictFn,
    device: str | torch.device,
    options: Optional[ValidationOptions] = None,
    normalizer: Any = None,
    residual_calculator: Any = None,
) -> ValidationResult:
    """Evaluate predictions over labeled batches.

    This helper may be called by trainers for realtime validation or by offline
    benchmark scripts. It owns target/metric aggregation and is intentionally
    outside the inference backend.
    """
    opts = options or ValidationOptions()
    torch_device = torch.device(device)
    totals: Dict[str, float] = {}
    n_samples = 0
    n_batches = 0
    sample_records: list[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(dataloader):
            if opts.max_batches is not None and batch_idx >= int(opts.max_batches):
                break
            batch = move_batch_to_device(raw_batch, torch_device)
            target = target_from_batch(batch)
            pred = predict_batch(batch)
            batch_size = int(target.shape[0])
            records: list[Dict[str, Any]] = []
            if opts.record_samples:
                records = [
                    {
                        "sample_index": n_samples + sample_idx,
                        "batch_index": batch_idx,
                        "batch_sample_index": sample_idx,
                        **sample_metadata_from_batch(batch, sample_idx, batch_size),
                    }
                    for sample_idx in range(batch_size)
                ]

            metrics = compute_field_metrics(
                pred,
                target,
                batch=batch,
                volumes=volumes_from_batch(batch),
                config=opts.field_metrics,
            )
            if opts.record_samples:
                for sample_idx, record in enumerate(records):
                    sample_metrics = compute_field_metrics(
                        _slice_tensor(pred, sample_idx, batch_size, keep_batch=True),
                        _slice_tensor(target, sample_idx, batch_size, keep_batch=True),
                        batch=_sample_batch(batch, sample_idx, batch_size),
                        volumes=_slice_optional_tensor(volumes_from_batch(batch), sample_idx, batch_size, keep_batch=True),
                        config=opts.field_metrics,
                    )
                    record.update(_prefix_metrics("field", sample_metrics))

            pred_for_physics = pred
            target_for_physics = target
            needs_physical_fields = (
                opts.compute_physical_field_metrics
                or opts.compute_forces
                or opts.compute_residuals
            )
            if needs_physical_fields and opts.inverse_transform_for_physics:
                pred_for_physics = maybe_inverse_transform(normalizer, pred_for_physics)
                target_for_physics = maybe_inverse_transform(normalizer, target_for_physics)

            if opts.compute_physical_field_metrics:
                physical_metrics = compute_field_metrics(
                    pred_for_physics,
                    target_for_physics,
                    batch=batch,
                    volumes=volumes_from_batch(batch),
                    config=opts.physical_field_metrics,
                )
                metrics.update(_prefix_metrics("physical_field", physical_metrics))
                if opts.record_samples:
                    for sample_idx, record in enumerate(records):
                        sample_physical_metrics = compute_field_metrics(
                            _slice_tensor(pred_for_physics, sample_idx, batch_size, keep_batch=True),
                            _slice_tensor(target_for_physics, sample_idx, batch_size, keep_batch=True),
                            batch=_sample_batch(batch, sample_idx, batch_size),
                            volumes=_slice_optional_tensor(volumes_from_batch(batch), sample_idx, batch_size, keep_batch=True),
                            config=opts.physical_field_metrics,
                        )
                        record.update(_prefix_metrics("physical_field", sample_physical_metrics))

            if opts.compute_forces:
                if "coords_vertex" not in batch:
                    raise KeyError("compute_forces=True requires 'coords_vertex' in validation batch")
                metrics.update(
                    compute_force_metrics(
                        pred_for_physics,
                        target_for_physics,
                        batch["coords_vertex"],
                        batch["flow_conditions"],
                        config=opts.force_metrics,
                    )
                )
                if opts.record_samples:
                    for sample_idx, record in enumerate(records):
                        sample_force_metrics = compute_force_metrics(
                            _slice_tensor(pred_for_physics, sample_idx, batch_size, keep_batch=True),
                            _slice_tensor(target_for_physics, sample_idx, batch_size, keep_batch=True),
                            _slice_tensor(batch["coords_vertex"], sample_idx, batch_size, keep_batch=True),
                            _slice_tensor(batch["flow_conditions"], sample_idx, batch_size, keep_batch=True),
                            config=opts.force_metrics,
                        )
                        record.update(sample_force_metrics)

            if opts.compute_residuals:
                if residual_calculator is None:
                    raise ValueError("compute_residuals=True requires residual_calculator")
                residual_targets = _normalize_residual_targets(opts.residual_metrics)
                residual_records_by_target: dict[str, list[Dict[str, float]]] = {}
                if "pred" in residual_targets:
                    pred_residual_metrics, pred_residual_records = compute_residual_batch_evaluation(
                        residual_calculator,
                        _residual_samples_from_batch(
                            pred_for_physics,
                            batch,
                            config=opts.residual_metrics,
                        ),
                        prefix="pred_residual",
                        include_records=True,
                    )
                    metrics.update(pred_residual_metrics)
                    residual_records_by_target["pred"] = pred_residual_records
                    if opts.record_samples:
                        _merge_record_updates(records, pred_residual_records)
                if "target" in residual_targets:
                    target_residual_metrics, target_residual_records = compute_residual_batch_evaluation(
                        residual_calculator,
                        _residual_samples_from_batch(
                            target_for_physics,
                            batch,
                            config=opts.residual_metrics,
                        ),
                        prefix="target_residual",
                        include_records=True,
                    )
                    metrics.update(target_residual_metrics)
                    residual_records_by_target["target"] = target_residual_records
                    if opts.record_samples:
                        _merge_record_updates(records, target_residual_records)
                if {"pred", "target"}.issubset(residual_records_by_target):
                    metrics.update(
                        compute_residual_pair_metrics(
                            [
                                record["pred_residual_score"]
                                for record in residual_records_by_target["pred"]
                            ],
                            [
                                record["target_residual_score"]
                                for record in residual_records_by_target["target"]
                            ],
                        )
                    )
                    if opts.record_samples:
                        for record in records:
                            if "pred_residual_score" in record and "target_residual_score" in record:
                                diff = float(record["pred_residual_score"]) - float(record["target_residual_score"])
                                record["residual_score_error"] = diff
                                record["residual_score_abs_error"] = abs(diff)

            if opts.record_samples:
                sample_records.extend(records)

            _accumulate_weighted(totals, metrics, batch_size)
            n_samples += batch_size
            n_batches += 1

    return ValidationResult(
        metrics=_finalize_weighted(totals, n_samples),
        n_samples=n_samples,
        n_batches=n_batches,
        sample_records=sample_records,
    )


__all__ = [
    "ValidationOptions",
    "ValidationResult",
    "coords_from_batch",
    "evaluate_prediction_batches",
    "maybe_inverse_transform",
    "move_batch_to_device",
    "sample_metadata_from_batch",
    "target_from_batch",
    "tensor_to_device",
    "volumes_from_batch",
]
