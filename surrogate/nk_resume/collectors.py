"""Model-side collectors for clean NK resume workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from NK_resume import (
    ContractError,
    ResumeCase,
    build_direct_case,
    build_fsb_case,
)
from surrogate.configs import load_config
from surrogate.data import H5MultiFieldDataset
from surrogate.nk_resume.contracts import ResumePrediction, ResumeRequest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): item for key, item in dict(value or {}).items()}


def _path_text(value: str | Path | None) -> str:
    if value is None:
        return ""
    return str(value)


def _project_path(value: str | Path, *, name: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ContractError(f"{name} is required")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _tensor_to_numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.size == 0:
        raise ContractError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return np.asarray(array)


def _single_array(value: Any, *, name: str) -> np.ndarray:
    array = _tensor_to_numpy(value, name=name)
    if array.ndim >= 1 and int(array.shape[0]) == 1:
        return np.asarray(array[0])
    return array


def _single_field(value: Any, *, name: str) -> np.ndarray:
    array = _single_array(value, name=name)
    if array.ndim != 3:
        raise ContractError(f"{name} must have shape (C,H,W) or (1,C,H,W), got {array.shape}")
    if int(array.shape[0]) not in {4, 5}:
        raise ContractError(f"{name} must have 4 or 5 channels, got {array.shape}")
    return np.asarray(array, dtype=np.float64)


def _single_coords(value: Any, *, name: str) -> np.ndarray:
    array = _single_array(value, name=name)
    if array.ndim != 3:
        raise ContractError(f"{name} must have shape (C,H,W) or (1,C,H,W), got {array.shape}")
    return np.asarray(array, dtype=np.float64)


def _single_flow_tuple(value: Any) -> tuple[float, ...]:
    array = _single_array(value, name="flow_conditions").reshape(-1)
    if array.size == 0:
        raise ContractError("flow_conditions must not be empty")
    return tuple(float(item) for item in array.tolist())


def _case_dir_from_cgns_basename(cgns_basename: str) -> str:
    suffix = "_000_vol.cgns"
    text = str(cgns_basename).strip()
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return Path(text).stem


def _normalize_cgns_location(cgns_root: str | Path, cgns_basename: str) -> tuple[str, str]:
    root_text = str(cgns_root).strip()
    basename_text = str(cgns_basename).strip()
    if not root_text or not basename_text:
        return root_text, basename_text

    basename_path = Path(basename_text)
    if basename_path.is_absolute():
        return root_text, basename_text

    direct_path = Path(root_text) / basename_path
    if direct_path.exists():
        return root_text, basename_text

    nested_rel = Path(_case_dir_from_cgns_basename(basename_text)) / basename_path.name
    nested_path = Path(root_text) / nested_rel
    if nested_path.exists():
        return root_text, str(nested_rel)

    return root_text, basename_text


def _predictor_kind(value: str | None, *, config_family: str | None = None) -> str:
    text = str(value or config_family or "").strip().lower()
    if text not in {"direct", "fsb"}:
        raise ContractError("predictor_kind must be one of: direct, fsb")
    return text


def _required_batch_value(batch: Mapping[str, Any], key: str) -> Any:
    if key not in batch:
        raise ContractError(f"FSB ordinal batch is missing {key!r}")
    return batch[key]


def _first_text(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return str(value.item())
        return str(value.tolist())
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return str(value[0])
    return str(value)


def _plain_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _batch_metadata(batch: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "cgns_basename",
        "source_name",
        "source_kind",
        "source_chunk",
        "source_index_path",
        "source_shard_root",
        "source_shard_path",
        "index",
    )
    return {
        key: _plain_value(batch[key])
        for key in keys
        if key in batch
    }


def _inverse_field_if_needed(field: Any, *, normalizer: Any) -> Any:
    if field is None:
        return None
    if normalizer is None:
        return field
    tensor = field if isinstance(field, torch.Tensor) else torch.as_tensor(field)
    with torch.no_grad():
        return normalizer.inverse_transform(tensor)


@dataclass(frozen=True)
class FinalOnlyOrdinalModelBatch:
    """Model-side tensors needed to build a final-only NK_resume case."""

    predictor_kind: str
    ordinal: int
    config_path: str
    index_path: str
    stats_path: str
    checkpoint_path: str
    geometry: Any
    flow_conditions: Any
    coords: Any
    coords_center: Any
    coords_vertex: Any
    target_field: Any | None = None
    cgns_basename: str = ""
    source_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.ordinal) < 0:
            raise ContractError("FinalOnlyOrdinalModelBatch.ordinal must be non-negative")
        object.__setattr__(self, "predictor_kind", _predictor_kind(self.predictor_kind))
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "config_path", _path_text(self.config_path))
        object.__setattr__(self, "index_path", _path_text(self.index_path))
        object.__setattr__(self, "stats_path", _path_text(self.stats_path))
        object.__setattr__(self, "checkpoint_path", _path_text(self.checkpoint_path))
        object.__setattr__(self, "source_info", _metadata(self.source_info))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def resume_request(
        self,
        *,
        initial_field: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ResumeRequest:
        """Build the small model-side prediction request for this ordinal."""

        return ResumeRequest(
            geometry=self.geometry,
            flow_conditions=self.flow_conditions,
            coords=self.coords,
            initial_field=initial_field,
            metadata={
                **self.metadata,
                **_metadata(metadata),
                "ordinal": self.ordinal,
                "predictor_kind": self.predictor_kind,
                "config_path": self.config_path,
                "index_path": self.index_path,
                "stats_path": self.stats_path,
                "checkpoint_path": self.checkpoint_path,
            },
        )

    def target_field_physical(self, *, normalizer: Any = None) -> np.ndarray | None:
        """Return the target field in physical units when present."""

        value = _inverse_field_if_needed(self.target_field, normalizer=normalizer)
        if value is None:
            return None
        return _single_field(value, name="target_field")


@dataclass(frozen=True)
class FinalOnlyCaseRequest:
    """Inputs for turning one model prediction into a canonical final-only case."""

    model_batch: FinalOnlyOrdinalModelBatch
    prediction: ResumePrediction
    predictor_kind: str | None = None
    case_id: str = ""
    state_name: str = "final"
    step_index: int | None = None
    cgns_root: str | Path = ""
    cgns_basename: str = ""
    flow_conditions_dict: Mapping[str, Any] | None = None
    target_normalizer: Any = None
    force_coefficients: Mapping[str, Any] | None = None
    residual_reference: Mapping[str, Any] | None = None
    output_dir: str | Path = ""
    options_version: int = 2
    l2conv: float = 1.0e-8
    ranks_per_case: int = 1
    mpi_launcher: str = "auto"
    mpi_omp_threads: int = 1
    device: str = ""
    inference_steps: int | None = None
    custom_timesteps: Iterable[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predictor_kind",
            _predictor_kind(self.predictor_kind, config_family=self.model_batch.predictor_kind),
        )
        state_name = str(self.state_name).strip().lower()
        if not state_name:
            raise ContractError("FinalOnlyCaseRequest.state_name is required")
        object.__setattr__(self, "state_name", state_name)
        if self.step_index is not None and int(self.step_index) < 0:
            raise ContractError("FinalOnlyCaseRequest.step_index must be non-negative")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


def _batch_from_config_sample(
    *,
    config_path: str | Path,
    ordinal: int,
    expected_family: str | None = None,
    index_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> FinalOnlyOrdinalModelBatch:
    resolved_config_path = _project_path(config_path, name="config_path")
    config = load_config(resolved_config_path)
    family = _predictor_kind(config.model.family)
    if expected_family is not None and family != _predictor_kind(expected_family):
        raise ContractError(
            f"ordinal collection requires model.family={expected_family!r}, got {config.model.family!r}"
        )

    resolved_index_path = _project_path(index_path or config.data.index_path, name="index_path")
    resolved_stats_path = _project_path(
        stats_path or config.data.stats_path or "turbulent_scale_stats.json",
        name="stats_path",
    )
    checkpoint_value = checkpoint_path or config.runtime.checkpoint
    checkpoint_text = (
        ""
        if checkpoint_value is None or not str(checkpoint_value).strip()
        else str(_project_path(checkpoint_value, name="checkpoint_path"))
    )

    dataset = H5MultiFieldDataset(
        index_path=str(resolved_index_path),
        normalize=bool(config.data.normalize),
        scale_turbulent=bool(config.data.scale_turbulent),
        turbulent_stats_file=str(resolved_stats_path),
        num_samples=None,
        use_geometry_orig=bool(config.data.use_geometry_orig),
    )
    sample = dataset[int(ordinal)]
    batch = dict(default_collate([sample]))
    dataset.close_all_handles()

    return FinalOnlyOrdinalModelBatch(
        predictor_kind=family,
        ordinal=int(ordinal),
        config_path=str(resolved_config_path),
        index_path=str(resolved_index_path),
        stats_path=str(resolved_stats_path),
        checkpoint_path=checkpoint_text,
        geometry=_required_batch_value(batch, "geometry"),
        flow_conditions=_required_batch_value(batch, "flow_conditions"),
        coords=_required_batch_value(batch, "coords_center"),
        coords_center=_required_batch_value(batch, "coords_center_pde"),
        coords_vertex=_required_batch_value(batch, "coords_vertex"),
        target_field=batch.get("fields"),
        cgns_basename=_first_text(batch.get("cgns_basename", "")),
        source_info=_batch_metadata(batch),
        metadata={
            "collector": "load_finalonly_ordinal_model_batch",
            "normalize": bool(config.data.normalize),
            "scale_turbulent": bool(config.data.scale_turbulent),
            "use_geometry_orig": bool(config.data.use_geometry_orig),
        },
    )


@dataclass(frozen=True)
class FSBOrdinalModelBatch:
    """One collated model batch selected by dataset ordinal."""

    ordinal: int
    config_path: str
    index_path: str
    stats_path: str
    checkpoint_path: str
    geometry: Any
    flow_conditions: Any
    coords: Any
    coords_center: Any
    coords_vertex: Any
    target_field: Any | None = None
    cgns_basename: str = ""
    source_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.ordinal) < 0:
            raise ContractError("FSBOrdinalModelBatch.ordinal must be non-negative")
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "config_path", _path_text(self.config_path))
        object.__setattr__(self, "index_path", _path_text(self.index_path))
        object.__setattr__(self, "stats_path", _path_text(self.stats_path))
        object.__setattr__(self, "checkpoint_path", _path_text(self.checkpoint_path))
        object.__setattr__(self, "source_info", _metadata(self.source_info))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def target_field_physical(self, *, normalizer: Any) -> np.ndarray | None:
        """Return the target field in physical units when present."""

        value = _inverse_field_if_needed(self.target_field, normalizer=normalizer)
        if value is None:
            return None
        return _tensor_to_numpy(value, name="target_field")


def finalonly_batch_from_fsb_batch(batch: FSBOrdinalModelBatch) -> FinalOnlyOrdinalModelBatch:
    """View an existing FSB ordinal batch as a final-only model batch."""

    return FinalOnlyOrdinalModelBatch(
        predictor_kind="fsb",
        ordinal=batch.ordinal,
        config_path=batch.config_path,
        index_path=batch.index_path,
        stats_path=batch.stats_path,
        checkpoint_path=batch.checkpoint_path,
        geometry=batch.geometry,
        flow_conditions=batch.flow_conditions,
        coords=batch.coords,
        coords_center=batch.coords_center,
        coords_vertex=batch.coords_vertex,
        target_field=batch.target_field,
        cgns_basename=batch.cgns_basename,
        source_info=batch.source_info,
        metadata=batch.metadata,
    )


def build_finalonly_case_from_prediction(request: FinalOnlyCaseRequest) -> ResumeCase:
    """Build a canonical final-only `ResumeCase` from one model prediction."""

    batch = request.model_batch
    predictor_kind = _predictor_kind(request.predictor_kind, config_family=batch.predictor_kind)
    prediction_field = _single_field(request.prediction.fields, name="prediction.fields")
    target_field = batch.target_field_physical(normalizer=request.target_normalizer)
    cgns_basename = str(request.cgns_basename or batch.cgns_basename).strip()
    if not cgns_basename:
        raise ContractError("final-only case collection requires cgns_basename")
    cgns_root, cgns_basename = _normalize_cgns_location(request.cgns_root, cgns_basename)
    case_id = str(request.case_id).strip()
    if not case_id:
        case_id = f"nk_resume_ordinal{batch.ordinal:04d}_{predictor_kind}_finalonly"
    builder = build_direct_case if predictor_kind == "direct" else build_fsb_case
    return builder(
        case_id=case_id,
        cgns_basename=cgns_basename,
        prediction_field=prediction_field,
        state_name=request.state_name,
        step_index=request.step_index,
        cgns_root=cgns_root,
        flow_conditions=_single_flow_tuple(batch.flow_conditions),
        flow_conditions_dict=_metadata(request.flow_conditions_dict),
        source_info={
            **batch.source_info,
            "collector": "build_finalonly_case_from_prediction",
        },
        options_version=request.options_version,
        l2conv=request.l2conv,
        ranks_per_case=request.ranks_per_case,
        mpi_launcher=request.mpi_launcher,
        mpi_omp_threads=request.mpi_omp_threads,
        ground_truth_field=target_field,
        coords_center=_single_coords(batch.coords_center, name="coords_center"),
        coords_vertex=_single_coords(batch.coords_vertex, name="coords_vertex"),
        force_coefficients=_metadata(request.force_coefficients),
        residual_reference=_metadata(request.residual_reference),
        output_dir=request.output_dir,
        ordinal=batch.ordinal,
        dataset_index=batch.ordinal,
        config_path=batch.config_path,
        checkpoint_path=batch.checkpoint_path,
        stats_path=batch.stats_path,
        device=request.device,
        inference_steps=request.inference_steps,
        custom_timesteps=tuple(int(value) for value in request.custom_timesteps or ()),
        model_metadata={
            **batch.metadata,
            "collector": "build_finalonly_case_from_prediction",
        },
        prediction_metadata={
            **_metadata(request.prediction.metadata),
            "collector": "build_finalonly_case_from_prediction",
            "predictor_kind": predictor_kind,
        },
        runtime_metadata={
            **request.metadata,
            "collector": "build_finalonly_case_from_prediction",
        },
    )


def collect_finalonly_case_from_batch(
    *,
    model_batch: FinalOnlyOrdinalModelBatch,
    predictor: Any,
    initial_field: Any = None,
    case_id: str = "",
    cgns_root: str | Path = "",
    cgns_basename: str = "",
    target_normalizer: Any = None,
    metadata: Mapping[str, Any] | None = None,
    **case_fields: Any,
) -> ResumeCase:
    """Run a model-side predictor adapter for one batch and build a final-only case."""

    predict = getattr(predictor, "predict", None)
    if not callable(predict):
        raise ContractError("predictor must expose predict(request)")
    request = model_batch.resume_request(
        initial_field=initial_field,
        metadata=metadata,
    )
    prediction = predict(request)
    if not isinstance(prediction, ResumePrediction):
        prediction = ResumePrediction(fields=prediction, metadata={})
    return build_finalonly_case_from_prediction(
        FinalOnlyCaseRequest(
            model_batch=model_batch,
            prediction=prediction,
            case_id=case_id,
            cgns_root=cgns_root,
            cgns_basename=cgns_basename,
            target_normalizer=target_normalizer,
            metadata=_metadata(metadata),
            **case_fields,
        )
    )


def load_fsb_ordinal_model_batch(
    *,
    config_path: str | Path,
    ordinal: int,
    index_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> FSBOrdinalModelBatch:
    """Load the model-side tensors for one FSB dataset ordinal."""

    batch = _batch_from_config_sample(
        config_path=config_path,
        ordinal=int(ordinal),
        expected_family="fsb",
        index_path=index_path,
        stats_path=stats_path,
        checkpoint_path=checkpoint_path,
    )
    return FSBOrdinalModelBatch(
        ordinal=batch.ordinal,
        config_path=batch.config_path,
        index_path=batch.index_path,
        stats_path=batch.stats_path,
        checkpoint_path=batch.checkpoint_path,
        geometry=batch.geometry,
        flow_conditions=batch.flow_conditions,
        coords=batch.coords,
        coords_center=batch.coords_center,
        coords_vertex=batch.coords_vertex,
        target_field=batch.target_field,
        cgns_basename=batch.cgns_basename,
        source_info=batch.source_info,
        metadata={
            **batch.metadata,
            "collector": "load_fsb_ordinal_model_batch",
        },
    )


def load_finalonly_ordinal_model_batch(
    *,
    config_path: str | Path,
    ordinal: int,
    predictor_kind: str | None = None,
    index_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> FinalOnlyOrdinalModelBatch:
    """Load model-side tensors for one direct/FSB final-only dataset ordinal."""

    return _batch_from_config_sample(
        config_path=config_path,
        ordinal=int(ordinal),
        expected_family=predictor_kind,
        index_path=index_path,
        stats_path=stats_path,
        checkpoint_path=checkpoint_path,
    )


__all__ = [
    "FinalOnlyCaseRequest",
    "FinalOnlyOrdinalModelBatch",
    "FSBOrdinalModelBatch",
    "build_finalonly_case_from_prediction",
    "collect_finalonly_case_from_batch",
    "finalonly_batch_from_fsb_batch",
    "load_finalonly_ordinal_model_batch",
    "load_fsb_ordinal_model_batch",
]
