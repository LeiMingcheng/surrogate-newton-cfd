"""Real-backend final-only collection helpers for clean NK_resume cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from NK_resume import ContractError, ResumeCase
from surrogate.inference.backends import DirectPredictorBackend, FSBPredictorBackend
from surrogate.inference.contracts import DirectPredictorConfig, FSBPredictorConfig
from surrogate.nk_resume.adapters import DirectResumePredictorAdapter, FSBResumePredictorAdapter
from surrogate.nk_resume.collectors import (
    FinalOnlyOrdinalModelBatch,
    collect_finalonly_case_from_batch,
    load_finalonly_ordinal_model_batch,
)


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): item for key, item in dict(value or {}).items()}


def _path_or_none(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_tuple(values: Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class FinalOnlyBackendCaseRequest:
    """Request for collecting one final-only case through a real predictor backend."""

    config_path: str | Path
    ordinal: int
    index_path: str | Path | None = None
    stats_path: str | Path | None = None
    checkpoint_path: str | Path | None = None
    predictor_kind: str | None = None
    device: str = "cuda"
    use_ema: bool = True
    n_inference_steps: int | None = None
    custom_timesteps: tuple[int, ...] = ()
    eta: float = 0.0
    noise_mode: str = "zeros"
    initial_field: Any = None
    case_id: str = ""
    cgns_root: str | Path = ""
    cgns_basename: str = ""
    output_dir: str | Path = ""
    options_version: int = 2
    l2conv: float = 1.0e-8
    ranks_per_case: int = 1
    mpi_launcher: str = "auto"
    mpi_omp_threads: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.ordinal) < 0:
            raise ContractError("FinalOnlyBackendCaseRequest.ordinal must be non-negative")
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "device", str(self.device))
        object.__setattr__(self, "custom_timesteps", _int_tuple(self.custom_timesteps))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


def _predictor_for_batch(
    batch: FinalOnlyOrdinalModelBatch,
    request: FinalOnlyBackendCaseRequest,
) -> tuple[Any, Any]:
    checkpoint = _path_or_none(request.checkpoint_path) or _path_or_none(batch.checkpoint_path)
    if batch.predictor_kind == "direct":
        backend = DirectPredictorBackend(
            DirectPredictorConfig(
                config_path=batch.config_path,
                checkpoint_path=checkpoint,
                device=request.device,
                use_ema=bool(request.use_ema),
            )
        )
        return DirectResumePredictorAdapter(backend), backend
    if batch.predictor_kind == "fsb":
        backend = FSBPredictorBackend(
            FSBPredictorConfig(
                config_path=batch.config_path,
                checkpoint_path=checkpoint,
                device=request.device,
                use_ema=bool(request.use_ema),
                n_inference_steps=request.n_inference_steps,
                custom_timesteps=list(request.custom_timesteps) if request.custom_timesteps else None,
                eta=float(request.eta),
                noise_mode=str(request.noise_mode),
            )
        )
        return FSBResumePredictorAdapter(backend), backend
    raise ContractError(f"Unsupported predictor_kind: {batch.predictor_kind!r}")


def collect_finalonly_case_from_config(
    request: FinalOnlyBackendCaseRequest,
) -> ResumeCase:
    """Load model/data context, run prediction, and build one canonical ResumeCase."""

    batch = load_finalonly_ordinal_model_batch(
        config_path=request.config_path,
        ordinal=request.ordinal,
        predictor_kind=request.predictor_kind,
        index_path=request.index_path,
        stats_path=request.stats_path,
        checkpoint_path=request.checkpoint_path,
    )
    predictor, backend = _predictor_for_batch(batch, request)
    return _collect_case_from_batch_and_backend(batch, request, predictor, backend)


def _collect_case_from_batch_and_backend(
    batch: FinalOnlyOrdinalModelBatch,
    request: FinalOnlyBackendCaseRequest,
    predictor: Any,
    backend: Any,
) -> ResumeCase:
    """Build one case from a loaded batch and already-instantiated backend."""

    return collect_finalonly_case_from_batch(
        model_batch=batch,
        predictor=predictor,
        initial_field=request.initial_field,
        case_id=request.case_id,
        cgns_root=request.cgns_root,
        cgns_basename=request.cgns_basename,
        target_normalizer=getattr(backend, "normalizer", None),
        metadata={
            **request.metadata,
            "collector": "collect_finalonly_case_from_config",
            "device": request.device,
            "use_ema": bool(request.use_ema),
        },
        options_version=request.options_version,
        l2conv=request.l2conv,
        ranks_per_case=request.ranks_per_case,
        mpi_launcher=request.mpi_launcher,
        mpi_omp_threads=request.mpi_omp_threads,
        device=request.device,
        inference_steps=request.n_inference_steps,
        custom_timesteps=request.custom_timesteps,
        output_dir=request.output_dir,
    )


def collect_finalonly_cases_from_config(
    *,
    config_path: str | Path,
    ordinals: Iterable[int],
    **kwargs: Any,
) -> list[ResumeCase]:
    """Collect final-only cases for multiple ordinals through one request shape."""

    cases: list[ResumeCase] = []
    predictor = None
    backend = None
    predictor_kind = ""
    for ordinal in ordinals:
        request = FinalOnlyBackendCaseRequest(
            config_path=config_path,
            ordinal=int(ordinal),
            **kwargs,
        )
        batch = load_finalonly_ordinal_model_batch(
            config_path=request.config_path,
            ordinal=request.ordinal,
            predictor_kind=request.predictor_kind,
            index_path=request.index_path,
            stats_path=request.stats_path,
            checkpoint_path=request.checkpoint_path,
        )
        if predictor is None or backend is None:
            predictor, backend = _predictor_for_batch(batch, request)
            predictor_kind = batch.predictor_kind
        elif batch.predictor_kind != predictor_kind:
            raise ContractError(
                "collect_finalonly_cases_from_config requires one predictor_kind per call"
            )
        cases.append(_collect_case_from_batch_and_backend(batch, request, predictor, backend))
    return cases


__all__ = [
    "FinalOnlyBackendCaseRequest",
    "collect_finalonly_case_from_config",
    "collect_finalonly_cases_from_config",
]
