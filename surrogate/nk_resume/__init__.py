"""Model-side integration for final-only and alternating NK resume."""

from __future__ import annotations

from typing import Any


__all__ = [
    "ALTERNATING_FSB_NK_EXPERIMENT_SUMMARY_SCHEMA",
    "AlternatingFSBNKCaseArtifact",
    "AlternatingFSBNKExperimentRequest",
    "AlternatingFSBNKExperimentResult",
    "AlternatingFSBNKStageArtifact",
    "DirectResumePredictorAdapter",
    "FSBAlternatingSchedulerState",
    "FSBOrdinalModelBatch",
    "FSBResumePredictorAdapter",
    "FinalOnlyBackendCaseRequest",
    "FinalOnlyCaseRequest",
    "FinalOnlyExperimentRequest",
    "FinalOnlyExperimentResult",
    "FinalOnlyOrdinalModelBatch",
    "ResumePrediction",
    "ResumeRequest",
    "build_finalonly_case_from_prediction",
    "collect_finalonly_case_from_batch",
    "collect_finalonly_case_from_config",
    "collect_finalonly_cases_from_config",
    "finalonly_batch_from_fsb_batch",
    "load_alternating_scheduler_state",
    "load_finalonly_ordinal_model_batch",
    "load_fsb_ordinal_model_batch",
    "load_geometry_bundle",
    "load_payload",
    "run_alternating_fsb_nk_experiment",
    "run_finalonly_experiment",
    "write_alternating_scheduler_state",
]


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ALTERNATING_FSB_NK_EXPERIMENT_SUMMARY_SCHEMA": (
        "surrogate.nk_resume.alternating",
        "ALTERNATING_FSB_NK_EXPERIMENT_SUMMARY_SCHEMA",
    ),
    "AlternatingFSBNKCaseArtifact": (
        "surrogate.nk_resume.alternating",
        "AlternatingFSBNKCaseArtifact",
    ),
    "AlternatingFSBNKExperimentRequest": (
        "surrogate.nk_resume.alternating",
        "AlternatingFSBNKExperimentRequest",
    ),
    "AlternatingFSBNKExperimentResult": (
        "surrogate.nk_resume.alternating",
        "AlternatingFSBNKExperimentResult",
    ),
    "AlternatingFSBNKStageArtifact": (
        "surrogate.nk_resume.alternating",
        "AlternatingFSBNKStageArtifact",
    ),
    "DirectResumePredictorAdapter": (
        "surrogate.nk_resume.adapters",
        "DirectResumePredictorAdapter",
    ),
    "FSBResumePredictorAdapter": (
        "surrogate.nk_resume.adapters",
        "FSBResumePredictorAdapter",
    ),
    "FSBAlternatingSchedulerState": (
        "surrogate.nk_resume.alternating_state",
        "FSBAlternatingSchedulerState",
    ),
    "FinalOnlyCaseRequest": (
        "surrogate.nk_resume.collectors",
        "FinalOnlyCaseRequest",
    ),
    "FinalOnlyBackendCaseRequest": (
        "surrogate.nk_resume.finalonly",
        "FinalOnlyBackendCaseRequest",
    ),
    "FinalOnlyExperimentRequest": (
        "surrogate.nk_resume.experiment",
        "FinalOnlyExperimentRequest",
    ),
    "FinalOnlyExperimentResult": (
        "surrogate.nk_resume.experiment",
        "FinalOnlyExperimentResult",
    ),
    "FinalOnlyOrdinalModelBatch": (
        "surrogate.nk_resume.collectors",
        "FinalOnlyOrdinalModelBatch",
    ),
    "FSBOrdinalModelBatch": (
        "surrogate.nk_resume.collectors",
        "FSBOrdinalModelBatch",
    ),
    "ResumePrediction": (
        "surrogate.nk_resume.contracts",
        "ResumePrediction",
    ),
    "ResumeRequest": (
        "surrogate.nk_resume.contracts",
        "ResumeRequest",
    ),
    "load_geometry_bundle": (
        "surrogate.nk_resume.payloads",
        "load_geometry_bundle",
    ),
    "load_payload": (
        "surrogate.nk_resume.payloads",
        "load_payload",
    ),
    "build_finalonly_case_from_prediction": (
        "surrogate.nk_resume.collectors",
        "build_finalonly_case_from_prediction",
    ),
    "collect_finalonly_case_from_batch": (
        "surrogate.nk_resume.collectors",
        "collect_finalonly_case_from_batch",
    ),
    "collect_finalonly_case_from_config": (
        "surrogate.nk_resume.finalonly",
        "collect_finalonly_case_from_config",
    ),
    "collect_finalonly_cases_from_config": (
        "surrogate.nk_resume.finalonly",
        "collect_finalonly_cases_from_config",
    ),
    "finalonly_batch_from_fsb_batch": (
        "surrogate.nk_resume.collectors",
        "finalonly_batch_from_fsb_batch",
    ),
    "load_alternating_scheduler_state": (
        "surrogate.nk_resume.alternating_state",
        "load_alternating_scheduler_state",
    ),
    "load_finalonly_ordinal_model_batch": (
        "surrogate.nk_resume.collectors",
        "load_finalonly_ordinal_model_batch",
    ),
    "load_fsb_ordinal_model_batch": (
        "surrogate.nk_resume.collectors",
        "load_fsb_ordinal_model_batch",
    ),
    "run_finalonly_experiment": (
        "surrogate.nk_resume.experiment",
        "run_finalonly_experiment",
    ),
    "run_alternating_fsb_nk_experiment": (
        "surrogate.nk_resume.alternating",
        "run_alternating_fsb_nk_experiment",
    ),
    "write_alternating_scheduler_state": (
        "surrogate.nk_resume.alternating_state",
        "write_alternating_scheduler_state",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
