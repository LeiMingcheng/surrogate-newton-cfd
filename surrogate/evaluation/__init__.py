"""Evaluation helpers for canonical surrogate predictions."""

from surrogate.evaluation.metrics import (
    FieldMetricsConfig,
    compute_field_metrics,
)
from surrogate.evaluation.options import build_validation_options_from_config
from surrogate.evaluation.physics import (
    ForceMetricsConfig,
    ResidualMetricsConfig,
    compute_force_coefficients_from_fields,
    compute_force_metrics,
    compute_residual_batch_evaluation,
    compute_residual_batch_metrics,
    compute_residual_pair_metrics,
)
from surrogate.evaluation.runners import (
    ValidationOptions,
    ValidationResult,
    evaluate_prediction_batches,
)
from surrogate.evaluation.reports import (
    EvaluationReport,
    load_evaluation_report,
    save_benchmark_summary,
    save_evaluation_report,
    save_sample_records_csv,
    summarize_reports,
)
from surrogate.evaluation.direct_validation import (
    DirectValidationRunner,
    create_direct_validation_runner,
)
from surrogate.evaluation.fsb_validation import (
    FSBValidationRunner,
    create_fsb_validation_runner,
)

__all__ = [
    "DirectValidationRunner",
    "EvaluationReport",
    "FieldMetricsConfig",
    "FSBValidationRunner",
    "ForceMetricsConfig",
    "ResidualMetricsConfig",
    "ValidationOptions",
    "ValidationResult",
    "build_validation_options_from_config",
    "create_direct_validation_runner",
    "create_fsb_validation_runner",
    "compute_field_metrics",
    "compute_force_coefficients_from_fields",
    "compute_force_metrics",
    "compute_residual_batch_evaluation",
    "compute_residual_batch_metrics",
    "compute_residual_pair_metrics",
    "evaluate_prediction_batches",
    "load_evaluation_report",
    "save_benchmark_summary",
    "save_evaluation_report",
    "save_sample_records_csv",
    "summarize_reports",
]
