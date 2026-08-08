"""Report serialization and benchmark summaries for validation results."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

from surrogate.evaluation.runners import ValidationResult


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


@dataclass
class EvaluationReport:
    """Serializable validation report."""

    result: ValidationResult
    model_family: str
    model_key: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "model_family": str(self.model_family),
            "model_key": str(self.model_key),
            "n_samples": int(self.result.n_samples),
            "n_batches": int(self.result.n_batches),
            "metrics": {key: float(value) for key, value in self.result.metrics.items()},
            "sample_records": _to_jsonable(self.result.sample_records),
            "metadata": _to_jsonable(dict(self.metadata)),
        }

    def save(self, output_dir: str | Path, *, stem: str = "evaluation") -> dict[str, Path]:
        """Save JSON and metrics CSV files under output_dir."""
        return save_evaluation_report(self, output_dir, stem=stem)


def save_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON payload with deterministic formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_to_jsonable(dict(payload)), file, indent=2, sort_keys=True)
    return output_path


def save_metrics_csv(metrics: Mapping[str, float], path: str | Path) -> Path:
    """Write metric name/value rows."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, float(metrics[key])])
    return output_path


def _csv_cell(value: Any) -> Any:
    value = _to_jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def save_sample_records_csv(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write per-sample metric records to CSV."""
    record_list = [dict(record) for record in records]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in record_list for key in record.keys()})
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in record_list:
            writer.writerow({key: _csv_cell(record.get(key, "")) for key in fieldnames})
    return output_path


def save_evaluation_report(
    report: EvaluationReport,
    output_dir: str | Path,
    *,
    stem: str = "evaluation",
) -> dict[str, Path]:
    """Save a validation report as JSON and metrics CSV."""
    output_path = Path(output_dir)
    payload = report.to_dict()
    json_path = save_json(payload, output_path / f"{stem}.json")
    metrics_path = save_metrics_csv(payload["metrics"], output_path / f"{stem}_metrics.csv")
    outputs = {"json": json_path, "metrics_csv": metrics_path}
    if payload.get("sample_records"):
        outputs["samples_csv"] = save_sample_records_csv(
            payload["sample_records"],
            output_path / f"{stem}_samples.csv",
        )
    return outputs


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON evaluation report as a plain dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if "metrics" not in payload:
        raise ValueError(f"Evaluation report missing metrics: {path}")
    return payload


def _report_to_dict(report: EvaluationReport | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(report, EvaluationReport):
        return report.to_dict()
    return dict(report)


def summarize_reports(
    reports: Iterable[EvaluationReport | Mapping[str, Any]],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Aggregate metric statistics across validation reports."""
    report_dicts = [_report_to_dict(report) for report in reports]
    if not report_dicts:
        raise ValueError("summarize_reports requires at least one report")

    metric_names = sorted({
        metric_name
        for report in report_dicts
        for metric_name in dict(report.get("metrics", {})).keys()
    })
    metrics_summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [
            float(report["metrics"][metric_name])
            for report in report_dicts
            if metric_name in report.get("metrics", {})
        ]
        array = np.asarray(values, dtype=np.float64)
        metrics_summary[metric_name] = {
            "count": float(array.size),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return {
        "created_at": _utc_now_iso(),
        "report_count": len(report_dicts),
        "metrics": metrics_summary,
        "reports": report_dicts,
        "metadata": _to_jsonable(dict(metadata or {})),
    }


def save_benchmark_summary(
    reports: Iterable[EvaluationReport | Mapping[str, Any]],
    output_dir: str | Path,
    *,
    stem: str = "benchmark_summary",
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Path]:
    """Save aggregate benchmark summary JSON and flat CSV."""
    summary = summarize_reports(reports, metadata=metadata)
    output_path = Path(output_dir)
    json_path = save_json(summary, output_path / f"{stem}.json")

    csv_path = output_path / f"{stem}_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "count", "mean", "std", "min", "max"])
        for metric_name, stats in sorted(summary["metrics"].items()):
            writer.writerow([
                metric_name,
                int(stats["count"]),
                stats["mean"],
                stats["std"],
                stats["min"],
                stats["max"],
            ])
    return {"json": json_path, "metrics_csv": csv_path}


__all__ = [
    "EvaluationReport",
    "load_evaluation_report",
    "save_benchmark_summary",
    "save_evaluation_report",
    "save_json",
    "save_metrics_csv",
    "save_sample_records_csv",
    "summarize_reports",
]
