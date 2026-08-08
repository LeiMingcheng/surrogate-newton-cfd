from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "datasets"
BENCHMARK_SOURCE_ROOT = DATA_ROOT / "benchmark" / "source"


def resolve_benchmark_metadata_file(filename: str) -> Path:
    candidate = DATA_ROOT / "benchmark" / "metadata" / filename
    return candidate.resolve() if candidate.exists() else candidate


def resolve_benchmark_source_root() -> Path:
    candidate = BENCHMARK_SOURCE_ROOT
    return candidate.resolve() if candidate.exists() else candidate


def resolve_benchmark_source_dat_dir() -> Path:
    candidate = BENCHMARK_SOURCE_ROOT / "airfoil_dat"
    return candidate.resolve() if candidate.exists() else candidate


def resolve_benchmark_source_mesh_dir() -> Path:
    candidate = BENCHMARK_SOURCE_ROOT / "mesh"
    return candidate.resolve() if candidate.exists() else candidate


def resolve_benchmark_source_case_root() -> Path:
    candidate = BENCHMARK_SOURCE_ROOT / "flowfield_sa"
    return candidate.resolve() if candidate.exists() else candidate


def resolve_benchmark_source_index_file(filename: str = "index.csv") -> Path:
    candidate = BENCHMARK_SOURCE_ROOT / "shards_sa" / filename
    return candidate.resolve() if candidate.exists() else candidate


def resolve_benchmark_cgns_root() -> Path:
    candidate = DATA_ROOT / "benchmark" / "flowfield_sa"
    return candidate.resolve() if candidate.exists() else candidate


__all__ = [
    "resolve_benchmark_cgns_root",
    "resolve_benchmark_metadata_file",
    "resolve_benchmark_source_case_root",
    "resolve_benchmark_source_dat_dir",
    "resolve_benchmark_source_index_file",
    "resolve_benchmark_source_mesh_dir",
    "resolve_benchmark_source_root",
]
