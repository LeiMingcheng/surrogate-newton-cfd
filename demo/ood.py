"""Validated offline geometry-distance index for the interactive demo."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from surrogate.utils.cst import cst26_to_shape_embedding

OOD_ASSET_FILENAMES = ("cst26_coefficients.npz", "ood_scores.csv")
OOD_DEFINITION = (
    "5-neighbour surface-RMS distance against offline training geometries"
)
_REQUIRED_SCORE_COLUMNS = {"split", "geometry_key", "ood_k5"}
_NEIGHBOUR_COUNT = 5
_OOD_PERCENTILE_THRESHOLD = 0.99


@dataclass(frozen=True)
class OodGeometryIndex:
    """In-memory training-geometry index loaded from a frozen external bundle."""

    embeddings: np.ndarray
    score_distribution: np.ndarray
    geometry_count: int
    training_count: int

    @classmethod
    def from_asset_root(cls, asset_root: Path | str) -> "OodGeometryIndex":
        root = Path(asset_root).expanduser()
        coefficient_path = root / OOD_ASSET_FILENAMES[0]
        score_path = root / OOD_ASSET_FILENAMES[1]
        for path in (coefficient_path, score_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing OOD geometry asset: {path.name}")

        try:
            with np.load(coefficient_path, allow_pickle=False) as data:
                if not {"geometry_key", "cst26"}.issubset(data.files):
                    raise ValueError(
                        "cst26_coefficients.npz must contain geometry_key and cst26 arrays"
                    )
                keys = np.asarray(data["geometry_key"]).astype(str)
                cst26 = np.asarray(data["cst26"], dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise ValueError("Unable to read cst26_coefficients.npz") from exc

        if keys.ndim != 1:
            raise ValueError("geometry_key must be a one-dimensional array")
        if cst26.shape != (keys.shape[0], 26):
            raise ValueError("cst26 must have shape (N, 26) matching geometry_key")
        if keys.shape[0] < _NEIGHBOUR_COUNT:
            raise ValueError("The OOD asset needs at least five source geometries")
        if any(not key for key in keys):
            raise ValueError("geometry_key entries must be non-empty")
        if len(set(keys.tolist())) != keys.shape[0]:
            raise ValueError("geometry_key entries must be unique")
        if not np.isfinite(cst26).all():
            raise ValueError("cst26 coefficients must be finite")

        key_to_index = {key: index for index, key in enumerate(keys)}
        train_indices: list[int] = []
        train_scores: list[float] = []
        seen_train_keys: set[str] = set()
        with score_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not _REQUIRED_SCORE_COLUMNS.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    "ood_scores.csv must contain split, geometry_key, and ood_k5 columns"
                )
            for row_number, row in enumerate(reader, start=2):
                if row["split"] != "train":
                    continue
                key = row["geometry_key"]
                if key in seen_train_keys:
                    raise ValueError(f"Duplicate training geometry_key on CSV row {row_number}")
                seen_train_keys.add(key)
                if key not in key_to_index:
                    raise ValueError(
                        f"Training geometry_key on CSV row {row_number} is missing from the NPZ"
                    )
                try:
                    score = float(row["ood_k5"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid ood_k5 value on CSV row {row_number}") from exc
                if not np.isfinite(score) or score < 0.0:
                    raise ValueError(
                        f"ood_k5 must be finite and non-negative on CSV row {row_number}"
                    )
                train_indices.append(key_to_index[key])
                train_scores.append(score)

        if len(train_indices) < _NEIGHBOUR_COUNT:
            raise ValueError("ood_scores.csv needs at least five matching training rows")

        x_fixed = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 201)))
        embeddings = cst26_to_shape_embedding(
            cst26[np.asarray(train_indices, dtype=np.int64)], x_fixed
        )
        distribution = np.sort(np.asarray(train_scores, dtype=np.float64))
        if not np.isfinite(embeddings).all():
            raise ValueError("The computed training geometry embeddings must be finite")
        return cls(
            embeddings=np.asarray(embeddings, dtype=np.float64),
            score_distribution=distribution,
            geometry_count=int(keys.shape[0]),
            training_count=len(train_indices),
        )

    def score(self, geometry: np.ndarray) -> dict[str, Any]:
        values = np.asarray(geometry, dtype=np.float64)
        if values.shape == (27,):
            cst26 = values[:26]
        elif values.shape == (26,):
            cst26 = values
        else:
            raise ValueError("OOD geometry must have shape (26,) or (27,)")
        if not np.isfinite(cst26).all():
            raise ValueError("OOD geometry coefficients must be finite")

        x_fixed = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 201)))
        query = cst26_to_shape_embedding(cst26, x_fixed)
        distances = np.sqrt(np.sum((self.embeddings - query) ** 2, axis=1))
        distance_k5 = float(
            np.mean(np.partition(distances, _NEIGHBOUR_COUNT - 1)[:_NEIGHBOUR_COUNT])
        )
        percentile = float(
            np.searchsorted(self.score_distribution, distance_k5, side="right")
        ) / float(self.training_count)
        label = "OOD" if percentile >= _OOD_PERCENTILE_THRESHOLD else "ID"
        return {
            "label": label,
            "is_ood": label == "OOD",
            "percentile": percentile,
            "distance_k5": distance_k5,
            "distance_units": "chord-normalized combined surface RMS",
            "threshold_percentile": _OOD_PERCENTILE_THRESHOLD,
            "scope": "geometry-only neighbourhood warning",
            "definition": OOD_DEFINITION,
        }

    def summary(self) -> dict[str, int | float | str]:
        return {
            "geometry_count": self.geometry_count,
            "training_count": self.training_count,
            "neighbour_count": _NEIGHBOUR_COUNT,
            "threshold_percentile": _OOD_PERCENTILE_THRESHOLD,
            "score_min": float(self.score_distribution[0]),
            "score_max": float(self.score_distribution[-1]),
            "definition": OOD_DEFINITION,
        }
