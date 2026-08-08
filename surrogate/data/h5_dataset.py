"""HDF5 multi-field dataset for direct and FSB workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd
import torch

from surrogate.data.base import BaseFieldDataset
from surrogate.data.normalizers import create_normalizer

REPO_ROOT = Path(__file__).resolve().parents[2]


class H5MultiFieldDataset(BaseFieldDataset):
    """HDF5 dataset returning flow fields, O-grid coordinates, geometry, and flow conditions."""

    def __init__(
        self,
        index_path: str,
        normalize: bool = False,
        scale_turbulent: bool = False,
        turbulent_stats_file: str = "turbulent_scale_stats.json",
        num_samples: Optional[int] = None,
        cache_size_mb: int = 256,
        preserve_coord_precision: bool = False,
        use_geometry_orig: bool = False,
    ) -> None:
        super().__init__()
        del preserve_coord_precision

        self.index_path = Path(index_path)
        self.shard_dir = self.index_path.parent
        self.cache_size_mb = int(cache_size_mb)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_geometry_orig = bool(use_geometry_orig)
        self.normalize = bool(normalize)
        self.scale_turbulent = bool(scale_turbulent)

        try:
            self.index_df = pd.read_csv(index_path, low_memory=False)
            if num_samples is not None:
                self.index_df = self.index_df.iloc[: int(num_samples)].reset_index(drop=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load index file {index_path}: {exc}") from exc

        self._load_metadata()
        self._h5_handles: dict[str, h5py.File] = {}
        self._worker_id: Optional[int] = None

        self.normalizer = None
        if self.scale_turbulent or self.normalize:
            stats_path = self._resolve_stats_path(turbulent_stats_file)
            self.normalizer = create_normalizer(
                stats_path=stats_path,
                scale_turbulent=self.scale_turbulent,
                normalize=self.normalize,
            )

    @staticmethod
    def _normalize_optional_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        return str(value)

    def _resolve_stats_path(self, turbulent_stats_file: str) -> Path:
        stats_path = Path(turbulent_stats_file)
        if stats_path.is_absolute():
            return stats_path

        candidates = [
            self.shard_dir / stats_path,
            self.index_path.parent / stats_path,
            Path.cwd() / stats_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return self.shard_dir / stats_path

    def _resolve_row_shard_path(self, row: pd.Series) -> str:
        shard_path = str(row["shard_path"])
        full_path = self.shard_dir / shard_path
        if full_path.exists():
            return shard_path

        source_kind = self._normalize_optional_text(row.get("source_kind"))
        source_shard_root = self._normalize_optional_text(row.get("source_shard_root"))
        source_shard_path = self._normalize_optional_text(row.get("source_shard_path"))
        source_chunk = self._normalize_optional_text(row.get("source_chunk"))

        candidate = None
        if source_shard_root and source_shard_path:
            candidate = (Path(source_shard_root) / source_shard_path).resolve()
        elif source_kind == "laminar" and source_shard_path:
            candidate = (REPO_ROOT / "data" / "source" / "laminar" / source_shard_path).resolve()
        elif source_kind in {"supercritical_v1", "supercritical_legacy", "supercritical_archive"} and source_shard_path:
            candidate = (
                REPO_ROOT / "data" / "source" / "supercritical_legacy" / source_shard_path
            ).resolve()
        elif source_kind in {"supercritical_production", "supercritical_current"} and source_shard_path and source_chunk:
            candidate = (
                REPO_ROOT / "data" / "source" / "supercritical" / source_chunk / "shards_sa" / source_shard_path
            ).resolve()

        if candidate is not None and candidate.exists():
            return str(candidate)
        return shard_path

    def _load_metadata(self) -> None:
        metadata_path = self.shard_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    self.metadata = json.load(file)
            except Exception:
                self._load_metadata_from_first_shard()
        else:
            self._load_metadata_from_first_shard()

        self.n_fields = int(self.metadata.get("n_fields", 5))
        self.image_shape = tuple(self.metadata.get("image_shape", [84, 304]))
        default_names = ["Density", "VelocityX", "VelocityY", "Pressure", "TurbulentSANuTilde"]
        self.field_names = list(self.metadata.get("field_names", default_names))[: self.n_fields]
        self.geometry_dim = int(self.metadata.get("geometry_dim", 27))

    def _load_metadata_from_first_shard(self) -> None:
        if len(self.index_df) == 0:
            raise RuntimeError("Cannot infer metadata from an empty dataset index")
        first_row = self.index_df.iloc[0]
        resolved_shard_path = Path(self._resolve_row_shard_path(first_row))
        if not resolved_shard_path.is_absolute():
            resolved_shard_path = (self.shard_dir / resolved_shard_path).resolve()
        if not resolved_shard_path.exists():
            raise FileNotFoundError(f"First shard not found: {resolved_shard_path}")

        with h5py.File(resolved_shard_path, "r", locking=False) as file:
            if "fields" in file:
                n_channels = int(file["fields"].shape[1])
                image_shape = list(file["fields"].shape[2:])
            elif "coords_center" in file:
                n_channels = 5
                image_shape = list(file["coords_center"].shape[2:])
            else:
                raise KeyError("First shard must contain either 'fields' or 'coords_center'")

            default_names = ["Density", "VelocityX", "VelocityY", "Pressure", "TurbulentSANuTilde"]
            preferred_key = "geometry_orig" if self.use_geometry_orig else "geometry"
            fallback_key = "geometry" if self.use_geometry_orig else "geometry_orig"
            if preferred_key in file:
                geometry_dim = int(file[preferred_key].shape[1])
            elif fallback_key in file:
                geometry_dim = int(file[fallback_key].shape[1])
            else:
                geometry_dim = 27

            self.metadata = {
                "n_fields": n_channels,
                "image_shape": image_shape,
                "field_names": list(file.attrs.get("field_names", default_names[:n_channels])),
                "geometry_dim": geometry_dim,
                "has_wall_distance": "wall_distance" in file,
            }

    def _get_h5_handle(self, shard_path: str) -> h5py.File:
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        current_worker = worker_info.id if worker_info is not None else -1
        if self._worker_id != current_worker:
            self.close_all_handles()
            self._worker_id = current_worker

        if shard_path not in self._h5_handles:
            full_path = Path(shard_path)
            if not full_path.is_absolute():
                full_path = self.shard_dir / shard_path
            if not full_path.exists():
                raise FileNotFoundError(f"HDF5 shard not found: {full_path}")
            self._h5_handles[shard_path] = h5py.File(
                full_path,
                "r",
                rdcc_nbytes=self.cache_size_mb * 1024 * 1024,
                rdcc_nslots=1007,
                swmr=True,
                libver="latest",
                locking=False,
            )
        return self._h5_handles[shard_path]

    @staticmethod
    def _add_normalized_indices(coords: torch.Tensor) -> torch.Tensor:
        _, height, width = coords.shape
        j_indices = torch.arange(height, dtype=torch.float32).view(height, 1).expand(height, width)
        i_indices = torch.arange(width, dtype=torch.float32).view(1, width).expand(height, width)
        j_norm = j_indices / max(height - 1, 1)
        i_norm = i_indices / max(width - 1, 1)
        return torch.cat([coords, i_norm.unsqueeze(0), j_norm.unsqueeze(0)], dim=0)

    def __len__(self) -> int:
        return int(len(self.index_df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.index_df.iloc[int(idx)]
        shard_path = self._resolve_row_shard_path(row)
        local_idx = int(row["local_index"])
        h5_file = self._get_h5_handle(shard_path)

        sample: Dict[str, Any] = {}
        if "fields" in h5_file:
            fields = torch.tensor(h5_file["fields"][local_idx], dtype=torch.float32)
            if self.normalizer is not None:
                fields = self.normalizer.transform(fields.unsqueeze(0)).squeeze(0)
            sample["fields"] = fields

        if "coords_center" not in h5_file:
            raise KeyError(f"Required key 'coords_center' missing in shard {shard_path}")
        coords_center_pde = torch.tensor(h5_file["coords_center"][local_idx], dtype=torch.float64)
        sample["coords_center_pde"] = coords_center_pde
        sample["coords_center"] = self._add_normalized_indices(coords_center_pde.float())

        if "coords_vertex" not in h5_file:
            raise KeyError(f"Required key 'coords_vertex' missing in shard {shard_path}")
        sample["coords_vertex"] = torch.tensor(
            np.asarray(h5_file["coords_vertex"][local_idx], dtype=np.float64),
            dtype=torch.float64,
        )

        preferred_key = "geometry_orig" if self.use_geometry_orig else "geometry"
        fallback_key = "geometry" if self.use_geometry_orig else "geometry_orig"
        if preferred_key in h5_file:
            geometry_key = preferred_key
        elif fallback_key in h5_file:
            geometry_key = fallback_key
        else:
            raise KeyError(f"Required key 'geometry' or 'geometry_orig' missing in shard {shard_path}")
        sample["geometry"] = torch.tensor(
            np.asarray(h5_file[geometry_key][local_idx], dtype=np.float32),
            dtype=torch.float32,
        )

        if "flow" not in h5_file:
            raise KeyError(f"Required key 'flow' missing in shard {shard_path}")
        sample["flow_conditions"] = torch.tensor(
            np.asarray(h5_file["flow"][local_idx], dtype=np.float32),
            dtype=torch.float32,
        )

        cgns_basename = row.get("cgns_basename", "")
        sample.update({
            "cgns_basename": self._normalize_optional_text(cgns_basename),
            "source_name": self._normalize_optional_text(row.get("source_name")),
            "source_kind": self._normalize_optional_text(row.get("source_kind")),
            "source_chunk": self._normalize_optional_text(row.get("source_chunk")),
            "source_index_path": self._normalize_optional_text(row.get("source_index_path")),
            "source_shard_root": self._normalize_optional_text(row.get("source_shard_root")),
            "source_shard_path": self._normalize_optional_text(row.get("source_shard_path")),
            "index": int(idx),
        })
        if "global_id" in row:
            sample["global_id"] = int(row["global_id"])

        if "coefficients" in h5_file:
            sample["coefficients"] = torch.tensor(
                np.asarray(h5_file["coefficients"][local_idx], dtype=np.float32),
                dtype=torch.float32,
            )

        if "wall_distance" in h5_file:
            sample["wall_distance"] = torch.tensor(
                np.asarray(h5_file["wall_distance"][local_idx], dtype=np.float32),
                dtype=torch.float32,
            )
        else:
            sample["wall_distance"] = torch.zeros(
                self.image_shape[0],
                self.image_shape[1],
                dtype=torch.float32,
            )

        if not self.validate_sample(sample):
            raise ValueError(
                f"Sample {idx} failed validation. Shard={shard_path}, local_index={local_idx}"
            )
        return sample

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(self.metadata)
        metadata.update({
            "index_path": str(self.index_path),
            "shard_dir": str(self.shard_dir),
            "cache_size_mb": self.cache_size_mb,
            "n_samples": len(self.index_df),
            "normalization": self.normalizer is not None,
            "normalize": self.normalize,
            "scale_turbulent": self.scale_turbulent,
        })
        return metadata

    def get_normalizer(self):
        return self.normalizer

    def close_all_handles(self) -> None:
        for shard_path, handle in list(self._h5_handles.items()):
            try:
                if hasattr(handle, "id") and handle.id and handle.id.valid:
                    handle.close()
            except Exception as exc:
                self.logger.warning("Failed to close HDF5 handle for %s: %s", shard_path, exc)
        self._h5_handles.clear()

    def __del__(self) -> None:
        try:
            self.close_all_handles()
        except Exception:
            pass


__all__ = ["H5MultiFieldDataset"]
