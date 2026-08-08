"""Dataset utility helpers used by inference and evaluation."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch


def get_base_dataset(dataset: Any) -> Any:
    """Unwrap nested torch Subset datasets."""
    try:
        from torch.utils.data import Subset
    except Exception:
        return dataset

    base = dataset
    while isinstance(base, Subset):
        base = base.dataset
    return base


def prefer_geometry_orig_batch(
    base_dataset: Any,
    base_indices: Any,
    geometry_fallback: torch.Tensor,
    geometry_orig_key: str = "geometry_orig",
) -> Tuple[torch.Tensor, int, int]:
    """Load geometry_orig for selected samples when available."""
    if base_indices is None:
        return geometry_fallback, 0, int(geometry_fallback.shape[0])

    base_indices_list = (
        base_indices.detach().cpu().tolist()
        if torch.is_tensor(base_indices)
        else [int(value) for value in base_indices]
    )
    batch_size, geometry_dim = geometry_fallback.shape
    if len(base_indices_list) != batch_size:
        return geometry_fallback, 0, batch_size

    out = geometry_fallback.clone()
    used_orig = 0
    used_fallback = batch_size
    for j, base_idx in enumerate(base_indices_list):
        try:
            row = base_dataset.index_df.iloc[int(base_idx)]
            shard_path = base_dataset._resolve_row_shard_path(row)
            local_idx = int(row["local_index"])
            h5_file = base_dataset._get_h5_handle(shard_path)
            if geometry_orig_key not in h5_file:
                continue
            geom_orig = h5_file[geometry_orig_key][local_idx]
            if int(geom_orig.shape[0]) != int(geometry_dim):
                continue
            out[j].copy_(torch.from_numpy(np.asarray(geom_orig, dtype=np.float32)))
        except Exception:
            continue
        used_orig += 1
        used_fallback -= 1

    return out, used_orig, used_fallback


def collect_sample_ordinals(loader: Any, ordinals: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Collect batches by ordinal from a loader."""
    wanted = set(int(ordinal) for ordinal in ordinals)
    dataset = getattr(loader, "dataset", None)
    batch_size = int(getattr(loader, "batch_size", 1) or 1)
    collate_fn = getattr(loader, "collate_fn", None)
    if dataset is not None and hasattr(dataset, "__getitem__") and batch_size == 1 and callable(collate_fn):
        batches = {int(idx): collate_fn([dataset[int(idx)]]) for idx in sorted(wanted)}
    else:
        batches = {}
        for idx, batch in enumerate(loader):
            if idx in wanted:
                batches[int(idx)] = batch
                if len(batches) == len(wanted):
                    break

    missing = sorted(wanted.difference(batches.keys()))
    if missing:
        raise IndexError(f"Validation ordinals out of range: {missing}")
    return batches


__all__ = [
    "collect_sample_ordinals",
    "get_base_dataset",
    "prefer_geometry_orig_batch",
]
