"""ADFLOW-aligned wall-distance utilities for structured 2D grids."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray]


def _as_batched_coords(value: torch.Tensor, *, name: str) -> tuple[torch.Tensor, bool]:
    if value.ndim == 3:
        return value.unsqueeze(0), False
    if value.ndim == 4:
        return value, True
    raise ValueError(
        f"{name} must have shape (2,H,W) or (B,2,H,W), got {tuple(value.shape)}"
    )


def _batched_wall_segment_mask(
    wall_segment_mask: Optional[TensorLike],
    *,
    batch_size: int,
    segment_count: int,
    device: torch.device,
) -> torch.Tensor:
    if wall_segment_mask is None:
        return torch.ones(batch_size, segment_count, device=device, dtype=torch.bool)
    mask = torch.as_tensor(wall_segment_mask, device=device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    if mask.shape != (batch_size, segment_count):
        raise ValueError(
            "wall_segment_mask must have shape (S,) or (B,S), "
            f"got {tuple(mask.shape)} for B={batch_size}, S={segment_count}"
        )
    return mask


def _nearest_wall_segment_distance(
    coords_center: torch.Tensor,
    coords_vertex: torch.Tensor,
    *,
    wall_segment_mask: Optional[TensorLike],
    max_pairs_per_chunk: int,
) -> torch.Tensor:
    """Vectorized point-to-segment distance with bounded temporary memory."""
    batch_size, _, height, width = coords_center.shape
    if coords_vertex.shape[0] == 1 and batch_size > 1:
        coords_vertex = coords_vertex.expand(batch_size, -1, -1, -1)
    if coords_vertex.shape[0] != batch_size:
        raise ValueError("coords_vertex batch dimension must match coords_center")

    points = coords_center.permute(0, 2, 3, 1).reshape(batch_size, height * width, 2)
    segment_start = coords_vertex[:, :, 0, :-1].permute(0, 2, 1)
    segment_end = coords_vertex[:, :, 0, 1:].permute(0, 2, 1)
    segment_vector = segment_end - segment_start
    segment_length2 = torch.sum(segment_vector * segment_vector, dim=-1)
    segment_count = int(segment_start.shape[1])
    valid_segment = _batched_wall_segment_mask(
        wall_segment_mask,
        batch_size=batch_size,
        segment_count=segment_count,
        device=coords_center.device,
    ) & (segment_length2 > 0.0)
    if not bool(valid_segment.any(dim=1).all().item()):
        raise ValueError("each sample must contain at least one non-degenerate wall segment")

    pairs_per_point = batch_size * segment_count
    points_per_chunk = max(1, int(max_pairs_per_chunk) // pairs_per_point)
    distances = []
    for start in range(0, points.shape[1], points_per_chunk):
        point_chunk = points[:, start:start + points_per_chunk, :]
        point_from_start = point_chunk[:, :, None, :] - segment_start[:, None, :, :]
        projection_dot = torch.sum(
            point_from_start * segment_vector[:, None, :, :],
            dim=-1,
        )
        projection = torch.clamp(
            projection_dot / segment_length2.clamp_min(torch.finfo(coords_center.dtype).tiny)[:, None, :],
            min=0.0,
            max=1.0,
        )
        closest_delta = (
            point_from_start
            - projection[..., None] * segment_vector[:, None, :, :]
        )
        distance2 = torch.sum(closest_delta * closest_delta, dim=-1)
        distance2 = torch.where(
            valid_segment[:, None, :],
            distance2,
            torch.full_like(distance2, torch.inf),
        )
        distances.append(torch.sqrt(distance2.amin(dim=-1)))
    return torch.cat(distances, dim=1).reshape(batch_size, height, width)


def compute_wall_distance(
    coords_center: TensorLike,
    coords_vertex: Optional[TensorLike] = None,
    method: str = "nearest_wall_segment",
    *,
    wall_segment_mask: Optional[TensorLike] = None,
    max_pairs_per_chunk: int = 4_000_000,
    compute_dtype: torch.dtype = torch.float64,
) -> TensorLike:
    """Compute wall distance on CPU or GPU with batch-parallel Torch kernels.

    ``nearest_wall_segment`` is the default ADFLOW-aligned 2D O-grid method.
    ``same_i_projection`` is retained only for explicit legacy reproduction.
    ``max_pairs_per_chunk`` bounds the product ``B * points * wall_segments``
    used by each vectorized chunk.
    """
    is_numpy = isinstance(coords_center, np.ndarray)
    center = torch.as_tensor(coords_center)
    device = center.device
    center = center.to(dtype=compute_dtype)
    center, had_batch = _as_batched_coords(center, name="coords_center")

    vertex = None
    if coords_vertex is not None:
        vertex = torch.as_tensor(coords_vertex, device=device, dtype=compute_dtype)
        vertex, _ = _as_batched_coords(vertex, name="coords_vertex")

    method = str(method).lower()
    if method in {"nearest_wall", "nearest_wall_segment"}:
        if vertex is None:
            raise ValueError("nearest_wall_segment requires coords_vertex")
        distance = _nearest_wall_segment_distance(
            center,
            vertex,
            wall_segment_mask=wall_segment_mask,
            max_pairs_per_chunk=max_pairs_per_chunk,
        )
    elif method == "same_i_projection":
        if vertex is not None:
            wall = 0.5 * (vertex[:, :, 0, :-1] + vertex[:, :, 0, 1:])
        else:
            wall = center[:, :, 0, :]
        delta = center - wall[:, :, None, :]
        distance = torch.sqrt(torch.sum(delta * delta, dim=1))
    else:
        raise ValueError(f"Unknown wall-distance method: {method}")

    if not had_batch:
        distance = distance.squeeze(0)
    if is_numpy:
        return distance.cpu().numpy()
    return distance


def resolve_wall_distance_torch(
    coords_center: torch.Tensor,
    coords_vertex: Optional[torch.Tensor],
    wall_distance: Optional[torch.Tensor] = None,
    *,
    wall_segment_mask: Optional[TensorLike] = None,
    max_pairs_per_chunk: int = 4_000_000,
    compute_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, str]:
    """Select valid supplied distances and geometrically replace invalid samples.

    The returned source is ``provided``, ``torch_nearest_wall_segment``,
    ``mixed_provided_torch``, or ``legacy_same_i_projection``.
    """
    center, had_batch = _as_batched_coords(coords_center, name="coords_center")
    batch_size, _, height, width = center.shape

    if wall_distance is not None:
        supplied = torch.as_tensor(
            wall_distance,
            device=center.device,
            dtype=compute_dtype,
        )
        if supplied.ndim == 2:
            supplied = supplied.unsqueeze(0)
            if batch_size > 1:
                supplied = supplied.expand(batch_size, -1, -1)
        if supplied.shape != (batch_size, height, width):
            raise ValueError(
                "wall_distance shape must match coords_center spatial/batch dimensions, "
                f"got {tuple(supplied.shape)} expected {(batch_size, height, width)}"
            )
        flat = supplied.reshape(batch_size, -1)
        valid = torch.isfinite(flat).all(dim=1) & (flat.amin(dim=1) > 0.0)
        if bool(valid.all().item()):
            return (supplied if had_batch else supplied.squeeze(0)), "provided"
    else:
        supplied = None
        valid = torch.zeros(batch_size, device=center.device, dtype=torch.bool)

    fallback_method = "nearest_wall_segment" if coords_vertex is not None else "same_i_projection"
    fallback = compute_wall_distance(
        center,
        coords_vertex,
        method=fallback_method,
        wall_segment_mask=wall_segment_mask,
        max_pairs_per_chunk=max_pairs_per_chunk,
        compute_dtype=compute_dtype,
    )
    if fallback.ndim == 2:
        fallback = fallback.unsqueeze(0)

    if supplied is None or not bool(valid.any().item()):
        resolved = fallback
        source = (
            "torch_nearest_wall_segment"
            if fallback_method == "nearest_wall_segment"
            else "legacy_same_i_projection"
        )
    else:
        resolved = torch.where(valid[:, None, None], supplied, fallback)
        source = "mixed_provided_torch"
    return (resolved if had_batch else resolved.squeeze(0)), source


def compute_wall_distance_per_column(
    coords_center: torch.Tensor,
    coords_vertex: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit legacy same-column wall distance."""
    return compute_wall_distance(coords_center, coords_vertex, method="same_i_projection")


def get_wall_distance_statistics(
    d_wall: TensorLike,
    percentiles: list[int] | None = None,
) -> dict[str, object]:
    """Return basic wall-distance distribution statistics."""
    percentiles = percentiles or [1, 5, 10, 25, 50, 75, 90, 95, 99]
    d_wall_t = torch.as_tensor(d_wall)
    d_flat = d_wall_t.flatten()
    stats: dict[str, object] = {
        "min": float(d_flat.min()),
        "max": float(d_flat.max()),
        "mean": float(d_flat.mean()),
        "std": float(d_flat.std()),
    }
    for percentile in percentiles:
        stats[f"p{percentile}"] = float(torch.quantile(d_flat.float(), percentile / 100.0))
    if d_wall_t.ndim == 2:
        per_j_mean = d_wall_t.mean(dim=1)
    else:
        per_j_mean = d_wall_t.mean(dim=(0, 2))
    stats["per_j_mean"] = per_j_mean.tolist()
    return stats


__all__ = [
    "compute_wall_distance",
    "compute_wall_distance_per_column",
    "get_wall_distance_statistics",
    "resolve_wall_distance_torch",
]
