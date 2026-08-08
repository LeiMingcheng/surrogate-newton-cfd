"""DataLoader construction for surrogate HDF5 datasets."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from surrogate.data.h5_dataset import H5MultiFieldDataset


def create_dataloaders(
    index_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    normalize: bool = False,
    scale_turbulent: bool = False,
    turbulent_stats_file: str = "turbulent_scale_stats.json",
    use_ddp: bool = False,
    seed: Optional[int] = None,
    num_samples: Optional[int] = None,
    use_geometry_orig: bool = False,
) -> Tuple[DataLoader, DataLoader, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create train/validation dataloaders from an HDF5 index."""
    full_dataset = H5MultiFieldDataset(
        index_path=index_path,
        normalize=normalize,
        scale_turbulent=scale_turbulent,
        turbulent_stats_file=turbulent_stats_file,
        num_samples=num_samples,
        use_geometry_orig=use_geometry_orig,
    )

    n_total = len(full_dataset)
    if train_split >= 1.0:
        train_dataset = full_dataset
        val_dataset = full_dataset
    else:
        n_train = int(n_total * float(train_split))
        n_val = n_total - n_train
        sampler_seed = 42 if seed is None else int(seed)
        generator = torch.Generator().manual_seed(sampler_seed)
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    val_kwargs = dict(train_kwargs)

    if num_workers > 0:
        import torch.multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        train_kwargs["persistent_workers"] = bool(persistent_workers)
        train_kwargs["prefetch_factor"] = int(prefetch_factor)
        val_kwargs["persistent_workers"] = bool(persistent_workers)
        val_kwargs["prefetch_factor"] = int(prefetch_factor)

    sampler_seed = 42 if seed is None else int(seed)
    if use_ddp and dist.is_initialized():
        train_kwargs["sampler"] = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=False,
            seed=sampler_seed,
        )
        val_kwargs["sampler"] = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
            seed=sampler_seed,
        )
    else:
        train_kwargs["shuffle"] = True
        val_kwargs["shuffle"] = False

    return (
        DataLoader(train_dataset, **train_kwargs),
        DataLoader(val_dataset, **val_kwargs),
        train_dataset,
        val_dataset,
    )


def create_dataloaders_from_config(
    config,
    *,
    use_ddp: bool = False,
):
    """Create dataloaders from a clean ExperimentConfig-like object."""
    data = config.data
    training = getattr(config, "training", None)
    batch_size = data.batch_size
    if training is not None and getattr(training, "batch_size", None) is not None:
        batch_size = training.batch_size
    pin_memory = bool(data.pin_memory)
    runtime = getattr(config, "runtime", None)
    runtime_device = str(getattr(runtime, "device", "")).lower()
    if runtime_device.startswith("cpu"):
        pin_memory = False
    return create_dataloaders(
        index_path=data.index_path,
        batch_size=int(batch_size),
        num_workers=int(data.num_workers),
        train_split=float(data.train_split),
        pin_memory=pin_memory,
        persistent_workers=bool(data.persistent_workers),
        prefetch_factor=int(data.prefetch_factor),
        normalize=bool(data.normalize),
        scale_turbulent=bool(data.scale_turbulent),
        turbulent_stats_file=str(data.stats_path or "turbulent_scale_stats.json"),
        use_ddp=use_ddp,
        seed=int(getattr(config, "seed", 42)),
        num_samples=data.num_samples,
        use_geometry_orig=bool(data.use_geometry_orig),
    )


__all__ = ["create_dataloaders", "create_dataloaders_from_config"]
