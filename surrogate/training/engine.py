"""Shared training runtime for direct and FSB surrogate models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from surrogate.common.checkpointing import load_model_checkpoint
from surrogate.common.ema import EMAModel


def _save_json(data: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return path


@dataclass
class ResumeState:
    """Training state restored from a checkpoint."""

    start_epoch: int = 0
    global_step: int = 0
    history: list[dict[str, float]] | None = None
    payload: dict[str, Any] | None = None


def create_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    """Create a shared optimizer from ``config.training.optimizer``."""

    optimizer_cfg = dict(config.training.optimizer or {})
    name = str(optimizer_cfg.get("name", "adamw")).lower()
    lr = float(optimizer_cfg.get("lr", 1.0e-4))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    foreach = optimizer_cfg.get("foreach")
    tensorlist_options = {} if foreach is None else {"foreach": bool(foreach)}
    params = [param for param in model.parameters() if param.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **tensorlist_options,
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **tensorlist_options,
        )
    if name == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.0))
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Any,
    *,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """Create the shared per-step warmup+cosine scheduler."""

    total_epochs = int(config.training.epochs)
    if total_epochs <= 0 or steps_per_epoch <= 0:
        return None
    total_steps = total_epochs * int(steps_per_epoch)
    warmup_steps = int(float(config.training.warmup_ratio) * float(total_steps))
    eta_min_ratio = float(config.training.eta_min_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return eta_min_ratio + (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_training_checkpoint(config: Any, override: str | Path | None) -> Optional[str]:
    """Resolve the training checkpoint path without borrowing runtime.checkpoint."""

    if override is not None:
        return str(override)
    checkpoint_path = getattr(config.training, "checkpoint_path", None)
    if checkpoint_path is None:
        return None
    checkpoint_text = str(checkpoint_path).strip()
    return checkpoint_text or None


def _scheduler_state_from_payload(payload: Mapping[str, Any]) -> Any:
    if "lr_scheduler_state_dict" in payload:
        return payload["lr_scheduler_state_dict"]
    if "scheduler_state_dict" in payload:
        return payload["scheduler_state_dict"]
    return None


def _infer_start_epoch(payload: Mapping[str, Any], history: list[dict[str, float]]) -> int:
    if "completed_epochs" in payload:
        return int(payload["completed_epochs"])
    if "global_step" in payload and "epoch" in payload:
        return int(payload["epoch"]) + 1
    if "epoch" in payload:
        return int(payload["epoch"])
    return len(history)


def load_training_state(
    *,
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    load_training_state: bool,
    steps_per_epoch: int,
) -> ResumeState:
    """Load model weights and, optionally, full training state."""

    payload = load_model_checkpoint(
        model,
        checkpoint_path,
        device,
        use_ema=False,
        context="training_checkpoint",
    )
    history = [dict(item) for item in payload.get("history", [])]
    if not load_training_state:
        return ResumeState(start_epoch=0, global_step=0, history=[], payload=payload)

    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler_state = _scheduler_state_from_payload(payload)
    if lr_scheduler is not None and scheduler_state is not None:
        lr_scheduler.load_state_dict(scheduler_state)

    start_epoch = _infer_start_epoch(payload, history)
    global_step = int(payload.get("global_step", start_epoch * max(1, int(steps_per_epoch))))
    return ResumeState(
        start_epoch=start_epoch,
        global_step=global_step,
        history=history,
        payload=payload,
    )


def create_ema(model: torch.nn.Module, config: Any) -> Optional[EMAModel]:
    """Create EMA tracker when requested by the training config."""

    if not bool(getattr(config.training, "use_ema", False)):
        return None
    return EMAModel(model, decay=float(getattr(config.training, "ema_decay", 0.999)))


def load_ema_state(ema: Optional[EMAModel], payload: Optional[Mapping[str, Any]]) -> None:
    """Restore EMA state if available; otherwise keep EMA initialized from model weights."""

    if ema is None or payload is None:
        return
    if "ema_state_dict" in payload:
        ema.load_state_dict(payload["ema_state_dict"])


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _batch_size_from(batch: Mapping[str, Any]) -> float:
    if isinstance(batch, (list, tuple)):
        for value in batch:
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return float(value.shape[0])
        return 1.0
    if not isinstance(batch, Mapping):
        return 1.0
    for key in ("target", "fields", "geometry"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return float(value.shape[0])
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return float(value.shape[0])
    return 1.0


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().item())
        return float(value.detach().mean().item())
    return float(value)


def _reduce_metrics(
    metric_sums: Mapping[str, float],
    metric_counts: Mapping[str, float],
    *,
    device: torch.device,
) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return {
            key: float(metric_sums[key]) / float(max(metric_counts.get(key, 1.0), 1.0))
            for key in metric_sums
        }

    keys = sorted(metric_sums.keys())
    if not keys:
        return {}
    sums_tensor = torch.tensor([metric_sums[key] for key in keys], device=device, dtype=torch.float64)
    counts_tensor = torch.tensor([metric_counts.get(key, 1.0) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(sums_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
    return {
        key: float(sums_tensor[index].item()) / float(max(counts_tensor[index].item(), 1.0))
        for index, key in enumerate(keys)
    }


class TrainingEngine:
    """Common epoch/checkpoint/EMA runtime for direct and FSB trainers."""

    def __init__(
        self,
        *,
        trainer: Any,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
        ema: Optional[EMAModel],
        config: Any,
        context: Any,
        output_dir: str | Path,
        history: Optional[list[dict[str, float]]] = None,
    ) -> None:
        self.trainer = trainer
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema = ema
        self.config = config
        self.context = context
        self.output_dir = Path(output_dir)
        self.history = list(history or [])

    @property
    def is_main(self) -> bool:
        return bool(getattr(self.context, "is_main", True))

    @property
    def device(self) -> torch.device:
        return torch.device(getattr(self.context, "device", "cpu"))

    def _format_epoch_metrics(self, epoch_metrics: Mapping[str, float]) -> str:
        parts = [f"Epoch {int(epoch_metrics.get('epoch', -1)) + 1} Summary:"]
        summary_order = [
            "train_loss",
            "val_loss",
            "train_reconstruction_loss",
            "val_reconstruction_loss",
            "train_res_loss",
            "val_res_loss",
            "train_final_pde",
            "val_final_pde",
            "val_CL_mae",
            "val_CD_mae",
            "val_Cm_mae",
            "train_normalized_loss",
            "val_normalized_loss",
            "train_physical_loss",
            "val_physical_loss",
        ]
        for key in summary_order:
            if key in epoch_metrics:
                parts.append(f"  {key}: {float(epoch_metrics[key]):.8e}")
        return "\n".join(parts)

    def _create_writer(self) -> SummaryWriter | None:
        if not bool(self.config.training.tensorboard) or not self.is_main:
            return None
        log_dir = Path("runs") / str(self.config.experiment.name)
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))

    def _write_checkpoint(self, checkpoint_out: Path, *, completed_epochs: int) -> None:
        payload: dict[str, Any] = {
            "model_state_dict": _unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "history": self.history,
            "completed_epochs": int(completed_epochs),
            "epoch": int(completed_epochs),
            "global_step": int(getattr(self.trainer, "global_step", 0)),
        }
        if self.lr_scheduler is not None:
            payload["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if self.ema is not None:
            payload["ema_state_dict"] = self.ema.state_dict()
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint_out)

    def _run_train_epoch(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        *,
        epoch: int,
    ) -> dict[str, float]:
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, float] = {}
        for batch in train_loader:
            result = self.trainer.train_step(batch, epoch=epoch)
            batch_size = _batch_size_from(batch)
            metrics = dict(result.get("metrics", {}))
            metrics.setdefault("loss", _to_float(result["loss"]))
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + _to_float(value) * batch_size
                metric_counts[key] = metric_counts.get(key, 0.0) + batch_size
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.ema is not None:
                self.ema.update()
        return _reduce_metrics(metric_sums, metric_counts, device=self.device)

    @torch.no_grad()
    def _run_validation_epoch(
        self,
        val_loader: Iterable[Mapping[str, Any]],
        *,
        epoch: int,
    ) -> dict[str, float]:
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, float] = {}
        for batch in val_loader:
            result = self.trainer.validate_step(batch, epoch=epoch)
            batch_size = _batch_size_from(batch)
            metrics = dict(result.get("metrics", {}))
            metrics.setdefault("loss", _to_float(result["loss"]))
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + _to_float(value) * batch_size
                metric_counts[key] = metric_counts.get(key, 0.0) + batch_size
        return _reduce_metrics(metric_sums, metric_counts, device=self.device)

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        *,
        start_epoch: int,
        total_epochs: int,
        val_loader: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        writer = self._create_writer()
        checkpoint_every = self.config.training.checkpoint_every_epochs
        artifacts: dict[str, str] = {}
        try:
            for epoch in range(int(start_epoch), int(total_epochs)):
                train_sampler = getattr(train_loader, "sampler", None)
                if bool(getattr(self.context, "use_ddp", False)) and hasattr(train_sampler, "set_epoch"):
                    train_sampler.set_epoch(epoch)
                if self.is_main:
                    print("=" * 60, flush=True)
                    print(f"Epoch {epoch + 1}/{total_epochs}", flush=True)
                    print("=" * 60, flush=True)

                train_metrics = self._run_train_epoch(train_loader, epoch=epoch)
                epoch_metrics: dict[str, float] = {"epoch": float(epoch)}
                for key, value in train_metrics.items():
                    epoch_metrics[f"train_{key}"] = float(value)
                epoch_metrics["train_loss"] = float(train_metrics.get("total_loss", train_metrics.get("loss", 0.0)))

                if val_loader is not None:
                    if self.ema is not None:
                        self.ema.apply_shadow()
                    try:
                        val_metrics = self._run_validation_epoch(val_loader, epoch=epoch)
                    finally:
                        if self.ema is not None:
                            self.ema.restore()
                    for key, value in val_metrics.items():
                        epoch_metrics[f"val_{key}"] = float(value)
                    epoch_metrics["val_loss"] = float(val_metrics.get("total_loss", val_metrics.get("loss", 0.0)))

                self.history.append(epoch_metrics)
                if self.is_main:
                    print(self._format_epoch_metrics(epoch_metrics), flush=True)
                    if writer is not None:
                        for key, value in epoch_metrics.items():
                            if key == "epoch":
                                continue
                            writer.add_scalar(key, float(value), epoch + 1)
                    _save_json({"history": self.history, "config": self.config.to_dict()}, self.output_dir / "history.json")

                if self.is_main and checkpoint_every is not None and (epoch + 1) % int(checkpoint_every) == 0:
                    checkpoint_out = self.output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
                    self._write_checkpoint(checkpoint_out, completed_epochs=epoch + 1)
                    print(f"Saved checkpoint: {checkpoint_out}", flush=True)

            if self.is_main:
                history_path = _save_json(
                    {"history": self.history, "config": self.config.to_dict()},
                    self.output_dir / "history.json",
                )
                checkpoint_out = self.output_dir / "final_model.pt"
                self._write_checkpoint(checkpoint_out, completed_epochs=int(total_epochs))
                print(f"Saved final checkpoint: {checkpoint_out}", flush=True)
                artifacts = {"history": str(history_path), "checkpoint": str(checkpoint_out)}
        finally:
            if writer is not None:
                writer.close()
        return artifacts


__all__ = [
    "ResumeState",
    "TrainingEngine",
    "create_ema",
    "create_lr_scheduler",
    "create_optimizer",
    "load_ema_state",
    "load_training_state",
    "resolve_training_checkpoint",
]
