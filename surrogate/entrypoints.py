"""Clean workflow entrypoints for training, evaluation, inference, and resume runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional

import torch.distributed as dist
import torch

from surrogate.configs import ExperimentConfig, load_config
from surrogate.data import (
    UniformFlowInitializer,
    create_dataloaders_from_config,
    get_base_dataset,
)
from surrogate.direct.training import DirectTrainer, DirectTrainerConfig
from surrogate.evaluation.direct_validation import DirectValidationRunner
from surrogate.evaluation.fsb_validation import FSBValidationRunner
from surrogate.evaluation.options import build_validation_options_from_config
from surrogate.evaluation.reports import save_json
from surrogate.fsb.training import FSBTrainer, FSBTrainerConfig
from surrogate.inference import (
    DirectPredictorBackend,
    DirectPredictorConfig,
    FSBPredictorBackend,
    FSBPredictorConfig,
)
from surrogate.models import create_model
from surrogate.physics.residual import get_residual_calculator
from surrogate.training.engine import (
    TrainingEngine,
    create_ema,
    create_lr_scheduler,
    create_optimizer,
    load_ema_state,
    load_training_state,
    resolve_training_checkpoint,
)


@dataclass
class WorkflowResult:
    """Small serializable result returned by clean workflow entrypoints."""

    task: str
    output_dir: str
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "output_dir": self.output_dir,
            "artifacts": dict(self.artifacts),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


@dataclass
class DistributedContext:
    """Distributed runtime metadata for one workflow invocation."""

    use_ddp: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    initialized_here: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _setup_distributed(
    *,
    requested_device: str,
    force_ddp: bool = False,
    backend: str = "nccl",
) -> DistributedContext:
    use_ddp = bool(force_ddp) or int(os.environ.get("WORLD_SIZE", "1")) > 1
    initialized_here = False
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True

    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if str(requested_device).startswith("cuda"):
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(requested_device)
        return DistributedContext(
            use_ddp=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=local_rank,
            device=device,
            initialized_here=initialized_here,
        )

    return DistributedContext(
        use_ddp=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device=torch.device(requested_device),
        initialized_here=False,
    )


def _resolve_output_dir(
    config: ExperimentConfig,
    *,
    task: str,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if task == "train":
        training_output_dir = getattr(config.training, "output_dir", None)
        if training_output_dir is not None:
            return Path(training_output_dir)
        return Path("checkpoints") / config.experiment.name
    return Path(config.evaluation.output_dir) / config.experiment.name / task


def _checkpoint_path(config: ExperimentConfig, override: str | Path | None) -> Optional[str]:
    if override is not None:
        return str(override)
    return config.runtime.checkpoint


def _base_normalizer_from_dataset(dataset: Any) -> Any:
    base_dataset = get_base_dataset(dataset)
    if hasattr(base_dataset, "get_normalizer"):
        normalizer = base_dataset.get_normalizer()
        if normalizer is not None:
            return copy.deepcopy(normalizer)
    return None


def _wrap_ddp(
    model: torch.nn.Module,
    context: DistributedContext,
    *,
    sync_bn: bool = False,
) -> torch.nn.Module:
    if not context.use_ddp:
        return model
    if sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if context.device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            gradient_as_bucket_view=True,
        )
    return torch.nn.parallel.DistributedDataParallel(
        model,
        gradient_as_bucket_view=True,
    )


def run_training(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    force_ddp: bool = False,
    dist_backend: str = "nccl",
    sync_bn: bool = False,
) -> WorkflowResult:
    """Run clean Direct/FSB training from an ``ExperimentConfig`` file."""
    config = load_config(config_path)
    if device is not None:
        config.runtime.device = str(device)
    config.task.kind = "train"
    config.validate()
    _set_seed(config.seed)

    context = _setup_distributed(
        requested_device=config.runtime.device,
        force_ddp=force_ddp,
        backend=dist_backend,
    )
    try:
        train_loader, val_loader, train_dataset, _ = create_dataloaders_from_config(
            config,
            use_ddp=context.use_ddp,
        )
        normalizer = _base_normalizer_from_dataset(train_dataset)
        model = create_model(config.model).to(context.device)
        optimizer = create_optimizer(model, config)
        lr_scheduler = create_lr_scheduler(
            optimizer,
            config,
            steps_per_epoch=len(train_loader),
        )
        resume_path = resolve_training_checkpoint(config, checkpoint_path)
        resume_state = None
        if resume_path:
            resume_state = load_training_state(
                checkpoint_path=resume_path,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=context.device,
                load_training_state=bool(config.training.load_training_state),
                steps_per_epoch=len(train_loader),
            )
        model = _wrap_ddp(model, context, sync_bn=sync_bn)
        ema = create_ema(model, config)
        if resume_state is not None:
            load_ema_state(ema, resume_state.payload)

        if config.model.family == "direct":
            trainer = DirectTrainer(
                model,
                optimizer,
                config=DirectTrainerConfig(**config.training.to_direct_trainer_config()),
                normalizer=normalizer,
                device=context.device,
            )
        elif config.model.family == "fsb":
            uniform_initializer = UniformFlowInitializer(
                normalizer=normalizer,
                device=context.device,
            )

            def initial_field_fn(
                flow_conditions: torch.Tensor,
                coords: torch.Tensor,
                batch: Mapping[str, Any],
            ) -> torch.Tensor:
                del batch
                return uniform_initializer.generate_uniform_field(
                    flow_conditions=flow_conditions,
                    spatial_shape=(int(coords.shape[-2]), int(coords.shape[-1])),
                    coords=coords,
                )

            trainer = FSBTrainer(
                model,
                optimizer,
                experiment_config=config,
                config=FSBTrainerConfig(**config.training.to_fsb_trainer_config()),
                initial_field_fn=initial_field_fn,
                normalizer=normalizer,
                lr_scheduler=lr_scheduler,
                device=context.device,
            )
        else:
            raise ValueError(f"Unsupported model family: {config.model.family}")

        if resume_state is not None:
            trainer.set_global_step(resume_state.global_step)
        else:
            trainer.set_global_step(0)

        out_dir = _resolve_output_dir(config, task="train", output_dir=output_dir)
        history = list(resume_state.history or []) if resume_state is not None else []
        start_epoch = int(resume_state.start_epoch) if resume_state is not None else 0
        engine = TrainingEngine(
            trainer=trainer,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            config=config,
            context=context,
            output_dir=out_dir,
            history=history,
        )
        artifacts = engine.fit(
            train_loader,
            start_epoch=start_epoch,
            total_epochs=int(config.training.epochs),
            val_loader=val_loader,
        )
        history = engine.history
        return WorkflowResult(
            task="train",
            output_dir=str(out_dir),
            artifacts=artifacts,
            metrics=dict(history[-1]) if history else {},
            metadata={
                "model_key": config.model.get_public_model_key(),
                "rank": context.rank,
                "world_size": context.world_size,
            },
        )
    finally:
        context.close()


def _build_validation_options(
    config: ExperimentConfig,
    *,
    max_batches: int | None = None,
    record_samples: bool | None = None,
    compute_physical_field_metrics: bool | None = None,
    compute_forces: bool | None = None,
    compute_residuals: bool | None = None,
):
    return build_validation_options_from_config(
        config,
        max_batches=max_batches,
        record_samples=record_samples,
        compute_physical_field_metrics=compute_physical_field_metrics,
        compute_forces=compute_forces,
        compute_residuals=compute_residuals,
    )


def run_evaluation(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    use_ema: bool = True,
    max_batches: int | None = None,
    record_samples: bool | None = None,
    compute_physical_field_metrics: bool | None = None,
    compute_forces: bool | None = None,
    compute_residuals: bool | None = None,
    n_inference_steps: int | None = None,
    custom_timesteps: Optional[list[int]] = None,
    eta: float = 0.0,
    noise_mode: str = "zeros",
    stem: str = "evaluation",
) -> WorkflowResult:
    """Run labeled validation/benchmark evaluation from a clean config."""
    config = load_config(config_path)
    if device is not None:
        config.runtime.device = str(device)
    config.task.kind = "validate"
    config.validate()
    _, val_loader, _, _ = create_dataloaders_from_config(config, use_ddp=False)
    options = _build_validation_options(
        config,
        max_batches=max_batches,
        record_samples=record_samples,
        compute_physical_field_metrics=compute_physical_field_metrics,
        compute_forces=compute_forces,
        compute_residuals=compute_residuals,
    )
    residual_calculator = None
    if options.compute_residuals:
        residual_calculator = get_residual_calculator(device=config.runtime.device)

    ckpt = _checkpoint_path(config, checkpoint_path)
    if config.model.family == "direct":
        backend = DirectPredictorBackend(
            DirectPredictorConfig(
                config_path=str(config_path),
                checkpoint_path=ckpt,
                device=config.runtime.device,
                use_ema=use_ema,
            )
        )
        runner = DirectValidationRunner(
            backend,
            options=options,
            residual_calculator=residual_calculator,
        )
    elif config.model.family == "fsb":
        backend = FSBPredictorBackend(
            FSBPredictorConfig(
                config_path=str(config_path),
                checkpoint_path=ckpt,
                device=config.runtime.device,
                use_ema=use_ema,
                n_inference_steps=n_inference_steps,
                custom_timesteps=custom_timesteps,
                eta=eta,
                noise_mode=noise_mode,
            )
        )
        runner = FSBValidationRunner(
            backend,
            options=options,
            residual_calculator=residual_calculator,
        )
    else:
        raise ValueError(f"Unsupported model family: {config.model.family}")

    report = runner.evaluate_report(
        val_loader,
        metadata={
            "config_path": str(config_path),
            "checkpoint_path": ckpt,
            "task": stem,
        },
    )
    out_dir = _resolve_output_dir(config, task=stem, output_dir=output_dir)
    artifacts = {key: str(path) for key, path in report.save(out_dir, stem=stem).items()}
    return WorkflowResult(
        task=stem,
        output_dir=str(out_dir),
        artifacts=artifacts,
        metrics=dict(report.result.metrics),
        metadata={
            "model_key": config.model.get_public_model_key(),
            "n_samples": report.result.n_samples,
            "n_batches": report.result.n_batches,
        },
    )


def run_inference(
    config_path: str | Path,
    **kwargs: Any,
) -> WorkflowResult:
    """Run the offline inference path over the configured validation split."""
    return run_evaluation(config_path, stem="inference", **kwargs)


def run_nk_resume(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    use_ema: bool = True,
    max_cases: int | None = None,
    payload_only: bool = True,
    backend_command: Optional[tuple[str, ...]] = None,
    cgns_root: str = "",
    ranks_per_case: int = 1,
    mpi_launcher: str | None = "auto",
    mpi_omp_threads: int = 1,
    command_timeout_s: float | None = None,
    plan_preset: str = "finalonly",
    n_inference_steps: int | None = None,
    custom_timesteps: Optional[list[int]] = None,
    eta: float = 0.0,
    noise_mode: str = "zeros",
    ordinals: tuple[int, ...] | None = None,
    index_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    resume_mode: str = "ank_nk",
    max_work: int = 2000,
    time_limit_s: float = 10.0,
    nk_switch_tolerance: float = 1.0e-4,
    l2conv: float = 1.0e-8,
    repeated_nk_cycles: tuple[int, ...] = (6, 8, 10),
    solver_preset: str | None = None,
    fixed_cycles: int | None = None,
) -> WorkflowResult:
    """Run model-side inference and export canonical final-only NK cases."""
    from surrogate.nk_resume import FinalOnlyExperimentRequest, run_finalonly_experiment

    config = load_config(config_path)
    if device is not None:
        config.runtime.device = str(device)
    if str(plan_preset).strip().lower() != "finalonly":
        raise ValueError(
            "surrogate CLI model-side export supports finalonly only; "
            "use surrogate.nk_resume.alternating for FSB/NK scheduler feedback"
        )
    if not payload_only or backend_command:
        raise ValueError(
            "surrogate CLI only exports model-side NK cases; execute the manifest with "
            "`python -m NK_resume.cli run-manifest`"
        )
    del command_timeout_s
    if ordinals is not None and max_cases is not None:
        raise ValueError("Specify either ordinals or max_cases, not both")
    selected_ordinals = tuple(int(value) for value in (ordinals or ()))
    if not selected_ordinals:
        if max_cases is None or int(max_cases) <= 0:
            raise ValueError("nk_resume export requires --ordinals or a positive --max-cases")
        selected_ordinals = tuple(range(int(max_cases)))

    ckpt = _checkpoint_path(config, checkpoint_path)
    out_dir = _resolve_output_dir(config, task="nk_resume", output_dir=output_dir)
    result = run_finalonly_experiment(
        FinalOnlyExperimentRequest(
            config_path=config_path,
            ordinals=selected_ordinals,
            output_dir=out_dir,
            index_path=index_path,
            stats_path=stats_path,
            checkpoint_path=ckpt,
            predictor_kind=config.model.family,
            device=config.runtime.device,
            use_ema=use_ema,
            n_inference_steps=n_inference_steps,
            custom_timesteps=tuple(custom_timesteps or ()),
            eta=eta,
            noise_mode=noise_mode,
            cgns_root=cgns_root,
            ranks_per_case=ranks_per_case,
            mpi_launcher=str(mpi_launcher or "auto"),
            mpi_omp_threads=mpi_omp_threads,
            resume_mode=resume_mode,
            max_work=max_work,
            time_limit_s=time_limit_s,
            nk_switch_tolerance=nk_switch_tolerance,
            l2conv=l2conv,
            repeated_nk_cycles=repeated_nk_cycles,
            solver_preset=solver_preset,
            fixed_cycles=fixed_cycles,
            executor="export",
        )
    )
    return WorkflowResult(
        task="nk_resume",
        output_dir=str(out_dir),
        artifacts={
            "manifest": result.manifest_path,
            "summary": result.summary_path,
        },
        metrics={"job_count": float(len(result.ordinals))},
        metadata={
            "model_key": config.model.get_public_model_key(),
            "predictor_kind": result.predictor_kind,
            "ordinals": list(result.ordinals),
            "executor": result.executor,
        },
    )


def run_experiment(
    config_path: str | Path,
    *,
    task_kind: str | None = None,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    force_ddp: bool = False,
    dist_backend: str = "nccl",
    sync_bn: bool = False,
    use_ema: bool = True,
    max_batches: int | None = None,
    record_samples: bool | None = None,
    compute_physical_field_metrics: bool | None = None,
    compute_forces: bool | None = None,
    compute_residuals: bool | None = None,
    max_cases: int | None = None,
    payload_only: bool = True,
    backend_command: Optional[tuple[str, ...]] = None,
    cgns_root: str = "",
    ranks_per_case: int = 1,
    mpi_launcher: str | None = "auto",
    mpi_omp_threads: int = 1,
    command_timeout_s: float | None = None,
    plan_preset: str = "finalonly",
    n_inference_steps: int | None = None,
    custom_timesteps: Optional[list[int]] = None,
    eta: float = 0.0,
    noise_mode: str = "zeros",
    ordinals: tuple[int, ...] | None = None,
    index_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    resume_mode: str = "ank_nk",
    max_work: int = 2000,
    time_limit_s: float = 10.0,
    nk_switch_tolerance: float = 1.0e-4,
    l2conv: float = 1.0e-8,
    repeated_nk_cycles: tuple[int, ...] = (6, 8, 10),
    solver_preset: str | None = None,
    fixed_cycles: int | None = None,
) -> WorkflowResult:
    """Dispatch a clean surrogate workflow by ``task.kind``."""
    config = load_config(config_path)
    task = str(task_kind or config.task.kind).lower()
    if task == "train":
        return run_training(
            config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=device,
            force_ddp=force_ddp,
            dist_backend=dist_backend,
            sync_bn=sync_bn,
        )
    if task == "validate":
        return run_evaluation(
            config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=device,
            use_ema=use_ema,
            max_batches=max_batches,
            record_samples=record_samples,
            compute_physical_field_metrics=compute_physical_field_metrics,
            compute_forces=compute_forces,
            compute_residuals=compute_residuals,
            n_inference_steps=n_inference_steps,
            custom_timesteps=custom_timesteps,
            eta=eta,
            noise_mode=noise_mode,
            stem="evaluation",
        )
    if task == "infer":
        return run_inference(
            config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=device,
            use_ema=use_ema,
            max_batches=max_batches,
            record_samples=record_samples,
            compute_physical_field_metrics=compute_physical_field_metrics,
            compute_forces=compute_forces,
            compute_residuals=compute_residuals,
            n_inference_steps=n_inference_steps,
            custom_timesteps=custom_timesteps,
            eta=eta,
            noise_mode=noise_mode,
        )
    if task == "nk_resume":
        return run_nk_resume(
            config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=device,
            use_ema=use_ema,
            max_cases=max_cases,
            payload_only=payload_only,
            backend_command=backend_command,
            cgns_root=cgns_root,
            ranks_per_case=ranks_per_case,
            mpi_launcher=mpi_launcher,
            mpi_omp_threads=mpi_omp_threads,
            command_timeout_s=command_timeout_s,
            plan_preset=plan_preset,
            n_inference_steps=n_inference_steps,
            custom_timesteps=custom_timesteps,
            eta=eta,
            noise_mode=noise_mode,
            ordinals=ordinals,
            index_path=index_path,
            stats_path=stats_path,
            resume_mode=resume_mode,
            max_work=max_work,
            time_limit_s=time_limit_s,
            nk_switch_tolerance=nk_switch_tolerance,
            l2conv=l2conv,
            repeated_nk_cycles=repeated_nk_cycles,
            solver_preset=solver_preset,
            fixed_cycles=fixed_cycles,
        )
    raise ValueError("task.kind must be one of: train, validate, infer, nk_resume")


def workflow_result_to_json(result: WorkflowResult) -> str:
    """Serialize a workflow result for CLI output."""
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)


__all__ = [
    "DistributedContext",
    "WorkflowResult",
    "run_evaluation",
    "run_experiment",
    "run_inference",
    "run_nk_resume",
    "run_training",
    "workflow_result_to_json",
]
