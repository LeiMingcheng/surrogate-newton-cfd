"""Clean configuration schema for the Surrogate-Newton CFD surrogate core package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from surrogate.training.loss_config import SharedTrainingLossConfig

__all__ = [
    "ExperimentInfo",
    "ModelConfig",
    "DataConfig",
    "TaskConfig",
    "TrainingLossConfig",
    "TrainingConfig",
    "FSBBridgeConfig",
    "FSBInferenceConfig",
    "FSBConfig",
    "RuntimeConfig",
    "EvaluationFieldMetricsConfig",
    "EvaluationForceMetricsConfig",
    "EvaluationResidualMetricsConfig",
    "EvaluationRichConfig",
    "EvaluationConfig",
    "NKResumeConfig",
    "ExperimentConfig",
    "ConfigManager",
    "load_config",
    "save_config",
]


_VALID_MODEL_FAMILIES = {"direct", "fsb"}
_VALID_BACKBONES = {"fno", "dit"}
_VALID_MODEL_KEYS = {
    "direct_fno",
    "direct_dit",
    "fsb_fno",
    "fsb_dit",
}
_VALID_TASK_KINDS = {"train", "validate", "infer", "nk_resume"}
_VALID_FSB_BETA_SCHEDULES = {"cosine", "linear", "symmetric_sine"}
_VALID_FSB_TIMESTEP_SPACING = {"quadratic", "uniform"}
_VALID_FSB_PREDICTION_TYPES = {"epsilon", "v_prediction", "x0"}


@dataclass
class ExperimentInfo:
    """Experiment identity and reproducibility metadata."""

    name: str = "default"
    seed: int = 42

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ValueError("experiment.name must be non-empty")
        self.seed = int(self.seed)


@dataclass
class ModelConfig:
    """Canonical direct/fsb model selection and constructor parameters."""

    family: str = "direct"
    backbone: str = "fno"
    params: dict[str, Any] = field(default_factory=dict)

    def get_family_backbone(self) -> tuple[str, str]:
        family = str(self.family).lower()
        backbone = str(self.backbone).lower()
        if family not in _VALID_MODEL_FAMILIES:
            raise ValueError(f"model.family must be one of {sorted(_VALID_MODEL_FAMILIES)}, got {self.family!r}")
        if backbone not in _VALID_BACKBONES:
            raise ValueError(f"model.backbone must be one of {sorted(_VALID_BACKBONES)}, got {self.backbone!r}")
        return family, backbone

    def get_public_model_key(self) -> str:
        family, backbone = self.get_family_backbone()
        key = f"{family}_{backbone}"
        if key not in _VALID_MODEL_KEYS:
            raise ValueError(f"Unsupported model key: {key}")
        return key

    def get_model_params(self) -> dict[str, Any]:
        params = dict(self.params or {})
        if isinstance(params.get("spatial_shape"), list):
            params["spatial_shape"] = tuple(params["spatial_shape"])
        return params

    def get_spatial_shape(self) -> tuple[int, int] | None:
        shape = (self.params or {}).get("spatial_shape")
        if shape is None:
            return None
        if len(shape) != 2:
            raise ValueError(f"model.params.spatial_shape must have two values, got {shape!r}")
        return int(shape[0]), int(shape[1])

    def validate(self) -> None:
        family, backbone = self.get_family_backbone()
        self.family = family
        self.backbone = backbone
        if not isinstance(self.params, dict):
            raise ValueError("model.params must be a mapping")


@dataclass
class DataConfig:
    """Dataset paths and loader controls for direct/fsb workflows."""

    index_path: str
    stats_path: str | None = None
    normalize: bool = True
    scale_turbulent: bool = True
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_samples: int | None = None
    use_geometry_orig: bool = False

    def validate(self) -> None:
        if not str(self.index_path).strip():
            raise ValueError("data.index_path must be non-empty")
        total = float(self.train_split) + float(self.val_split) + float(self.test_split)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("data train/val/test splits must sum to 1.0")
        if int(self.batch_size) <= 0:
            raise ValueError("data.batch_size must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("data.num_workers must be non-negative")


@dataclass
class TaskConfig:
    """Workflow selector."""

    kind: str = "train"

    def validate(self) -> None:
        kind = str(self.kind).lower()
        if kind not in _VALID_TASK_KINDS:
            raise ValueError(f"task.kind must be one of {sorted(_VALID_TASK_KINDS)}, got {self.kind!r}")
        self.kind = kind


@dataclass
class TrainingLossConfig(SharedTrainingLossConfig):
    """Loss controls shared by direct and fsb trainers."""

    fsb_reconstruction_weight: float = 0.0
    fsb_use_l1_reconstruction: bool = False
    fsb_loss_prediction_type: str | None = None

    def validate(self) -> None:
        super().validate()
        self.fsb_reconstruction_weight = float(self.fsb_reconstruction_weight)
        self.fsb_use_l1_reconstruction = bool(self.fsb_use_l1_reconstruction)
        if self.fsb_reconstruction_weight < 0:
            raise ValueError("training.loss.fsb_reconstruction_weight must be non-negative")
        if self.fsb_loss_prediction_type is not None:
            self.fsb_loss_prediction_type = str(self.fsb_loss_prediction_type).lower()
            if self.fsb_loss_prediction_type not in {"v_prediction"}:
                raise ValueError("training.loss.fsb_loss_prediction_type currently supports only 'v_prediction'")


@dataclass
class TrainingConfig:
    """Shared training controls for direct and fsb models."""

    epochs: int = 1000
    checkpoint_every_epochs: int | None = None
    checkpoint_path: str | None = None
    load_training_state: bool = False
    tensorboard: bool = False
    output_dir: str | None = None
    batch_size: int | None = None
    optimizer: dict[str, Any] = field(default_factory=lambda: {"name": "adamw", "lr": 1.0e-4})
    warmup_ratio: float = 0.0
    eta_min_ratio: float = 1.0
    use_ema: bool = False
    ema_decay: float = 0.999
    gradient_clip_norm: float | None = 0.5
    loss: TrainingLossConfig = field(default_factory=TrainingLossConfig)

    def validate(self) -> None:
        self.epochs = int(self.epochs)
        if self.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if self.checkpoint_every_epochs is not None:
            self.checkpoint_every_epochs = int(self.checkpoint_every_epochs)
            if self.checkpoint_every_epochs <= 0:
                raise ValueError("training.checkpoint_every_epochs must be positive when set")
        if self.checkpoint_path is not None and not str(self.checkpoint_path).strip():
            self.checkpoint_path = None
        self.load_training_state = bool(self.load_training_state)
        self.tensorboard = bool(self.tensorboard)
        if self.output_dir is not None and not str(self.output_dir).strip():
            raise ValueError("training.output_dir must be non-empty when set")
        if self.batch_size is not None and int(self.batch_size) <= 0:
            raise ValueError("training.batch_size must be positive when set")
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)
            if self.gradient_clip_norm < 0:
                raise ValueError("training.gradient_clip_norm must be non-negative")
        optimizer = dict(self.optimizer or {})
        name = str(optimizer.get("name", "adamw")).lower()
        if name not in {"adamw", "adam", "sgd"}:
            raise ValueError("training.optimizer.name must be adamw, adam, or sgd")
        lr = float(optimizer.get("lr", 0.0))
        if lr <= 0:
            raise ValueError("training.optimizer.lr must be positive")
        optimizer["name"] = name
        optimizer["lr"] = lr
        self.optimizer = optimizer
        self.warmup_ratio = float(self.warmup_ratio)
        self.eta_min_ratio = float(self.eta_min_ratio)
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("training.warmup_ratio must be between 0 and 1")
        if not 0.0 < self.eta_min_ratio <= 1.0:
            raise ValueError("training.eta_min_ratio must be in (0, 1]")
        self.use_ema = bool(self.use_ema)
        self.ema_decay = float(self.ema_decay)
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("training.ema_decay must be in (0, 1)")
        if isinstance(self.loss, dict):
            self.loss = TrainingLossConfig(**self.loss)
        self.loss.validate()

    def to_direct_trainer_config(self) -> dict[str, Any]:
        """Return kwargs accepted by DirectTrainerConfig."""
        return {
            "gradient_clip_norm": self.gradient_clip_norm,
            **self.loss.to_shared_loss_kwargs(),
        }

    def to_fsb_trainer_config(self) -> dict[str, Any]:
        """Return kwargs accepted by FSBTrainerConfig."""
        return {
            "lambda_reconstruction": self.loss.fsb_reconstruction_weight,
            "use_l1_reconstruction": self.loss.fsb_use_l1_reconstruction,
            "gradient_clip_norm": self.gradient_clip_norm,
            "loss_prediction_type": self.loss.fsb_loss_prediction_type,
            **self.loss.to_shared_loss_kwargs(),
        }


@dataclass
class FSBBridgeConfig:
    """I2SB bridge schedule for fsb models."""

    type: str = "i2sb"
    num_timesteps: int = 1000
    beta_max: float = 0.3
    beta_schedule: str = "symmetric_sine"
    timestep_spacing: str = "quadratic"
    prediction_type: str = "x0"

    def validate(self) -> None:
        self.type = str(self.type).lower()
        if self.type != "i2sb":
            raise ValueError("fsb.bridge.type must be 'i2sb'")
        self.num_timesteps = int(self.num_timesteps)
        self.beta_max = float(self.beta_max)
        if self.num_timesteps <= 0:
            raise ValueError("fsb.bridge.num_timesteps must be positive")
        if self.beta_max <= 0:
            raise ValueError("fsb.bridge.beta_max must be positive")
        self.beta_schedule = str(self.beta_schedule).lower()
        if self.beta_schedule not in _VALID_FSB_BETA_SCHEDULES:
            raise ValueError(
                f"fsb.bridge.beta_schedule must be one of {sorted(_VALID_FSB_BETA_SCHEDULES)}"
            )
        self.timestep_spacing = str(self.timestep_spacing).lower()
        if self.timestep_spacing not in _VALID_FSB_TIMESTEP_SPACING:
            raise ValueError(
                f"fsb.bridge.timestep_spacing must be one of {sorted(_VALID_FSB_TIMESTEP_SPACING)}"
            )
        self.prediction_type = str(self.prediction_type).lower()
        if self.prediction_type not in _VALID_FSB_PREDICTION_TYPES:
            raise ValueError(
                f"fsb.bridge.prediction_type must be one of {sorted(_VALID_FSB_PREDICTION_TYPES)}"
            )


@dataclass
class FSBInferenceConfig:
    """Multi-step fsb inference controls."""

    custom_timesteps: list[int] = field(default_factory=lambda: [999, 874, 499, 124, 80, 40, 20, 0])
    n_steps: int | None = None
    eta: float = 0.0

    def validate(self) -> None:
        if self.n_steps is not None and int(self.n_steps) <= 0:
            raise ValueError("fsb.inference.n_steps must be positive when set")
        if not isinstance(self.custom_timesteps, list):
            raise ValueError("fsb.inference.custom_timesteps must be a list")
        self.custom_timesteps = [int(v) for v in self.custom_timesteps]
        if any(v < 0 for v in self.custom_timesteps):
            raise ValueError("fsb.inference.custom_timesteps must be non-negative")
        self.eta = float(self.eta)
        if self.eta < 0:
            raise ValueError("fsb.inference.eta must be non-negative")


@dataclass
class FSBConfig:
    """FSB-specific bridge and inference schema."""

    bridge: FSBBridgeConfig = field(default_factory=FSBBridgeConfig)
    inference: FSBInferenceConfig = field(default_factory=FSBInferenceConfig)

    def validate(self) -> None:
        self.bridge.validate()
        self.inference.validate()


@dataclass
class RuntimeConfig:
    """Runtime device and checkpoint selection."""

    checkpoint: str | None = None
    device: str = "cuda"

    def validate(self) -> None:
        if self.checkpoint is not None and not str(self.checkpoint).strip():
            raise ValueError("runtime.checkpoint must be non-empty when set")


@dataclass
class EvaluationFieldMetricsConfig:
    """Field metrics used by offline validation/inference reports."""

    use_volume_weighted_mse: bool = False
    volume_weight_alpha: float = 0.5
    volume_weight_max: float = 1000.0
    wall_layers: int | None = None
    channel_names: list[str] | None = None

    def validate(self) -> None:
        self.use_volume_weighted_mse = bool(self.use_volume_weighted_mse)
        self.volume_weight_alpha = float(self.volume_weight_alpha)
        self.volume_weight_max = float(self.volume_weight_max)
        if self.volume_weight_alpha < 0:
            raise ValueError("evaluation field volume_weight_alpha must be non-negative")
        if self.volume_weight_max <= 0:
            raise ValueError("evaluation field volume_weight_max must be positive")
        if self.wall_layers is not None:
            self.wall_layers = int(self.wall_layers)
            if self.wall_layers <= 0:
                raise ValueError("evaluation field wall_layers must be positive when set")
        if self.channel_names is not None:
            self.channel_names = [str(name) for name in self.channel_names]


@dataclass
class EvaluationForceMetricsConfig:
    """Aerodynamic coefficient metrics used by offline validation."""

    gamma: float = 1.4
    chord_ref: float = 1.0
    area_ref: float = 1.0
    moment_center: tuple[float, float] = (0.25, 0.0)
    compute_viscous: bool = True
    t_inf: float = 300.0

    def validate(self) -> None:
        self.gamma = float(self.gamma)
        self.chord_ref = float(self.chord_ref)
        self.area_ref = float(self.area_ref)
        if len(self.moment_center) != 2:
            raise ValueError("evaluation.force_metrics.moment_center must contain two values")
        self.moment_center = (float(self.moment_center[0]), float(self.moment_center[1]))
        self.compute_viscous = bool(self.compute_viscous)
        self.t_inf = float(self.t_inf)
        if self.gamma <= 0 or self.chord_ref <= 0 or self.area_ref <= 0 or self.t_inf <= 0:
            raise ValueError("evaluation force gamma/chord_ref/area_ref/t_inf must be positive")


@dataclass
class EvaluationResidualMetricsConfig:
    """PDE residual metrics used by offline validation."""

    targets: tuple[str, ...] = ("pred",)
    weights: dict[str, float] = field(default_factory=dict)
    wall_layers: int | None = None
    spatial_wall_layers: int | None = None
    periodic_xi: bool = True
    dtype: str | None = None
    preserve_residual_dtype: bool = False
    state_is_adflow_consistent: bool = False
    state_is_adflow_mixed: bool = False
    return_components: bool = True

    def validate(self) -> None:
        if isinstance(self.targets, list):
            self.targets = tuple(self.targets)
        self.targets = tuple(str(target).lower() for target in self.targets)
        if not self.targets:
            raise ValueError("evaluation.residual_metrics.targets must not be empty")
        unknown = sorted(set(self.targets) - {"pred", "target"})
        if unknown:
            raise ValueError(f"Unknown evaluation residual target(s): {unknown}")
        self.weights = {str(key): float(value) for key, value in dict(self.weights or {}).items()}
        for attr in ("wall_layers", "spatial_wall_layers"):
            value = getattr(self, attr)
            if value is not None:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"evaluation.residual_metrics.{attr} must be positive when set")
                setattr(self, attr, value)
        self.periodic_xi = bool(self.periodic_xi)
        if self.dtype is not None:
            self.dtype = str(self.dtype).lower()
        self.preserve_residual_dtype = bool(self.preserve_residual_dtype)
        self.state_is_adflow_consistent = bool(self.state_is_adflow_consistent)
        self.state_is_adflow_mixed = bool(self.state_is_adflow_mixed)
        self.return_components = bool(self.return_components)


@dataclass
class EvaluationRichConfig:
    """Explicit offline validation controls.

    Training validation remains trainer-owned and lightweight; this schema is
    only used by validate/infer workflows.
    """

    compute_physical_field_metrics: bool = False
    compute_forces: bool = False
    compute_residuals: bool = False
    record_samples: bool = False
    max_batches: int | None = None
    inverse_transform_for_physics: bool = True
    field_metrics: EvaluationFieldMetricsConfig = field(default_factory=EvaluationFieldMetricsConfig)
    physical_field_metrics: EvaluationFieldMetricsConfig = field(default_factory=EvaluationFieldMetricsConfig)
    force_metrics: EvaluationForceMetricsConfig = field(default_factory=EvaluationForceMetricsConfig)
    residual_metrics: EvaluationResidualMetricsConfig = field(default_factory=EvaluationResidualMetricsConfig)

    def validate(self) -> None:
        self.compute_physical_field_metrics = bool(self.compute_physical_field_metrics)
        self.compute_forces = bool(self.compute_forces)
        self.compute_residuals = bool(self.compute_residuals)
        self.record_samples = bool(self.record_samples)
        if self.max_batches is not None:
            self.max_batches = int(self.max_batches)
            if self.max_batches <= 0:
                raise ValueError("evaluation.rich.max_batches must be positive when set")
        self.inverse_transform_for_physics = bool(self.inverse_transform_for_physics)
        self.field_metrics.validate()
        self.physical_field_metrics.validate()
        self.force_metrics.validate()
        self.residual_metrics.validate()


@dataclass
class EvaluationConfig:
    """Offline evaluation controls."""

    output_dir: str = "outputs/evaluation"
    rich: EvaluationRichConfig = field(default_factory=EvaluationRichConfig)

    def validate(self) -> None:
        if not str(self.output_dir).strip():
            raise ValueError("evaluation.output_dir must be non-empty")
        self.rich.validate()


@dataclass
class NKResumeConfig:
    """ADFLOW resume controls for NK experiments."""

    enabled: bool = False
    backend: str = "adflow"

    def validate(self) -> None:
        self.backend = str(self.backend).lower()
        if self.backend != "adflow":
            raise ValueError("nk_resume.backend must be 'adflow'")


@dataclass
class ExperimentConfig:
    """Complete clean Surrogate-Newton CFD surrogate configuration."""

    experiment: ExperimentInfo = field(default_factory=ExperimentInfo)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(index_path=""))
    task: TaskConfig = field(default_factory=TaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fsb: FSBConfig = field(default_factory=FSBConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    nk_resume: NKResumeConfig = field(default_factory=NKResumeConfig)

    @property
    def experiment_name(self) -> str:
        return self.experiment.name

    @property
    def seed(self) -> int:
        return self.experiment.seed

    @property
    def device(self) -> str:
        return self.runtime.device

    def validate(self) -> None:
        self.experiment.validate()
        self.model.validate()
        self.data.validate()
        self.task.validate()
        self.training.validate()
        if self.model.family == "fsb":
            self.fsb.validate()
        self.runtime.validate()
        self.evaluation.validate()
        self.nk_resume.validate()
        if self.task.kind == "nk_resume" and not self.nk_resume.enabled:
            raise ValueError("task.kind='nk_resume' requires nk_resume.enabled=true")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self)


class ConfigManager:
    """Load and save clean Surrogate-Newton CFD surrogate configs."""

    @staticmethod
    def load_config(config_path: str | Path) -> ExperimentConfig:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix in {".yaml", ".yml"}:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return ConfigManager.dict_to_config(config_dict or {})

    @staticmethod
    def dict_to_config(config_dict: dict[str, Any]) -> ExperimentConfig:
        if not isinstance(config_dict, dict):
            raise ValueError("config root must be a mapping")
        _reject_unknown_top_level_keys(config_dict)

        experiment = _build_dataclass(ExperimentInfo, config_dict.get("experiment", {}))
        model = _build_dataclass(ModelConfig, config_dict.get("model", {}))
        data = _build_dataclass(DataConfig, config_dict.get("data", {}))
        task = _build_dataclass(TaskConfig, config_dict.get("task", {}))
        training = _build_training_config(config_dict.get("training", {}))
        fsb = _build_fsb_config(config_dict.get("fsb", {}))

        runtime = _build_dataclass(RuntimeConfig, config_dict.get("runtime", {}))
        evaluation = _build_evaluation_config(config_dict.get("evaluation", {}))
        nk_resume = _build_dataclass(NKResumeConfig, config_dict.get("nk_resume", {}))

        config = ExperimentConfig(
            experiment=experiment,
            model=model,
            data=data,
            task=task,
            training=training,
            fsb=fsb,
            runtime=runtime,
            evaluation=evaluation,
            nk_resume=nk_resume,
        )
        config.validate()
        return config

    @staticmethod
    def save_config(config: ExperimentConfig, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = config.to_dict()
        with open(save_path, "w", encoding="utf-8") as f:
            if save_path.suffix in {".yaml", ".yml"}:
                yaml.safe_dump(config_dict, f, sort_keys=False, default_flow_style=False)
            elif save_path.suffix == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported save format: {save_path.suffix}")

    @staticmethod
    def create_default_configs() -> dict[str, ExperimentConfig]:
        return {
            "direct_fno": ExperimentConfig(
                experiment=ExperimentInfo(name="direct_fno_default"),
                model=ModelConfig(family="direct", backbone="fno"),
                data=DataConfig(index_path="datasets/index.csv"),
            ),
            "fsb_dit": ExperimentConfig(
                experiment=ExperimentInfo(name="fsb_dit_default"),
                model=ModelConfig(family="fsb", backbone="dit"),
                data=DataConfig(index_path="datasets/index.csv"),
            ),
        }


def load_config(config_path: str | Path) -> ExperimentConfig:
    return ConfigManager.load_config(config_path)


def save_config(config: ExperimentConfig, save_path: str | Path) -> None:
    ConfigManager.save_config(config, save_path)


def _build_dataclass(cls: type, values: dict[str, Any]) -> Any:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} input must be a mapping")
    field_names = set(cls.__dataclass_fields__.keys())
    unknown = sorted(set(values.keys()) - field_names)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {unknown}")
    return cls(**values)


def _build_training_config(values: dict[str, Any]) -> TrainingConfig:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("TrainingConfig input must be a mapping")
    training_values = dict(values)
    loss = _build_dataclass(TrainingLossConfig, training_values.pop("loss", {}))
    training_values["loss"] = loss
    return _build_dataclass(TrainingConfig, training_values)


def _build_fsb_config(values: dict[str, Any]) -> FSBConfig:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("FSBConfig input must be a mapping")
    allowed = set(FSBConfig.__dataclass_fields__.keys())
    unknown = sorted(set(values.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown fsb config keys: {unknown}")
    return FSBConfig(
        bridge=_build_dataclass(FSBBridgeConfig, values.get("bridge", {})),
        inference=_build_dataclass(FSBInferenceConfig, values.get("inference", {})),
    )


def _build_evaluation_config(values: dict[str, Any]) -> EvaluationConfig:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("EvaluationConfig input must be a mapping")
    allowed = set(EvaluationConfig.__dataclass_fields__.keys())
    unknown = sorted(set(values.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown evaluation config keys: {unknown}")
    rich_values = dict(values.get("rich", {}) or {})
    if not isinstance(rich_values, dict):
        raise ValueError("evaluation.rich input must be a mapping")
    allowed_rich = set(EvaluationRichConfig.__dataclass_fields__.keys())
    unknown_rich = sorted(set(rich_values.keys()) - allowed_rich)
    if unknown_rich:
        raise ValueError(f"Unknown evaluation.rich config keys: {unknown_rich}")
    rich_values["field_metrics"] = _build_dataclass(
        EvaluationFieldMetricsConfig,
        rich_values.get("field_metrics", {}),
    )
    rich_values["physical_field_metrics"] = _build_dataclass(
        EvaluationFieldMetricsConfig,
        rich_values.get("physical_field_metrics", {}),
    )
    rich_values["force_metrics"] = _build_dataclass(
        EvaluationForceMetricsConfig,
        rich_values.get("force_metrics", {}),
    )
    rich_values["residual_metrics"] = _build_dataclass(
        EvaluationResidualMetricsConfig,
        rich_values.get("residual_metrics", {}),
    )
    return EvaluationConfig(
        output_dir=values.get("output_dir", "outputs/evaluation"),
        rich=_build_dataclass(EvaluationRichConfig, rich_values),
    )


def _reject_unknown_top_level_keys(config_dict: dict[str, Any]) -> None:
    allowed = set(ExperimentConfig.__dataclass_fields__.keys())
    unknown = sorted(set(config_dict.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {unknown}")

    model = config_dict.get("model", {})
    if isinstance(model, dict):
        allowed_model = set(ModelConfig.__dataclass_fields__.keys())
        unknown_model = sorted(set(model.keys()) - allowed_model)
        if unknown_model:
            raise ValueError(f"Unknown model config keys: {unknown_model}")


def _to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain_dict(asdict(value))
    if isinstance(value, dict):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_dict(v) for v in value]
    return value
