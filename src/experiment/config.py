from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf
from src.models.base import get_model_spec


@dataclass
class DatasetTestTimeConfig:
    transform: str | None = None
    power: float = 0.0


@dataclass
class DatasetConfig:
    name: str = "MNIST"
    args: dict = field(default_factory=dict)
    test_time: DatasetTestTimeConfig = field(default_factory=DatasetTestTimeConfig)


@dataclass
class FiltrationConfig:
    name: str = "pht_directional"
    args: dict = field(default_factory=dict)
    diagram_idx: list[int] | None = None
    positional_encoding: str | None = "sin5"


@dataclass
class EncoderConfig:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class ComponentConfig:
    name: str = "PHTS"
    args: dict = field(default_factory=dict)


@dataclass
class BudgetConfig:
    kind: str = "epochs"
    value: int = 10


@dataclass
class EarlyStopConfig:
    metric: str = "loss_val"
    patience: int = 0
    min_delta: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 128
    optimizer: dict = field(default_factory=lambda: {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0})
    scheduler: dict = field(default_factory=lambda: {"name": "none", "warmup_epochs": 0, "eta_min": 0.0})
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    grad_accum: int = 1
    max_grad_norm: float = 0.0


@dataclass
class LoggingConfig:
    experiment: str = "Test/default/default"
    tags: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    run_name: str = "${dataset.name}_${model.name}_${seed}"
    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    filtration: FiltrationConfig | None = field(default_factory=FiltrationConfig)
    encoder: EncoderConfig | None = None
    model: ComponentConfig = field(default_factory=ComponentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _normalize_legacy(cfg: dict) -> dict:
    cfg = dict(cfg)
    dataset = cfg.setdefault("dataset", {})
    dataset.setdefault("args", {})
    test_time = dataset.setdefault("test_time", {})
    if "transform" in cfg:
        test_time.setdefault("transform", cfg.pop("transform"))
    if "power" in cfg:
        test_time.setdefault("power", cfg.pop("power"))

    idx = cfg.pop("idx", None)
    eps = cfg.pop("eps", None)
    if idx is not None or eps is not None:
        filt = cfg.setdefault("filtration", {})
        if idx is not None:
            filt.setdefault("diagram_idx", idx)
        if eps is not None:
            filt_args = filt.setdefault("args", {})
            filt_args.setdefault("eps", eps)

    training = cfg.setdefault("training", {})
    optimizer = training.setdefault("optimizer", {})
    scheduler = training.setdefault("scheduler", {})
    budget = training.setdefault("budget", {})
    if "lr" in cfg:
        optimizer.setdefault("lr", cfg.pop("lr"))
    if "weight_decay" in cfg:
        optimizer.setdefault("weight_decay", cfg.pop("weight_decay"))
    if "warmup_epochs" in cfg:
        scheduler.setdefault("warmup_epochs", cfg.pop("warmup_epochs"))
        scheduler.setdefault("name", "warmup_cosine")
    if "eta_min" in cfg:
        scheduler.setdefault("eta_min", cfg.pop("eta_min"))
        scheduler.setdefault("name", "warmup_cosine")
    if "epochs" in cfg:
        budget.setdefault("kind", "epochs")
        budget.setdefault("value", cfg.pop("epochs"))
    if "batch_size" in cfg:
        training.setdefault("batch_size", cfg.pop("batch_size"))

    return cfg


def _validate_config(cfg: ExperimentConfig) -> None:
    spec = get_model_spec(cfg.model.name)
    if spec.input_kind in {"diagram", "encoded"} and cfg.filtration is None:
        raise ValueError("filtration config is required for model '{}'.".format(cfg.model.name))
    if spec.input_kind == "encoded" and cfg.encoder is None:
        raise ValueError("encoder config is required for encoded model '{}'.".format(cfg.model.name))
    if cfg.training.budget.kind not in {"epochs"}:
        raise ValueError("Unsupported training.budget.kind '{}'.".format(cfg.training.budget.kind))


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    base = OmegaConf.structured(ExperimentConfig)
    from_file = OmegaConf.load(Path(path))
    normalized = OmegaConf.create(_normalize_legacy(OmegaConf.to_container(from_file, resolve=False) or {}))
    merged = OmegaConf.merge(base, normalized)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    cfg = OmegaConf.to_object(merged)
    _validate_config(cfg)
    return cfg
