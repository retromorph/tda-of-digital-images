import time
from dataclasses import asdict, dataclass

from src.datasets import (
    ImageDatasetConfig,
    PersistenceDatasetConfig,
    apply_target_standardization,
    get_image_dataset,
    get_persistence_dataset,
)
from src.experiment.bootstrap import (
    build_collate,
    build_mlflow_logger,
    make_dataloader,
    resolve_device,
    select_dataset_output_by_input_kind,
    update_runtime_metrics,
)
from src.fixed_encoders import EncodedFeatureDataset
from src.encoders import create_encoder
from src.models.base import get_model_spec
from src.trainer import OptimConfig, TrainConfig, Trainer


@dataclass(frozen=True)
class RegistryRunArtifacts:
    model: object
    meta: object
    dataloader_train: object
    dataloader_val: object
    dataloader_test: object
    forward_takes_mask: bool


@dataclass(frozen=True)
class RunResult:
    history: dict
    meta: object
    model: object
    run_name: str
    experiment_name: str
    epochs_completed: int
    stopped_early: bool
    stopped_by_budget: bool = False
    total_steps: int = 0


def _sin_encoding_from_name(name: str | None) -> list[float] | None:
    if name is None:
        return None
    if name == "sin5":
        return [40.0, 90.0, 180.0, 130.0, 360.0]
    raise ValueError("Unknown filtration.positional_encoding '{}'.".format(name))


def _flatten_dict(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
            _flatten_dict(value, next_prefix, out)
        return out
    out[prefix] = obj
    return out


def _coerce_decoder_dims(model_kwargs: dict):
    value = model_kwargs.get("decoder_hidden_dims")
    if isinstance(value, str):
        model_kwargs["decoder_hidden_dims"] = tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _build_data_kwargs(cfg) -> dict:
    dataset_args = dict(cfg.dataset.args or {})
    data_kwargs = {
        "dataset_str": cfg.dataset.name,
        "seed": int(cfg.seed),
        "transform_str": cfg.dataset.test_time.transform,
        "power": float(cfg.dataset.test_time.power),
    }
    if "fractions" in dataset_args:
        data_kwargs["fractions"] = tuple(dataset_args["fractions"])
    return data_kwargs


def _build_persistence_kwargs(cfg) -> dict:
    data_kwargs = _build_data_kwargs(cfg)
    filtration = cfg.filtration
    if filtration is None:
        raise ValueError("filtration config is required for diagram/encoded models.")
    data_kwargs.update(
        {
            "idx": filtration.diagram_idx,
            "eps": (filtration.args or {}).get("eps"),
            "filtration": filtration.name,
            "filtration_params": dict(filtration.args or {}),
            "sin_encoding_config": _sin_encoding_from_name(filtration.positional_encoding),
        }
    )
    return data_kwargs


def build_from_registry(model_name: str, model_kwargs: dict, data_kwargs: dict, batch_size: int, num_workers: int):
    spec = get_model_spec(model_name)
    input_kind = spec.input_kind

    if input_kind in {"image", "flat"}:
        output = select_dataset_output_by_input_kind(input_kind)
        ds_train, ds_val, ds_test, meta = get_image_dataset(
            ImageDatasetConfig(output=output, **data_kwargs),
        )
        collate = None
    elif input_kind == "diagram":
        ds_train, ds_val, ds_test, meta = get_persistence_dataset(
            PersistenceDatasetConfig(**data_kwargs),
        )
        collate = build_collate(
            getattr(meta, "task", "classification"),
            idx=data_kwargs.get("idx"),
            eps=data_kwargs.get("eps"),
            sin_encoding_config=data_kwargs.get("sin_encoding_config"),
        )
    else:
        raise ValueError("Unsupported input_kind '{}' for unified registry runner.".format(input_kind))

    ds_train, ds_val, ds_test, meta = apply_target_standardization(ds_train, ds_val, ds_test, meta)

    model = spec.build(meta, **model_kwargs)
    dl_train = make_dataloader(ds_train, batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    dl_val = make_dataloader(ds_val, batch_size, collate_fn=collate, num_workers=num_workers)
    dl_test = make_dataloader(ds_test, batch_size, collate_fn=collate, num_workers=num_workers)
    return RegistryRunArtifacts(
        model=model,
        meta=meta,
        dataloader_train=dl_train,
        dataloader_val=dl_val,
        dataloader_test=dl_test,
        forward_takes_mask=spec.forward_takes_mask,
    )


def _build_encoded_from_registry(cfg, spec, model_kwargs, batch_size: int, num_workers: int):
    data_kwargs = _build_persistence_kwargs(cfg)
    ds_train, ds_val, ds_test, meta = get_persistence_dataset(PersistenceDatasetConfig(**data_kwargs))
    ds_train, ds_val, ds_test, meta = apply_target_standardization(ds_train, ds_val, ds_test, meta)

    if cfg.encoder is None:
        raise ValueError("encoder section is required when model input_kind is 'encoded'.")
    encoder = create_encoder(cfg.encoder.name, **dict(cfg.encoder.args or {}))
    base_cache_config = {
        "dataset_str": data_kwargs["dataset_str"],
        "seed": data_kwargs["seed"],
        "idx": data_kwargs.get("idx"),
        "eps": data_kwargs.get("eps"),
        "transform_str": data_kwargs.get("transform_str"),
        "power": data_kwargs.get("power"),
        "fractions": list(data_kwargs.get("fractions", (5 / 6, 1 / 6))),
        "filtration": data_kwargs.get("filtration"),
        "filtration_params": data_kwargs.get("filtration_params", {}),
    }
    train_features = EncodedFeatureDataset(ds_train, encoder, split="train", base_cache_config=base_cache_config)
    val_features = EncodedFeatureDataset(ds_val, encoder, split="val", base_cache_config=base_cache_config)
    test_features = EncodedFeatureDataset(ds_test, encoder, split="test", base_cache_config=base_cache_config)

    model = spec.build(meta, **model_kwargs)
    dl_train = make_dataloader(train_features, batch_size, shuffle=True, collate_fn=None, num_workers=num_workers)
    dl_val = make_dataloader(val_features, batch_size, collate_fn=None, num_workers=num_workers)
    dl_test = make_dataloader(test_features, batch_size, collate_fn=None, num_workers=num_workers)
    return RegistryRunArtifacts(
        model=model,
        meta=meta,
        dataloader_train=dl_train,
        dataloader_val=dl_val,
        dataloader_test=dl_test,
        forward_takes_mask=spec.forward_takes_mask,
    )


def _make_optim_config(cfg, lr: float) -> OptimConfig:
    optimizer_cfg = cfg.training.optimizer or {}
    scheduler_cfg = cfg.training.scheduler or {}
    eta_min_raw = float(scheduler_cfg.get("eta_min", 0.0))
    eta_min_ratio = (eta_min_raw / lr) if (lr > 0 and eta_min_raw > 0) else 0.0
    return OptimConfig(
        optimizer=str(optimizer_cfg.get("name", "adamw")),
        lr=lr,
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        scheduler=str(scheduler_cfg.get("name", "none")),
        warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 0)),
        eta_min=eta_min_ratio,
    )


def _make_train_config(cfg) -> TrainConfig:
    budget_cfg = cfg.training.budget
    kind = str(budget_cfg.kind)
    value = int(budget_cfg.value)
    epochs_hint = getattr(budget_cfg, "epochs_hint", None)
    # For epoch budgets the outer loop runs exactly `value` epochs.
    # For steps/seconds budgets, `epochs_hint` caps the outer loop and sets the
    # scheduler horizon; the actual stop is driven by the per-step check.
    n_epochs = value if kind == "epochs" else int(epochs_hint)
    early = cfg.training.early_stop
    log_cfg = cfg.logging
    bw_path = getattr(log_cfg, "best_weights_artifact_path", None) or "weights/best.pt"
    snapshots_raw = getattr(budget_cfg, "snapshots", None)
    snapshots: list[int] | None = None
    if snapshots_raw is not None:
        snapshots = [int(s) for s in snapshots_raw]
    return TrainConfig(
        epochs=n_epochs,
        budget_kind=kind,
        budget_value=value,
        batch_size=int(cfg.training.batch_size),
        grad_accum=int(getattr(cfg.training, "grad_accum", 1) or 1),
        max_grad_norm=float(getattr(cfg.training, "max_grad_norm", 0.0) or 0.0),
        early_stop_patience=int(early.patience),
        early_stop_metric=str(early.metric),
        early_stop_min_delta=float(early.min_delta),
        save_best_weights=bool(getattr(log_cfg, "save_best_weights", False)),
        best_weights_artifact_path=str(bw_path),
        budget_snapshots=snapshots,
    )


def run_experiment(cfg) -> RunResult:
    spec = get_model_spec(cfg.model.name)
    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.num_workers)
    model_kwargs = dict(cfg.model.args or {})
    _coerce_decoder_dims(model_kwargs)

    if spec.input_kind == "encoded":
        artifacts = _build_encoded_from_registry(cfg, spec, model_kwargs, batch_size, num_workers)
    elif spec.input_kind == "diagram":
        artifacts = build_from_registry(
            cfg.model.name,
            model_kwargs=model_kwargs,
            data_kwargs=_build_persistence_kwargs(cfg),
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        artifacts = build_from_registry(
            cfg.model.name,
            model_kwargs=model_kwargs,
            data_kwargs=_build_data_kwargs(cfg),
            batch_size=batch_size,
            num_workers=num_workers,
        )

    flat_params = _flatten_dict(asdict(cfg))
    run_name = str(cfg.run_name)
    experiment_name = str(cfg.logging.experiment)
    task = getattr(artifacts.meta, "task", "classification")

    tags = {
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "seed": int(cfg.seed),
        "task": task,
    }
    target_stats = getattr(artifacts.meta, "target_stats", None)
    if target_stats is not None:
        tags["target_mean"] = float(target_stats["mean"])
        tags["target_std"] = float(target_stats["std"])
    if cfg.logging.tags:
        tags.update(dict(cfg.logging.tags))

    sample_batch = next(iter(artifacts.dataloader_train))
    logger = build_mlflow_logger(
        experiment_name=experiment_name,
        run_name=run_name,
        params=flat_params,
        tags=tags,
        model=artifacts.model,
        sample_batch=sample_batch,
        forward_takes_mask=artifacts.forward_takes_mask,
    )

    device = resolve_device(cfg.device)
    trainer = Trainer(
        artifacts.model,
        device=device,
        logger=logger,
        task=task,
        forward_takes_mask=artifacts.forward_takes_mask,
        target_stats=getattr(artifacts.meta, "target_stats", None),
    )
    optim_cfg = _make_optim_config(cfg, lr=float((cfg.training.optimizer or {}).get("lr", 1e-3)))
    train_cfg = _make_train_config(cfg)

    started_at = time.time()
    try:
        outcome = trainer.fit(
            artifacts.dataloader_train,
            artifacts.dataloader_val,
            artifacts.dataloader_test,
            optim_config=optim_cfg,
            train_config=train_cfg,
            desc="{}, {}".format(cfg.model.name, cfg.seed),
        )
        epochs_completed = int(outcome["epochs_completed"])
        stopped_early = bool(outcome["stopped_early"])
        stopped_by_budget = bool(outcome.get("stopped_by_budget", False))
        total_steps = int(outcome.get("total_steps", 0))
        final_step = max(0, epochs_completed - 1)
        update_runtime_metrics(logger, started_at, device=device, step=final_step)
        logger.log(
            {"total_steps": total_steps, "stopped_by_budget": int(stopped_by_budget)},
            step=final_step,
        )
    finally:
        logger.end()

    return RunResult(
        history=trainer.history,
        meta=artifacts.meta,
        model=artifacts.model,
        run_name=run_name,
        experiment_name=experiment_name,
        epochs_completed=epochs_completed,
        stopped_early=stopped_early,
        stopped_by_budget=stopped_by_budget,
        total_steps=total_steps,
    )
