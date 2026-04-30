from dataclasses import dataclass

from src.datasets import (
    ImageDatasetConfig,
    PersistenceDatasetConfig,
    get_image_dataset,
    get_persistence_dataset,
)
from src.experiment.bootstrap import (
    build_collate,
    make_dataloader,
    select_dataset_output_by_input_kind,
)
from src.models.base import get_model_spec


@dataclass(frozen=True)
class RegistryRunArtifacts:
    model: object
    meta: object
    dataloader_train: object
    dataloader_val: object
    dataloader_test: object
    forward_takes_mask: bool


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
