import os
import pickle

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from src.datasets.cache import (
    CACHE_VERSION,
    load_cache,
    save_cache,
    split_cache_path,
    stable_hash,
    write_meta,
)
from src.datasets.transforms import build_image_transforms
from src.datasets.types import ImageDataset, PersistenceDataset
from src.filtrations import FILTRATIONS  # noqa: F401
from src.registry import DATASETS, FILTRATIONS


@dataclass(frozen=True)
class ImageDatasetConfig:
    dataset_str: str
    seed: int
    transform_str: str | None = None
    power: float = 0.0
    fractions: tuple[float, float] = (5 / 6, 1 / 6)
    output: str = "2d"


@dataclass(frozen=True)
class PersistenceDatasetConfig:
    dataset_str: str
    seed: int
    idx: list[int] | None = None
    eps: float | None = None
    transform_str: str | None = None
    power: float = 0.0
    fractions: tuple[float, float] = (5 / 6, 1 / 6)
    filtration: str = "pht_directional"
    filtration_params: dict | None = None
    sin_encoding_config: list[float] | None = None


def _validate_fractions(fractions):
    if len(fractions) != 2:
        raise ValueError("fractions must contain exactly two values: (train, val).")
    f_train, f_val = fractions
    if f_train < 0 or f_val < 0:
        raise ValueError("fractions must be non-negative.")
    if not np.isclose(f_train + f_val, 1.0):
        raise ValueError("fractions must sum to 1.0.")
    return f_train, f_val


def _apply_label_offset(labels, meta):
    offset = int(getattr(meta, "label_offset", 0) or 0)
    if offset == 0:
        return labels
    return labels + offset


def _get_targets_tensor(dataset):
    if hasattr(dataset, "targets"):
        targets = torch.as_tensor(dataset.targets)
    elif hasattr(dataset, "labels"):
        targets = torch.as_tensor(dataset.labels)
    else:
        raise AttributeError("Dataset has neither 'targets' nor 'labels' attributes.")

    if targets.ndim > 1 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)
    if targets.ndim > 1:
        targets = targets.reshape(targets.shape[0], -1)
        if targets.shape[1] == 1:
            targets = targets.squeeze(-1)
    return targets


def _prepare_images(raw_images, meta):
    images = torch.as_tensor(raw_images)

    target_h, target_w = tuple(meta.image_size)

    if images.ndim == 3:
        images = images.unsqueeze(1).float()
    elif images.ndim == 4:
        if images.shape[1] == 3:
            images = images.float()
        elif images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2).float()
        elif images.shape[1] == 1:
            images = images.float()
        elif images.shape[-1] == 1:
            images = images.permute(0, 3, 1, 2).float()
        else:
            raise ValueError(f"Unsupported 4D image tensor shape: {tuple(images.shape)}")
    else:
        raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")

    if meta.color == "gray" and images.shape[1] == 3:
        images = images.mean(dim=1, keepdim=True)
    if meta.color == "rgb" and images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[-2:] != (target_h, target_w):
        images = F.interpolate(images.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)

    return images


def _compute_diagrams(dataset, filtration_apply):
    diagrams = []
    schema = None
    for item in tqdm(dataset.data):
        out = filtration_apply(item)
        diagrams.append(out.points)
        if schema is None:
            schema = out.schema
    return diagrams, (schema or {})


def _load_legacy_cache(path):
    if not os.path.isfile(path):
        return None
    payload = pickle.load(open(path, "rb"))
    if isinstance(payload, tuple) and len(payload) == 2:
        diagrams, labels = payload
        return {"diagrams": diagrams, "labels": labels, "schema": {}}
    if isinstance(payload, dict):
        return {
            "diagrams": payload.get("diagrams", []),
            "labels": payload.get("labels"),
            "schema": payload.get("schema", {}),
        }
    return None


def get_image_dataset(cfg: ImageDatasetConfig):
    if cfg.output not in ["1d", "2d"]:
        raise ValueError("Supported outputs are '1d' or '2d'.")

    f_train, _f_val = _validate_fractions(cfg.fractions)
    transform_train_val, transform_test = build_image_transforms(cfg.transform_str, cfg.power, cfg.output)

    loader = DATASETS.get(cfg.dataset_str)()
    dataset_train_val_raw, dataset_test_raw, meta = loader(cfg.seed, cfg.fractions)

    n_train = int(len(dataset_train_val_raw) * f_train)
    random_idx = torch.randperm(len(dataset_train_val_raw), generator=torch.Generator().manual_seed(cfg.seed))

    train_val_images = _prepare_images(dataset_train_val_raw.data, meta)
    test_images = _prepare_images(dataset_test_raw.data, meta)
    train_val_targets = _get_targets_tensor(dataset_train_val_raw)
    test_targets = _get_targets_tensor(dataset_test_raw)

    x_train = transform_train_val(train_val_images[random_idx[:n_train]])
    y_train = _apply_label_offset(train_val_targets[random_idx[:n_train]], meta)
    x_val = transform_train_val(train_val_images[random_idx[n_train:]])
    y_val = _apply_label_offset(train_val_targets[random_idx[n_train:]], meta)
    x_test = transform_test(test_images)
    y_test = _apply_label_offset(test_targets, meta)

    if getattr(meta, "task", "classification") == "classification":
        y_train = y_train.long()
        y_val = y_val.long()
        y_test = y_test.long()

    dataset_train = ImageDataset(x_train, y_train)
    dataset_val = ImageDataset(x_val, y_val)
    dataset_test = ImageDataset(x_test, y_test)
    return dataset_train, dataset_val, dataset_test, meta


def get_persistence_dataset(cfg: PersistenceDatasetConfig):
    image_cfg = ImageDatasetConfig(
        dataset_str=cfg.dataset_str,
        seed=cfg.seed,
        transform_str=cfg.transform_str,
        power=cfg.power,
        fractions=cfg.fractions,
        output="2d",
    )
    dataset_train, dataset_val, dataset_test, meta = get_image_dataset(image_cfg)

    y_train = dataset_train.targets
    y_val = dataset_val.targets
    y_test = dataset_test.targets

    filt_params = cfg.filtration_params or {}
    cache_key = stable_hash(cfg.filtration, filt_params)
    train_filename = split_cache_path(cfg.dataset_str, cache_key, "train", cfg.seed)
    val_filename = split_cache_path(cfg.dataset_str, cache_key, "val", cfg.seed)
    test_filename = split_cache_path(
        cfg.dataset_str,
        cache_key,
        "test",
        cfg.seed,
        transform_str=cfg.transform_str,
        power=cfg.power,
    )

    filtration_apply = FILTRATIONS.get(cfg.filtration)(filt_params)
    transform_prefix = f"{cfg.transform_str}-{cfg.power}_" if cfg.transform_str is not None else ""
    suffix = f"seed-{cfg.seed}"
    legacy_train = f"./data/diagrams/{cfg.dataset_str}/{cfg.dataset_str}_train_{suffix}.pkl"
    legacy_val = f"./data/diagrams/{cfg.dataset_str}/{cfg.dataset_str}_val_{suffix}.pkl"
    legacy_test = f"./data/diagrams/{cfg.dataset_str}/{cfg.dataset_str}_test_{transform_prefix}{suffix}.pkl"

    def _get_split(split_name, ds, path, legacy_path):
        payload = load_cache(path)
        if payload is not None:
            return payload
        if cfg.filtration == "pht_directional" and len(filt_params) == 0:
            legacy = _load_legacy_cache(legacy_path)
            if legacy is not None:
                return legacy
        print(f"Computing filtration for {split_name}, this can take several minutes...")
        dgm, schema = _compute_diagrams(ds, filtration_apply)
        labels = ds.targets
        payload = {
            "diagrams": dgm,
            "labels": labels,
            "schema": schema,
            "version": CACHE_VERSION,
        }
        save_cache(path, payload)
        write_meta(
            dataset=cfg.dataset_str,
            cache_key=cache_key,
            filtration_name=cfg.filtration,
            filtration_params=filt_params,
            schema=schema,
        )
        return payload

    train_payload = _get_split("train", dataset_train, train_filename, legacy_train)
    val_payload = _get_split("val", dataset_val, val_filename, legacy_val)
    test_payload = _get_split("test", dataset_test, test_filename, legacy_test)

    dataset_pht_train = PersistenceDataset(train_payload["diagrams"], train_payload["labels"], schema=train_payload.get("schema", {}))
    dataset_pht_val = PersistenceDataset(val_payload["diagrams"], val_payload["labels"], schema=val_payload.get("schema", {}))
    dataset_pht_test = PersistenceDataset(test_payload["diagrams"], test_payload["labels"], schema=test_payload.get("schema", {}))
    return dataset_pht_train, dataset_pht_val, dataset_pht_test, meta
