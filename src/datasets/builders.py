import os
import pickle

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from src.datasets.fingerprints import get_nist_sd04_dataset
from src.datasets.porous import get_porous2d_clean_dataset
from src.datasets.registry import get_dataset_cfg
from src.datasets.transforms import build_image_transforms
from src.datasets.types import ImageDataset, PersistenceDataset
from src.persistence import pht
from src.transforms_dir import Direction


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


def _validate_fractions(fractions):
    if len(fractions) != 2:
        raise ValueError("fractions must contain exactly two values: (train, val).")
    f_train, f_val = fractions
    if f_train < 0 or f_val < 0:
        raise ValueError("fractions must be non-negative.")
    if not np.isclose(f_train + f_val, 1.0):
        raise ValueError("fractions must sum to 1.0.")
    return f_train, f_val


def _normalize_emnist_letters_labels(labels, dataset_str):
    if dataset_str == "EMNIST-L":
        return labels - 1
    return labels


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


def _prepare_images(raw_images):
    images = torch.as_tensor(raw_images)

    if images.ndim == 3:
        images = images.unsqueeze(1)
    elif images.ndim == 4:
        if images.shape[1] == 3:
            images = images.float().mean(dim=1, keepdim=True)
        elif images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2).float().mean(dim=1, keepdim=True)
        elif images.shape[1] == 1:
            images = images.float()
        elif images.shape[-1] == 1:
            images = images.permute(0, 3, 1, 2).float()
        else:
            raise ValueError(f"Unsupported 4D image tensor shape: {tuple(images.shape)}")
    else:
        raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")

    if images.shape[-2:] != (28, 28):
        images = F.interpolate(images.float(), size=(28, 28), mode="bilinear", align_corners=False)

    return images


def _build_pht_apply():
    alphas = list(np.linspace(0, 360, 16 + 1)[:-1])
    direction = Direction(alphas, agg="add")

    def pht_apply(image):
        return pht(direction(image), image, pos=alphas)

    return pht_apply


def _compute_or_load_diagrams(dataset, labels, filename, pht_apply):
    if os.path.isfile(filename):
        return pickle.load(open(filename, "rb"))

    diagrams = []
    for item in tqdm(dataset.data):
        diagrams.append(pht_apply(item))
    pickle.dump((diagrams, labels), open(filename, "wb"))
    return diagrams, labels


def get_image_dataset(cfg: ImageDatasetConfig):
    if cfg.output not in ["1d", "2d"]:
        raise ValueError("Supported outputs are '1d' or '2d'.")

    f_train, _f_val = _validate_fractions(cfg.fractions)
    transform_train_val, transform_test = build_image_transforms(cfg.transform_str, cfg.power, cfg.output)

    dataset_cfg = get_dataset_cfg(cfg.dataset_str)
    if cfg.dataset_str == "POROUS2D-CLEAN":
        dataset_train_val_raw = get_porous2d_clean_dataset(train=True, seed=cfg.seed, test_fraction=cfg.fractions[1])
        dataset_test_raw = get_porous2d_clean_dataset(train=False, seed=cfg.seed, test_fraction=cfg.fractions[1])
    elif cfg.dataset_str == "NIST-SD04":
        dataset_train_val_raw = get_nist_sd04_dataset(train=True)
        dataset_test_raw = get_nist_sd04_dataset(train=False)
    else:
        dataset_train_val_raw = dataset_cfg["dataset_train_val"]()
        dataset_test_raw = dataset_cfg["dataset_test"]()
    meta = dataset_cfg["meta"]

    n_train = int(len(dataset_train_val_raw) * f_train)
    random_idx = torch.randperm(len(dataset_train_val_raw), generator=torch.Generator().manual_seed(cfg.seed))

    train_val_images = _prepare_images(dataset_train_val_raw.data)
    test_images = _prepare_images(dataset_test_raw.data)
    train_val_targets = _get_targets_tensor(dataset_train_val_raw)
    test_targets = _get_targets_tensor(dataset_test_raw)

    x_train = transform_train_val(train_val_images[random_idx[:n_train]])
    y_train = _normalize_emnist_letters_labels(train_val_targets[random_idx[:n_train]], cfg.dataset_str)
    x_val = transform_train_val(train_val_images[random_idx[n_train:]])
    y_val = _normalize_emnist_letters_labels(train_val_targets[random_idx[n_train:]], cfg.dataset_str)
    x_test = transform_test(test_images)
    y_test = _normalize_emnist_letters_labels(test_targets, cfg.dataset_str)

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

    transform_prefix = f"{cfg.transform_str}-{cfg.power}_" if cfg.transform_str is not None else ""
    train_filename = f"./data/diagrams/{cfg.dataset_str}/{cfg.dataset_str}_train_seed-{cfg.seed}.pkl"
    val_filename = f"./data/diagrams/{cfg.dataset_str}/{cfg.dataset_str}_val_seed-{cfg.seed}.pkl"
    test_filename = f"./data/diagrams/{cfg.dataset_str}/{cfg.dataset_str}_test_{transform_prefix}seed-{cfg.seed}.pkl"

    pht_apply = _build_pht_apply()

    if not (os.path.isfile(train_filename) and os.path.isfile(val_filename) and os.path.isfile(test_filename)):
        print("Computing PHT of a dataset, this can take several minutes...")
        os.makedirs(f"./data/diagrams/{cfg.dataset_str}", exist_ok=True)

        if not (os.path.isfile(train_filename) and os.path.isfile(val_filename)):
            d_train, y_train = _compute_or_load_diagrams(dataset_train, y_train, train_filename, pht_apply)
            print("Train complete")
            d_val, y_val = _compute_or_load_diagrams(dataset_val, y_val, val_filename, pht_apply)
            print("Val complete")
        else:
            d_train, y_train = pickle.load(open(train_filename, "rb"))
            d_val, y_val = pickle.load(open(val_filename, "rb"))

        if not os.path.isfile(test_filename):
            d_test, y_test = _compute_or_load_diagrams(dataset_test, y_test, test_filename, pht_apply)
            print("Test complete")
        else:
            d_test, y_test = pickle.load(open(test_filename, "rb"))
    else:
        d_train, y_train = pickle.load(open(train_filename, "rb"))
        d_val, y_val = pickle.load(open(val_filename, "rb"))
        d_test, y_test = pickle.load(open(test_filename, "rb"))

    dataset_pht_train = PersistenceDataset(d_train, y_train, idx=cfg.idx, eps=cfg.eps)
    dataset_pht_val = PersistenceDataset(d_val, y_val, idx=cfg.idx, eps=cfg.eps)
    dataset_pht_test = PersistenceDataset(d_test, y_test, idx=cfg.idx, eps=cfg.eps)
    return dataset_pht_train, dataset_pht_val, dataset_pht_test, meta
