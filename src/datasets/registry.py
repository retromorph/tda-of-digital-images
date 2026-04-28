from types import SimpleNamespace

import torch

from medmnist import OrganAMNIST
from torchvision.datasets import CIFAR10, EMNIST, KMNIST, MNIST, SVHN, FashionMNIST

from src.datasets.fingerprints import get_nist_sd04_dataset
from src.datasets.porous import get_porous2d_clean_dataset
from src.datasets.synthetic import get_blobs_dataset


def _extract_medmnist_data_targets(ds):
    data = getattr(ds, "imgs", None)
    targets = getattr(ds, "labels", None)
    if data is None or targets is None:
        raise AttributeError("Unexpected medmnist dataset format: expected 'imgs' and 'labels'.")
    return torch.as_tensor(data), torch.as_tensor(targets)


def _get_organamnist_train_val():
    train_ds = OrganAMNIST(root="./data/image", split="train", download=True)
    val_ds = OrganAMNIST(root="./data/image", split="val", download=True)
    x_train, y_train = _extract_medmnist_data_targets(train_ds)
    x_val, y_val = _extract_medmnist_data_targets(val_ds)
    return SimpleNamespace(data=torch.cat([x_train, x_val], dim=0), targets=torch.cat([y_train, y_val], dim=0))


def _get_organamnist_test():
    test_ds = OrganAMNIST(root="./data/image", split="test", download=True)
    x_test, y_test = _extract_medmnist_data_targets(test_ds)
    return SimpleNamespace(data=x_test, targets=y_test)


def get_dataset_cfg(dataset_str):
    datasets = {
        "MNIST": {
            "dataset_train_val": lambda: MNIST(root="./data/image", train=True, download=True),
            "dataset_test": lambda: MNIST(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10, task="classification"),
        },
        "KMNIST": {
            "dataset_train_val": lambda: KMNIST(root="./data/image", train=True, download=True),
            "dataset_test": lambda: KMNIST(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10, task="classification"),
        },
        "EMNIST-B": {
            "dataset_train_val": lambda: EMNIST(root="./data/image", split="balanced", train=True, download=True),
            "dataset_test": lambda: EMNIST(root="./data/image", split="balanced", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=47, task="classification"),
        },
        "EMNIST-L": {
            "dataset_train_val": lambda: EMNIST(root="./data/image", split="letters", train=True, download=True),
            "dataset_test": lambda: EMNIST(root="./data/image", split="letters", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=26, task="classification"),
        },
        "FMNIST": {
            "dataset_train_val": lambda: FashionMNIST(root="./data/image", train=True, download=True),
            "dataset_test": lambda: FashionMNIST(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10, task="classification"),
        },
        "BLOBS": {
            "dataset_train_val": lambda: get_blobs_dataset(train=True),
            "dataset_test": lambda: get_blobs_dataset(train=False),
            "meta": SimpleNamespace(dim=64, n_classes=2, task="classification"),
        },
        "CIFAR-10": {
            "dataset_train_val": lambda: CIFAR10(root="./data/image", train=True, download=True),
            "dataset_test": lambda: CIFAR10(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10, task="classification"),
        },
        "SVHN": {
            "dataset_train_val": lambda: SVHN(root="./data/image", split="train", download=True),
            "dataset_test": lambda: SVHN(root="./data/image", split="test", download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10, task="classification"),
        },
        "POROUS2D-CLEAN": {
            "dataset_train_val": lambda: get_porous2d_clean_dataset(
                train=True,
            ),
            "dataset_test": lambda: get_porous2d_clean_dataset(
                train=False,
            ),
            "meta": SimpleNamespace(dim=28, task="regression"),
        },
        "NIST-SD04": {
            "dataset_train_val": lambda: get_nist_sd04_dataset(train=True),
            "dataset_test": lambda: get_nist_sd04_dataset(train=False),
            "meta": SimpleNamespace(dim=28, n_classes=5, task="classification"),
        },
        "OrganAMNIST": {
            "dataset_train_val": _get_organamnist_train_val,
            "dataset_test": _get_organamnist_test,
            "meta": SimpleNamespace(dim=28, n_classes=11, task="classification"),
        },
    }

    if dataset_str not in datasets:
        raise ValueError("Supported datasets are '{}'.".format("', '".join(datasets.keys())))
    return datasets[dataset_str]
