import torch
from medmnist import OrganAMNIST
from types import SimpleNamespace

from src.datasets.base import DatasetMeta
from src.registry import DATASETS


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


@DATASETS("OrganAMNIST")
def organamnist_loader():
    meta = DatasetMeta(task="classification", n_classes=11, color="gray", image_size=(28, 28))

    def loader(_seed, _fractions):
        return _get_organamnist_train_val(), _get_organamnist_test(), meta

    return loader
