from types import SimpleNamespace

from torchvision.datasets import CIFAR10, EMNIST, KMNIST, MNIST, SVHN, FashionMNIST

from src.datasets.synthetic import get_blobs_dataset


def get_dataset_cfg(dataset_str):
    datasets = {
        "MNIST": {
            "dataset_train_val": lambda: MNIST(root="./data/image", train=True, download=True),
            "dataset_test": lambda: MNIST(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10),
        },
        "KMNIST": {
            "dataset_train_val": lambda: KMNIST(root="./data/image", train=True, download=True),
            "dataset_test": lambda: KMNIST(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10),
        },
        "EMNIST-B": {
            "dataset_train_val": lambda: EMNIST(root="./data/image", split="balanced", train=True, download=True),
            "dataset_test": lambda: EMNIST(root="./data/image", split="balanced", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=47),
        },
        "EMNIST-L": {
            "dataset_train_val": lambda: EMNIST(root="./data/image", split="letters", train=True, download=True),
            "dataset_test": lambda: EMNIST(root="./data/image", split="letters", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=26),
        },
        "FMNIST": {
            "dataset_train_val": lambda: FashionMNIST(root="./data/image", train=True, download=True),
            "dataset_test": lambda: FashionMNIST(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10),
        },
        "BLOBS": {
            "dataset_train_val": lambda: get_blobs_dataset(train=True),
            "dataset_test": lambda: get_blobs_dataset(train=False),
            "meta": SimpleNamespace(dim=64, n_classes=2),
        },
        "CIFAR-10": {
            "dataset_train_val": lambda: CIFAR10(root="./data/image", train=True, download=True),
            "dataset_test": lambda: CIFAR10(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10),
        },
        "SVHN": {
            "dataset_train_val": lambda: SVHN(root="./data/image", split="train", download=True),
            "dataset_test": lambda: SVHN(root="./data/image", split="test", download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10),
        },
    }

    if dataset_str not in datasets:
        raise ValueError("Supported datasets are '{}'.".format("', '".join(datasets.keys())))
    return datasets[dataset_str]
