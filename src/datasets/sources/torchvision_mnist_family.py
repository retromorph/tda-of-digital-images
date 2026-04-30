from types import SimpleNamespace

from torchvision.datasets import CIFAR10, EMNIST, KMNIST, MNIST, SVHN, FashionMNIST

from src.datasets.base import DatasetMeta
from src.registry import DATASETS


def _build_loader(train_val_fn, test_fn, meta: DatasetMeta):
    def loader(_seed, _fractions):
        return train_val_fn(), test_fn(), meta

    return loader


@DATASETS("MNIST")
def mnist_loader():
    return _build_loader(
        lambda: MNIST(root="./data/image", train=True, download=True),
        lambda: MNIST(root="./data/image", train=False, download=True),
        DatasetMeta(task="classification", n_classes=10, color="gray", image_size=(28, 28)),
    )


@DATASETS("MNIST-RGB")
def mnist_rgb_loader():
    return _build_loader(
        lambda: MNIST(root="./data/image", train=True, download=True),
        lambda: MNIST(root="./data/image", train=False, download=True),
        DatasetMeta(task="classification", n_classes=10, color="rgb", image_size=(28, 28)),
    )


@DATASETS("KMNIST")
def kmnist_loader():
    return _build_loader(
        lambda: KMNIST(root="./data/image", train=True, download=True),
        lambda: KMNIST(root="./data/image", train=False, download=True),
        DatasetMeta(task="classification", n_classes=10, color="gray", image_size=(28, 28)),
    )


@DATASETS("EMNIST-B")
def emnist_b_loader():
    return _build_loader(
        lambda: EMNIST(root="./data/image", split="balanced", train=True, download=True),
        lambda: EMNIST(root="./data/image", split="balanced", train=False, download=True),
        DatasetMeta(task="classification", n_classes=47, color="gray", image_size=(28, 28)),
    )


@DATASETS("EMNIST-L")
def emnist_l_loader():
    return _build_loader(
        lambda: EMNIST(root="./data/image", split="letters", train=True, download=True),
        lambda: EMNIST(root="./data/image", split="letters", train=False, download=True),
        DatasetMeta(task="classification", n_classes=26, color="gray", image_size=(28, 28)),
    )


@DATASETS("FMNIST")
def fmnist_loader():
    return _build_loader(
        lambda: FashionMNIST(root="./data/image", train=True, download=True),
        lambda: FashionMNIST(root="./data/image", train=False, download=True),
        DatasetMeta(task="classification", n_classes=10, color="gray", image_size=(28, 28)),
    )


@DATASETS("CIFAR-10")
def cifar10_loader():
    return _build_loader(
        lambda: CIFAR10(root="./data/image", train=True, download=True),
        lambda: CIFAR10(root="./data/image", train=False, download=True),
        DatasetMeta(task="classification", n_classes=10, color="rgb", image_size=(32, 32)),
    )


@DATASETS("CIFAR-10-GRAY")
def cifar10_gray_loader():
    return _build_loader(
        lambda: CIFAR10(root="./data/image", train=True, download=True),
        lambda: CIFAR10(root="./data/image", train=False, download=True),
        DatasetMeta(task="classification", n_classes=10, color="gray", image_size=(32, 32)),
    )


@DATASETS("SVHN")
def svhn_loader():
    return _build_loader(
        lambda: SVHN(root="./data/image", split="train", download=True),
        lambda: SVHN(root="./data/image", split="test", download=True),
        DatasetMeta(task="classification", n_classes=10, color="rgb", image_size=(32, 32)),
    )
