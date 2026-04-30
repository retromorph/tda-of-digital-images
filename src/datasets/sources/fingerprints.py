from src.datasets.base import DatasetMeta
from src.datasets.fingerprints import get_nist_sd04_dataset
from src.registry import DATASETS


@DATASETS("NIST-SD04")
def nist_sd04_loader():
    meta = DatasetMeta(task="classification", n_classes=5, color="gray", image_size=(28, 28))

    def loader(_seed, _fractions):
        return get_nist_sd04_dataset(train=True), get_nist_sd04_dataset(train=False), meta

    return loader
