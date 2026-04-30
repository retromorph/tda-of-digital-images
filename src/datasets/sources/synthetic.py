from src.datasets.base import DatasetMeta
from src.datasets.synthetic import get_blobs_dataset
from src.registry import DATASETS


@DATASETS("BLOBS")
def blobs_loader():
    meta = DatasetMeta(task="classification", n_classes=2, color="gray", image_size=(64, 64))

    def loader(_seed, _fractions):
        return get_blobs_dataset(train=True), get_blobs_dataset(train=False), meta

    return loader
