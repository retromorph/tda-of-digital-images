from src.datasets.base import DatasetMeta
from src.datasets.loaders.fingerprints import get_nist_sd04_dataset
from src.registry import DATASETS


@DATASETS("NIST-SD04")
def nist_sd04_loader():
    meta = DatasetMeta(task="classification", n_classes=5, color="gray", image_size=(28, 28))

    def loader(seed, fractions):
        test_fraction = fractions[1]
        return (
            get_nist_sd04_dataset(train=True, seed=seed, test_fraction=test_fraction),
            get_nist_sd04_dataset(train=False, seed=seed, test_fraction=test_fraction),
            meta,
        )

    loader.meta = meta
    return loader
