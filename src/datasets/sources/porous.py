from src.datasets.base import DatasetMeta
from src.datasets.porous import get_porous2d_clean_dataset
from src.registry import DATASETS


@DATASETS("POROUS2D-CLEAN")
def porous_loader():
    meta = DatasetMeta(task="regression", n_classes=None, color="gray", image_size=(28, 28))

    def loader(seed, fractions):
        test_fraction = fractions[1]
        return (
            get_porous2d_clean_dataset(train=True, seed=seed, test_fraction=test_fraction),
            get_porous2d_clean_dataset(train=False, seed=seed, test_fraction=test_fraction),
            meta,
        )

    return loader
