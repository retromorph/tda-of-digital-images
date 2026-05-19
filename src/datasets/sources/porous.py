from src.datasets.base import DatasetMeta
from src.datasets.loaders.porous import get_porous2d_clean_datasets
from src.registry import DATASETS


@DATASETS("POROUS2D-CLEAN")
def porous_loader():
    meta = DatasetMeta(task="regression", n_classes=None, color="gray", image_size=(256, 256))

    def loader(seed, fractions):
        test_fraction = fractions[1]
        train_ds, test_ds = get_porous2d_clean_datasets(
            seed=seed,
            test_fraction=test_fraction,
            target_size=meta.image_size,
        )
        return train_ds, test_ds, meta

    loader.meta = meta
    return loader
