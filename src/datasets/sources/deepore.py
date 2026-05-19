from src.datasets.base import DatasetMeta
from src.datasets.loaders.deepore import get_deepore_3d_datasets
from src.registry import DATASETS


@DATASETS("DEEPORE-3D")
def deepore_loader():
    meta = DatasetMeta(
        task="regression",
        n_classes=None,
        color="gray",
        image_size=(64, 64, 64),
    )

    def loader(seed, fractions):
        test_fraction = fractions[1]
        train_ds, test_ds = get_deepore_3d_datasets(
            seed=seed,
            test_fraction=test_fraction,
            target_size=meta.image_size,
            target_index=0,
            log_target=True,
        )
        return train_ds, test_ds, meta

    return loader
