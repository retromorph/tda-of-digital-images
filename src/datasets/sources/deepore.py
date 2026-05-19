import os

from src.datasets.base import DatasetMeta
from src.datasets.loaders.deepore import get_deepore_3d_datasets
from src.registry import DATASETS


def _shape_from_env() -> tuple[int, ...]:
    """Allow OOM-constrained boxes to override DeepOre target volume size."""
    raw = os.environ.get("DEEPORE_TARGET_SIZE")
    if not raw:
        return (64, 64, 64)
    parts = raw.lower().split("x")
    return tuple(int(p) for p in parts)


@DATASETS("DEEPORE-3D")
def deepore_loader():
    shape = _shape_from_env()
    meta = DatasetMeta(
        task="regression",
        n_classes=None,
        color="gray",
        image_size=shape,
    )

    def loader(seed, fractions):
        # Env-var overrides for memory-constrained machines (no thread-through plumbing needed).
        max_samples_raw = os.environ.get("DEEPORE_MAX_SAMPLES")
        max_samples = int(max_samples_raw) if max_samples_raw else None
        test_fraction = fractions[1]
        train_ds, test_ds = get_deepore_3d_datasets(
            seed=seed,
            test_fraction=test_fraction,
            target_size=meta.image_size,
            target_index=0,
            log_target=True,
            max_samples=max_samples,
        )
        return train_ds, test_ds, meta

    loader.meta = meta
    return loader
