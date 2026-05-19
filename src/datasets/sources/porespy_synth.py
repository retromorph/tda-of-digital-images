import os

from src.datasets.base import DatasetMeta
from src.datasets.loaders.porespy_synth import (
    DEFAULT_MASTER_SEED,
    DEFAULT_N_SAMPLES,
    DEFAULT_SHAPE,
    get_porespy_synth_datasets,
)
from src.registry import DATASETS


def _shape_from_env() -> tuple[int, ...]:
    """Allow smoke configs to run on smaller volumes via ``PORESPY_SYNTH_SHAPE=DxHxW``."""
    raw = os.environ.get("PORESPY_SYNTH_SHAPE")
    if not raw:
        return DEFAULT_SHAPE
    parts = raw.lower().split("x")
    return tuple(int(p) for p in parts)


@DATASETS("PORESPY-3D")
def porespy_synth_loader():
    shape = _shape_from_env()
    meta = DatasetMeta(
        task="regression",
        n_classes=None,
        color="gray",
        image_size=shape,
    )

    def loader(seed, fractions):
        # Env-var overrides keep smoke configs cheap without thread-through plumbing.
        n_samples = int(os.environ.get("PORESPY_SYNTH_N_SAMPLES", DEFAULT_N_SAMPLES))
        master_seed = int(os.environ.get("PORESPY_SYNTH_MASTER_SEED", DEFAULT_MASTER_SEED))
        test_fraction = fractions[1]
        train_ds, test_ds = get_porespy_synth_datasets(
            seed=seed,
            test_fraction=test_fraction,
            target_size=meta.image_size,
            n_samples=n_samples,
            master_seed=master_seed,
        )
        return train_ds, test_ds, meta

    loader.meta = meta
    return loader
