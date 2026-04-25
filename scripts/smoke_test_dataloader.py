"""Quick checks for dataloaders (no PHT precompute by default)."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import collate_fn, get_image_dataset, get_pht_dataset

seed = 0


def smoke_image():
    dataset_train, dataset_val, dataset_test, meta = get_image_dataset(
        "MNIST", seed, transform_str=None, power=0.0, output="2d"
    )
    dl = DataLoader(dataset_train, batch_size=4, shuffle=False, num_workers=0)
    x, y = next(iter(dl))
    print("image batch", x.shape, y.shape, "n_classes=", meta.n_classes)


def smoke_pht_if_cached():
    """Only runs if diagram pickles already exist (avoids long PHT compute)."""
    train_pkl = ROOT / "data" / "diagrams" / "MNIST" / "MNIST_train_seed-0.pkl"
    if not train_pkl.is_file():
        print("PHT smoke skipped (no cached diagrams); run make_datasets or a PHT runner once first.")
        return
    dataset_pht_train, _, _, _ = get_pht_dataset("MNIST", seed)
    dl = DataLoader(dataset_pht_train, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    X, X_mask, Y = next(iter(dl))
    print("PHT batch", X.shape, X_mask.shape, Y.shape)


if __name__ == "__main__":
    smoke_image()
    if "--pht" in sys.argv:
        smoke_pht_if_cached()
