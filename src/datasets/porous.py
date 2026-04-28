import csv
from pathlib import Path

import numpy as np
import torch

from PIL import Image

from src.datasets.types import ImageDataset

POROUS_ROOT = "./data/2D-porous-media-images"
POROUS_IMAGES_SUBDIR = "porous_media_images"
POROUS_CSV_FILENAME = "permeability.csv"


def _read_permeability_csv(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        _header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            idx = int(row[0].strip())
            permeability = float(row[2].strip())
            rows.append((idx, permeability))
    return rows


def _index_images(images_dir):
    files_by_idx = {}
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        stem_digits = "".join(ch for ch in p.stem if ch.isdigit())
        if stem_digits:
            files_by_idx[int(stem_digits)] = p
    return files_by_idx


def _load_images_and_targets(dataset_root, images_subdir, csv_filename):
    root = Path(dataset_root)
    images_dir = root / images_subdir
    csv_path = root / csv_filename

    if not root.exists():
        raise FileNotFoundError(
            f"Porous dataset root not found: '{root}'. "
            "Download https://github.com/gengshaoyang/2D-porous-media-images and place files there."
        )
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: '{images_dir}'.")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: '{csv_path}'.")

    rows = _read_permeability_csv(csv_path)
    files_by_idx = _index_images(images_dir)

    images = []
    targets = []
    missing = []
    for idx, permeability in rows:
        img_path = files_by_idx.get(idx)
        if img_path is None:
            missing.append(idx)
            continue
        img = Image.open(img_path).convert("L")
        images.append(torch.from_numpy(np.array(img, dtype=np.uint8)))
        targets.append(permeability)

    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            "CSV/image mismatch for porous dataset. Missing image files for indexes: "
            f"{preview}{'...' if len(missing) > 10 else ''}"
        )

    data = torch.stack(images)
    y = torch.tensor(targets, dtype=torch.float32)
    return data, y


def get_porous2d_clean_dataset(
    train=True,
    *,
    dataset_root=POROUS_ROOT,
    images_subdir=POROUS_IMAGES_SUBDIR,
    csv_filename=POROUS_CSV_FILENAME,
    seed=0,
    test_fraction=1 / 6,
):
    data, targets = _load_images_and_targets(dataset_root, images_subdir, csv_filename)
    n_total = len(targets)
    n_test = int(n_total * test_fraction)

    if n_test <= 0 or n_test >= n_total:
        raise ValueError("Invalid test_fraction for porous dataset split.")

    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    test_idx = perm[:n_test]
    train_val_idx = perm[n_test:]

    if train:
        return ImageDataset(data[train_val_idx], targets[train_val_idx])
    return ImageDataset(data[test_idx], targets[test_idx])
