from pathlib import Path

import numpy as np
import torch

from PIL import Image

from src.datasets.types import ImageDataset

SD04_ROOT = "../../data/NISTSpecialDatabase4/sd04/png_txt"
SD04_FIG_DIRS = [f"figs_{i}" for i in range(8)]
SD04_CLASS_TO_IDX = {"L": 0, "W": 1, "R": 2, "T": 3, "A": 4}


def get_nist_sd04_dataset(train=True):
    data, targets = _load_nist_sd04()
    if train:
        return ImageDataset(data, targets)
    return ImageDataset(data, targets)


def _load_nist_sd04():
    root = Path(SD04_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"NIST SD04 root not found: '{root}'.")

    png_files = []
    for figs_dir_name in SD04_FIG_DIRS:
        figs_dir = root / figs_dir_name
        if not figs_dir.exists():
            raise FileNotFoundError(f"NIST SD04 missing directory: '{figs_dir}'.")
        png_files.extend(sorted(figs_dir.glob("*.png")))

    if not png_files:
        raise ValueError(f"No PNG files found in SD04 under '{root}'.")

    images = []
    labels = []
    for png_path in png_files:
        txt_path = png_path.with_suffix(".txt")
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing TXT metadata for image '{png_path.name}'.")

        class_token = _extract_sd04_class(txt_path)
        if class_token not in SD04_CLASS_TO_IDX:
            raise ValueError(
                f"Unsupported SD04 class '{class_token}' in '{txt_path}'. "
                f"Expected one of {list(SD04_CLASS_TO_IDX.keys())}."
            )

        img = Image.open(png_path).convert("L")
        images.append(torch.from_numpy(np.array(img, dtype=np.uint8)))
        labels.append(SD04_CLASS_TO_IDX[class_token])

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def _extract_sd04_class(txt_path):
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Class:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"Could not find 'Class:' field in '{txt_path}'.")
