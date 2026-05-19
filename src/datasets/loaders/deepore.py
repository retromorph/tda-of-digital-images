import os
from pathlib import Path

import numpy as np
import torch

from scipy.ndimage import zoom

from src.datasets.types import ImageDataset

DEEPORE_ROOT = "./data/DeePore"
DEEPORE_H5_FILENAME = "DeePore_Dataset.h5"
DEEPORE_CACHE_SUBDIR = "cache"
DEEPORE_DOWNLOAD_URL = "https://zenodo.org/records/3820900"


def _read_h5_dataset(h5_path, target_size, max_samples=None):
    import h5py

    target_size = tuple(target_size)
    with h5py.File(h5_path, "r") as f:
        # DeePore stores volumes under 'X' (binary, uint8) and labels under 'Y' (15+ scalar features).
        # Resolve keys defensively in case the upstream dataset re-keys them in a future release.
        x_key = "X" if "X" in f else next(k for k in f.keys() if k.upper() == "X")
        y_key = "Y" if "Y" in f else next(k for k in f.keys() if k.upper() == "Y")
        x_ds = f[x_key]
        y_ds = f[y_key]

        n_total = x_ds.shape[0]
        if max_samples is not None:
            n_total = min(int(max_samples), n_total)

        # Decide source spatial shape, ignoring a potential trailing channel dim of size 1.
        spatial_shape = tuple(x_ds.shape[1:])
        if len(spatial_shape) == 4 and spatial_shape[-1] == 1:
            spatial_shape = spatial_shape[:-1]
        if len(spatial_shape) != 3:
            raise ValueError(f"Expected (N, D, H, W[, 1]) volumes in DeePore, got shape {x_ds.shape}.")

        source_size = spatial_shape
        zoom_factors = tuple(t / s for t, s in zip(target_size, source_size))

        volumes = np.empty((n_total, *target_size), dtype=np.uint8)
        targets_raw = np.asarray(y_ds[:n_total], dtype=np.float32)

        for i in range(n_total):
            vol = np.asarray(x_ds[i])
            if vol.ndim == 4 and vol.shape[-1] == 1:
                vol = vol[..., 0]
            vol = vol.astype(np.float32)
            if zoom_factors != (1.0, 1.0, 1.0):
                # mean-pool downsampling on the continuous mask, then re-threshold to binary
                vol = zoom(vol, zoom_factors, order=1)
                vol = (vol > 0.5).astype(np.uint8)
            else:
                vol = (vol > 0.5).astype(np.uint8)
            volumes[i] = vol

    return volumes, targets_raw


def _cache_path(dataset_root, target_size, max_samples):
    parts = "x".join(str(s) for s in target_size)
    tag = f"{parts}" + (f"_n{max_samples}" if max_samples is not None else "")
    return Path(dataset_root) / DEEPORE_CACHE_SUBDIR / f"deepore_{tag}.pt"


def _load_or_build(dataset_root, h5_filename, target_size, max_samples):
    root = Path(dataset_root)
    cache_file = _cache_path(dataset_root, target_size, max_samples)
    if cache_file.exists():
        payload = torch.load(cache_file, map_location="cpu", weights_only=True)
        return payload["volumes"], payload["targets"]

    h5_path = root / h5_filename
    if not h5_path.exists():
        raise FileNotFoundError(
            f"DeePore HDF5 file not found at '{h5_path}'. Download it from "
            f"{DEEPORE_DOWNLOAD_URL} (file: {h5_filename}) and place it under '{root}'."
        )

    volumes_np, targets_np = _read_h5_dataset(h5_path, target_size, max_samples=max_samples)
    volumes = torch.from_numpy(volumes_np)
    targets = torch.from_numpy(targets_np)

    os.makedirs(cache_file.parent, exist_ok=True)
    torch.save({"volumes": volumes, "targets": targets}, cache_file)
    return volumes, targets


def get_deepore_3d_datasets(
    *,
    dataset_root=DEEPORE_ROOT,
    h5_filename=DEEPORE_H5_FILENAME,
    seed=0,
    test_fraction=1 / 6,
    target_size=(64, 64, 64),
    target_index=0,
    log_target=True,
    max_samples=None,
):
    volumes, targets_raw = _load_or_build(dataset_root, h5_filename, target_size, max_samples)

    if targets_raw.ndim == 1:
        y = targets_raw
    else:
        if target_index >= targets_raw.shape[1]:
            raise ValueError(
                f"target_index={target_index} out of range for DeePore target vector of width {targets_raw.shape[1]}."
            )
        y = targets_raw[:, target_index]

    y = y.to(torch.float32)
    if log_target:
        # Permeability spans many decades; guard against non-positive entries before log10.
        y = torch.where(y > 0, y, torch.full_like(y, float("nan")))
        y = torch.log10(y)
        finite = torch.isfinite(y)
        if not bool(finite.all()):
            mask = finite
            volumes = volumes[mask]
            y = y[mask]

    n_total = y.shape[0]
    n_test = int(n_total * test_fraction)
    if n_test <= 0 or n_test >= n_total:
        raise ValueError("Invalid test_fraction for DeePore dataset split.")

    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    test_idx = perm[:n_test]
    train_val_idx = perm[n_test:]

    return (
        ImageDataset(volumes[train_val_idx], y[train_val_idx]),
        ImageDataset(volumes[test_idx], y[test_idx]),
    )
