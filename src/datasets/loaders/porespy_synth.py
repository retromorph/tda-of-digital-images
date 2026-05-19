"""Synthetic 3D porous-media dataset (PORESPY-3D / D3).

Generates ``N`` binary blob volumes via ``porespy.generators.blobs`` with
controlled porosity and blobiness. The regression target is a Kozeny-Carman
proxy permeability (cheap to compute; documented as a proxy in paper Methods).

Each sample also has 10 physical/topological features computed once at build
time and stored in ``features.parquet`` for use as linear-probe targets (M10).

Cache layout under ``data/PoreSpySynth/``:
    cache/synth_seed{S}_n{N}_shape{D}x{H}x{W}.pt   — volumes + targets tensors
    features/features_seed{S}_n{N}_shape{D}x{H}x{W}.parquet — per-sample features
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.datasets.types import ImageDataset

PORESPY_SYNTH_ROOT = "./data/PoreSpySynth"
DEFAULT_SHAPE = (64, 64, 64)
DEFAULT_N_SAMPLES = 1000
DEFAULT_MASTER_SEED = 20260519

FEATURE_COLUMNS = (
    "porosity",
    "specific_surface_area",
    "mean_pore_radius",
    "max_pore_radius",
    "beta_0",
    "beta_1",
    "euler_characteristic",
    "persistence_entropy_h0",
    "persistence_entropy_h1",
    "total_persistence_h1",
)


def _shape_tag(shape: tuple[int, ...]) -> str:
    return "x".join(str(s) for s in shape)


def _cache_path(root: Path, master_seed: int, n_samples: int, shape: tuple[int, ...]) -> Path:
    return root / "cache" / f"synth_seed{master_seed}_n{n_samples}_shape{_shape_tag(shape)}.pt"


def _features_path(root: Path, master_seed: int, n_samples: int, shape: tuple[int, ...]) -> Path:
    return root / "features" / f"features_seed{master_seed}_n{n_samples}_shape{_shape_tag(shape)}.parquet"


def _persistence_entropy(lifetimes: np.ndarray) -> float:
    """Shannon entropy over normalized persistence intervals."""
    if lifetimes.size == 0:
        return 0.0
    total = float(lifetimes.sum())
    if total <= 0:
        return 0.0
    p = lifetimes / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _compute_features(volume_uint8: np.ndarray) -> dict:
    """Compute the 10 physical/topological features for a single volume.

    Args:
        volume_uint8: ``(D, H, W)`` uint8 with 1=pore, 0=solid.
    """
    import gudhi
    from scipy import ndimage as ndi
    from scipy.ndimage import distance_transform_edt
    from skimage import measure
    import porespy as ps

    pore = volume_uint8.astype(bool)
    n_voxels = int(pore.size)

    porosity = float(pore.mean())

    # Specific surface area per unit volume via marching cubes mesh.
    try:
        verts, faces, _, _ = measure.marching_cubes(pore.astype(float), level=0.5)
        ssa = float(measure.mesh_surface_area(verts, faces)) / n_voxels
    except (ValueError, RuntimeError):
        ssa = 0.0

    # Local thickness as a pore-radius distribution (voxel units).
    try:
        thickness = ps.filters.local_thickness(pore, sizes=10)
        radii = thickness[pore]
        mean_r = float(radii.mean()) if radii.size > 0 else 0.0
        max_r = float(radii.max()) if radii.size > 0 else 0.0
    except Exception:
        mean_r = max_r = 0.0

    # Connected components of pore phase.
    _, b0 = ndi.label(pore)

    # 3D Euler number (skimage convention: connectivity=3 means 26-neighbourhood).
    try:
        chi = int(measure.euler_number(pore, connectivity=3))
    except Exception:
        chi = 0

    # β2: count interior solid cavities (solid components that don't touch the boundary).
    solid_labels, n_solid = ndi.label(~pore)
    if n_solid > 0:
        boundary_labels = set()
        for face in (
            solid_labels[0], solid_labels[-1],
            solid_labels[:, 0], solid_labels[:, -1],
            solid_labels[:, :, 0], solid_labels[:, :, -1],
        ):
            boundary_labels.update(np.unique(face).tolist())
        boundary_labels.discard(0)
        b2 = max(0, n_solid - len(boundary_labels))
    else:
        b2 = 0
    # χ = β0 - β1 + β2  →  β1 = β0 - χ + β2
    b1 = int(b0 - chi + b2)

    # Cubical persistence on EDT of the pore phase.
    edt = distance_transform_edt(pore)
    cc = gudhi.CubicalComplex(dimensions=edt.shape, top_dimensional_cells=edt.flatten())
    cc.compute_persistence()
    pairs = cc.persistence()
    h0_life: list[float] = []
    h1_life: list[float] = []
    for dim, (b, d) in pairs:
        if not np.isfinite(d):
            continue
        life = d - b
        if life <= 0:
            continue
        if dim == 0:
            h0_life.append(life)
        elif dim == 1:
            h1_life.append(life)
    h0_arr = np.asarray(h0_life, dtype=np.float64)
    h1_arr = np.asarray(h1_life, dtype=np.float64)

    return dict(
        porosity=porosity,
        specific_surface_area=ssa,
        mean_pore_radius=mean_r,
        max_pore_radius=max_r,
        beta_0=int(b0),
        beta_1=b1,
        euler_characteristic=chi,
        persistence_entropy_h0=_persistence_entropy(h0_arr),
        persistence_entropy_h1=_persistence_entropy(h1_arr),
        total_persistence_h1=float(h1_arr.sum()),
    )


def _kozeny_carman_log_perm(porosity: float, ssa: float) -> float:
    """Kozeny-Carman proxy log10(permeability), in arbitrary units.

    k = φ^3 / (180 · (1-φ)^2 · S^2), with floors to avoid singularities.
    Returns log10(k); values fall in roughly [-6, 0] for blobs at φ∈[0.15, 0.55].
    """
    phi = max(min(porosity, 0.99), 0.01)
    s = max(ssa, 1e-6)
    k = (phi ** 3) / (180.0 * (1.0 - phi) ** 2 * s ** 2)
    return float(np.log10(max(k, 1e-30)))


def _generate_one(args):
    """Generate a single sample. Returns (volume_uint8, feature_dict, log_k)."""
    import porespy as ps

    i, master_seed, shape, porosity, blobiness = args
    rng = np.random.default_rng((master_seed, i))
    sample_seed = int(rng.integers(0, 2**31 - 1))
    vol_bool = ps.generators.blobs(
        shape=list(shape),
        porosity=float(porosity),
        blobiness=int(blobiness),
        seed=sample_seed,
    )
    vol_uint8 = vol_bool.astype(np.uint8)
    feats = _compute_features(vol_uint8)
    log_k = _kozeny_carman_log_perm(feats["porosity"], feats["specific_surface_area"])
    return vol_uint8, feats, log_k


def _build_dataset(
    master_seed: int,
    n_samples: int,
    shape: tuple[int, ...],
    n_jobs: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Generate all volumes, features, and permeability targets."""
    from joblib import Parallel, delayed

    rng = np.random.default_rng(master_seed)
    porosities = rng.uniform(0.15, 0.55, size=n_samples)
    blobinesses = rng.choice([1, 2, 3], size=n_samples)

    args_list = [
        (i, master_seed, shape, porosities[i], blobinesses[i])
        for i in range(n_samples)
    ]

    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_generate_one)(a) for a in args_list)

    volumes = np.empty((n_samples, *shape), dtype=np.uint8)
    targets = np.empty(n_samples, dtype=np.float32)
    feat_rows = []
    for i, (vol, feats, log_k) in enumerate(results):
        volumes[i] = vol
        targets[i] = log_k
        feat_rows.append({"index": i, "blobiness": int(blobinesses[i]), **feats})

    features_df = pd.DataFrame(feat_rows, columns=["index", "blobiness", *FEATURE_COLUMNS])
    return (
        torch.from_numpy(volumes),
        torch.from_numpy(targets),
        features_df,
    )


def _load_or_build(
    dataset_root: str,
    master_seed: int,
    n_samples: int,
    shape: tuple[int, ...],
    n_jobs: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    root = Path(dataset_root)
    cache_file = _cache_path(root, master_seed, n_samples, shape)
    feats_file = _features_path(root, master_seed, n_samples, shape)
    if cache_file.exists() and feats_file.exists():
        payload = torch.load(cache_file, map_location="cpu", weights_only=True)
        features_df = pd.read_parquet(feats_file)
        return payload["volumes"], payload["targets"], features_df

    volumes, targets, features_df = _build_dataset(master_seed, n_samples, shape, n_jobs=n_jobs)

    os.makedirs(cache_file.parent, exist_ok=True)
    os.makedirs(feats_file.parent, exist_ok=True)
    torch.save({"volumes": volumes, "targets": targets}, cache_file)
    features_df.to_parquet(feats_file, index=False)
    return volumes, targets, features_df


def get_porespy_synth_datasets(
    *,
    dataset_root: str = PORESPY_SYNTH_ROOT,
    seed: int = 0,
    test_fraction: float = 1 / 6,
    target_size: tuple[int, ...] = DEFAULT_SHAPE,
    n_samples: int = DEFAULT_N_SAMPLES,
    master_seed: int = DEFAULT_MASTER_SEED,
    n_jobs: int = -1,
) -> tuple[ImageDataset, ImageDataset]:
    """Return train/test ``ImageDataset`` pair for PORESPY-3D.

    ``seed`` controls the train/test split (not generation — generation is
    deterministic per ``master_seed``).
    """
    volumes, targets, _features_df = _load_or_build(
        dataset_root=dataset_root,
        master_seed=master_seed,
        n_samples=n_samples,
        shape=tuple(target_size),
        n_jobs=n_jobs,
    )

    n_total = volumes.shape[0]
    n_test = int(n_total * test_fraction)
    if n_test <= 0 or n_test >= n_total:
        raise ValueError("Invalid test_fraction for PORESPY-3D dataset split.")

    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return (
        ImageDataset(volumes[train_idx], targets[train_idx]),
        ImageDataset(volumes[test_idx], targets[test_idx]),
    )


def load_features_parquet(
    dataset_root: str = PORESPY_SYNTH_ROOT,
    master_seed: int = DEFAULT_MASTER_SEED,
    n_samples: int = DEFAULT_N_SAMPLES,
    shape: tuple[int, ...] = DEFAULT_SHAPE,
) -> pd.DataFrame:
    """Helper for analysis scripts to retrieve per-sample physical features."""
    path = _features_path(Path(dataset_root), master_seed, n_samples, tuple(shape))
    return pd.read_parquet(path)
