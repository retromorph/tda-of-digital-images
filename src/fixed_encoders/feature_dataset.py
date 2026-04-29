import inspect
import json
import os
import time
import uuid
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EncodedFeatureDataset(Dataset):
    """
    Wraps a dataset of persistence diagrams (already computed PHT) and applies a fixed encoder.

    Optionally caches encoder outputs to disk so that Gudhi transforms are not recomputed
    on every epoch / run.
    """

    def __init__(
        self,
        base_dataset,
        encoder,
        *,
        split: str,
        cache_dir: str | Path | None = None,
        base_cache_config: dict | None = None,
        enable_disk_cache: bool = True,
        disk_cache_lock_timeout_s: float = 60 * 10,
    ):
        self.base_dataset = base_dataset
        self.encoder = encoder
        self.split = split
        self.enable_disk_cache = enable_disk_cache
        self.disk_cache_lock_timeout_s = disk_cache_lock_timeout_s

        if cache_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            cache_dir = repo_root / "data" / "encoded_features"
        self.cache_dir = Path(cache_dir)
        self.base_cache_config = base_cache_config or {}

        # In no-cache mode we keep backward compatibility with previous behavior.
        if not self.enable_disk_cache:
            self._features = None
            self._mask = None
            return

        self._encoder_name = type(self.encoder).__name__
        self._encoder_version_hash = self._compute_encoder_version_hash()
        self._cache_key = self._compute_cache_key()

        self._encoder_cache_dir = self.cache_dir / self._encoder_name
        self._encoder_cache_dir.mkdir(parents=True, exist_ok=True)

        self._features_path = self._encoder_cache_dir / f"{self._cache_key}__{self.split}__features.npy"
        self._mask_path = self._encoder_cache_dir / f"{self._cache_key}__{self.split}__mask.npy"
        self._lock_path = self._encoder_cache_dir / f"{self._cache_key}__{self.split}__lock"

        self._features = None
        self._mask = None

        # Lazy behavior: we initialize the cache storage on first use, but do not force
        # full precomputation for the whole split (important for smoke tests using Subset()).
        if self._features_path.exists() and self._mask_path.exists():
            print(f"Loading cached encoded features from {self._features_path}")
            self._open_cache_for_readwrite()
        else:
            print(f"Creating encoded features cache at {self._encoder_cache_dir} (split={self.split})")
            self._initialize_cache_storage()

    def __len__(self):
        return len(self.base_dataset)

    def __getstate__(self):
        # Avoid pickling mmap-backed arrays across DataLoader workers.
        state = dict(self.__dict__)
        state["_features"] = None
        state["_mask"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if getattr(self, "enable_disk_cache", False):
            self._open_cache_for_readwrite()

    def _compute_encoder_version_hash(self) -> str:
        src_file = None
        try:
            src_file = inspect.getsourcefile(type(self.encoder))
        except Exception:
            src_file = None

        if src_file and os.path.isfile(src_file):
            with open(src_file, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        # Fallback: if we can't resolve source file (interactive / zipped), still make key stable.
        fallback = json.dumps(
            {"module": type(self.encoder).__module__, "qualname": type(self.encoder).__qualname__},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(fallback).hexdigest()

    def _compute_cache_key(self) -> str:
        if hasattr(self.encoder, "cache_config"):
            encoder_cfg = self.encoder.cache_config()
        else:
            encoder_cfg = {"class": type(self.encoder).__name__}

        key_obj = {
            "encoder_class": type(self.encoder).__qualname__,
            "encoder_module": type(self.encoder).__module__,
            "encoder_version_hash": self._encoder_version_hash,
            "encoder_cfg": encoder_cfg,
            "split": self.split,
            "base_cache_config": self.base_cache_config,
        }
        key_json = json.dumps(key_obj, sort_keys=True, default=str)
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def _acquire_lock(self):
        """
        Best-effort file lock via exclusive creation.
        """
        start = time.time()
        while True:
            try:
                fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}".encode("utf-8"))
                os.close(fd)
                return
            except FileExistsError:
                if time.time() - start > self.disk_cache_lock_timeout_s:
                    raise TimeoutError(f"Timeout waiting for cache lock: {self._lock_path}")
                time.sleep(1.0)

    def _release_lock(self):
        try:
            os.unlink(self._lock_path)
        except FileNotFoundError:
            pass

    def _initialize_cache_storage(self):
        # Initialization must know the feature shape. We compute it from the first item only.
        if len(self.base_dataset) == 0:
            raise ValueError("base_dataset must be non-empty to initialize encoded cache.")

        with self._lock_context():
            # Another process might have created the cache while we were waiting for the lock.
            if self._features_path.exists() and self._mask_path.exists():
                self._open_cache_for_readwrite()
                return

            # Compute a single example to infer output shape.
            diagram0, _ = self.base_dataset[0]
            feat0 = self.encoder(diagram0)
            feat0_np = feat0.detach().cpu().numpy().astype(np.float32, copy=False)

            n = len(self.base_dataset)
            feature_shape = tuple(feat0_np.shape)  # e.g. (1, R, R) or (L, R)
            features_shape = (n, *feature_shape)

            tmp_features_path = self._encoder_cache_dir / f"{self._features_path.name}.tmp.{uuid.uuid4().hex}"
            tmp_mask_path = self._encoder_cache_dir / f"{self._mask_path.name}.tmp.{uuid.uuid4().hex}"

            features_mm = np.lib.format.open_memmap(
                str(tmp_features_path),
                mode="w+",
                dtype=np.float32,
                shape=features_shape,
            )
            mask_mm = np.lib.format.open_memmap(
                str(tmp_mask_path),
                mode="w+",
                dtype=np.uint8,
                shape=(n,),
            )

            # Initialize mask to 0 (not computed).
            mask_mm[:] = 0
            features_mm[0] = feat0_np
            mask_mm[0] = 1

            features_mm.flush()
            mask_mm.flush()
            del features_mm
            del mask_mm

            # Atomically publish.
            os.replace(tmp_features_path, self._features_path)
            os.replace(tmp_mask_path, self._mask_path)

        # Re-open in read-write mode for subsequent partial fills.
        self._open_cache_for_readwrite()

    def _open_cache_for_readwrite(self):
        features_ro = np.load(str(self._features_path), mmap_mode="r")
        mask_ro = np.load(str(self._mask_path), mmap_mode="r")

        features_shape = features_ro.shape
        features_dtype = features_ro.dtype
        mask_shape = mask_ro.shape
        mask_dtype = mask_ro.dtype
        del features_ro
        del mask_ro

        self._features = np.lib.format.open_memmap(
            str(self._features_path),
            mode="r+",
            dtype=features_dtype,
            shape=features_shape,
        )
        self._mask = np.lib.format.open_memmap(
            str(self._mask_path),
            mode="r+",
            dtype=mask_dtype,
            shape=mask_shape,
        )

    class _LockContext:
        def __init__(self, outer):
            self.outer = outer

        def __enter__(self):
            self.outer._acquire_lock()

        def __exit__(self, exc_type, exc, tb):
            self.outer._release_lock()
            return False

    def _lock_context(self):
        return EncodedFeatureDataset._LockContext(self)

    def __getitem__(self, idx):
        diagram, target = self.base_dataset[idx]

        if not self.enable_disk_cache:
            feature = self.encoder(diagram)
            return feature, target

        # If cached, just load from memmap.
        if int(self._mask[idx]) == 1:
            feature_np = self._features[idx]
            feature = torch.from_numpy(feature_np).float()
            return feature, target

        # Compute missing entry once and persist.
        with self._lock_context():
            if int(self._mask[idx]) != 1:
                feat = self.encoder(diagram)
                feat_np = feat.detach().cpu().numpy().astype(np.float32, copy=False)
                self._features[idx] = feat_np
                self._mask[idx] = 1
                self._features.flush()
                self._mask.flush()

        feature_np = self._features[idx]
        feature = torch.from_numpy(feature_np).float()
        return feature, target
