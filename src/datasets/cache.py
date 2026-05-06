import hashlib
import json
import os
import pickle
from datetime import datetime, timezone


CACHE_VERSION = 2


def _stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(filtration_name: str, filtration_params: dict | None) -> str:
    payload = {
        "filtration_name": filtration_name,
        "filtration_params": filtration_params or {},
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]


def cache_dir(dataset: str, cache_key: str) -> str:
    return os.path.join("data", "cache", "diagrams", dataset, cache_key)


def _transform_suffix(transform_str: str | None, power) -> str:
    if transform_str is None:
        return ""
    return f"_t-{transform_str}-{power}"


def split_cache_path(
    dataset: str,
    cache_key: str,
    split: str,
    seed: int,
    *,
    transform_str: str | None = None,
    power=0.0,
) -> str:
    suffix = _transform_suffix(transform_str, power) if split == "test" else ""
    return os.path.join(cache_dir(dataset, cache_key), f"{split}{suffix}_seed-{seed}.pkl")


def meta_path(dataset: str, cache_key: str) -> str:
    return os.path.join(cache_dir(dataset, cache_key), "meta.json")


def load_cache(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_cache(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def write_meta(
    dataset: str,
    cache_key: str,
    filtration_name: str,
    filtration_params: dict | None,
    schema: dict,
) -> str:
    path = meta_path(dataset, cache_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "dataset": dataset,
        "filtration_name": filtration_name,
        "filtration_params": filtration_params or {},
        "schema": schema,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": CACHE_VERSION,
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, ensure_ascii=True, indent=2)
    os.replace(tmp, path)
    return path
