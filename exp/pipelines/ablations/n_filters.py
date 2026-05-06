"""Persformer sweep over different `idx` (direction) subsets, on top of the orchestrator."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.orchestrator import run  # noqa: E402


def _resolve_cfg_path() -> Path:
    new_path = ROOT / "exp" / "config" / "ablations" / "n_filters.yaml"
    if new_path.exists():
        return new_path
    return ROOT / "exp" / "config" / "n_filters.yaml"


def _persformer_method_cfg(legacy_cfg: dict) -> dict:
    return {
        "args": {
            "model": "PERSFORMER",
            "lr": legacy_cfg.get("lr", 1e-3),
            "batch_size": legacy_cfg.get("batch_size", 128),
            "epochs": legacy_cfg.get("epochs", 10),
            "warmup_epochs": legacy_cfg.get("warmup_epochs", 0),
            "weight_decay": legacy_cfg.get("weight_decay", 0.0),
            "eta_min": legacy_cfg.get("eta_min", 0.0),
        },
    }


def _build_orchestrator_cfg(legacy_cfg: dict, filter_idx: list[int]) -> dict:
    tasks = [{"name": ds.lower(), "dataset": ds, "transform": None, "power": 0.0} for ds in legacy_cfg["datasets"]]
    return {
        "experiment": legacy_cfg.get("experiment", "N_filters") + "_" + "-".join(str(x) for x in filter_idx),
        "device": legacy_cfg.get("device", "cpu"),
        "num_workers": legacy_cfg.get("num_workers", 0),
        "seeds": legacy_cfg["seeds"],
        "tasks": tasks,
        "common_persistence": {"idx": filter_idx, "eps": legacy_cfg.get("eps", 0.02)},
        "methods": {"persformer": _persformer_method_cfg(legacy_cfg)},
        "__source": str(_resolve_cfg_path()),
    }


def main():
    with open(_resolve_cfg_path(), "r", encoding="utf-8") as f:
        legacy_cfg = yaml.load(f, Loader=yaml.FullLoader)
    rc = 0
    for filter_idx in legacy_cfg["filters"]:
        rc |= run(cfg_dict=_build_orchestrator_cfg(legacy_cfg, filter_idx))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
