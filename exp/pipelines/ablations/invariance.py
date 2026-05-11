"""Sweep image-classifier runs over augmentation strengths via the unified orchestrator.

Builds one orchestrator task per (transform name, power) and runs the configured
models against it.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.orchestrator import run  # noqa: E402


def _resolve_cfg_path() -> Path:
    new_path = ROOT / "exp" / "config" / "ablations" / "invariance.yaml"
    if new_path.exists():
        return new_path
    return ROOT / "exp" / "config" / "invariance.yaml"


def _model_method_cfg(model: str, legacy_cfg: dict) -> dict:
    return {
        "args": {
            "model": model,
            "lr": legacy_cfg.get("lr", 1e-3),
            "batch_size": legacy_cfg.get("batch_size", 128),
            "epochs": legacy_cfg.get("epochs", 10),
        },
    }


def _build_orchestrator_cfg(legacy_cfg: dict, transform_name: str, power: float) -> dict:
    tasks = []
    for ds in legacy_cfg["datasets"]:
        tasks.append(
            {
                "name": f"{ds.lower()}_{transform_name}_p{power}",
                "dataset": ds,
                "transform": transform_name,
                "power": power,
            }
        )
    methods = {model.lower(): _model_method_cfg(model, legacy_cfg) for model in legacy_cfg.get("models", ["MLP"])}
    return {
        "experiment": legacy_cfg.get("experiment", "Invariance"),
        "device": legacy_cfg.get("device", "cpu"),
        "num_workers": legacy_cfg.get("num_workers", 0),
        "seeds": legacy_cfg["seeds"],
        "tasks": tasks,
        "methods": methods,
        "__source": str(_resolve_cfg_path()),
    }


def main():
    with open(_resolve_cfg_path(), "r", encoding="utf-8") as f:
        legacy_cfg = yaml.load(f, Loader=yaml.FullLoader)
    rc = 0
    for tspec in legacy_cfg["transformations"]:
        for power in tspec["std"]:
            rc |= run(cfg_dict=_build_orchestrator_cfg(legacy_cfg, tspec["name"], power))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
