"""Legacy `models x datasets x seeds` sweep, ported on top of the unified orchestrator.

The original script forked one subprocess per `run_<model>.py` with positional
flags. Now it builds an orchestrator-shaped config in memory and calls
`exp.pipelines.orchestrator.run` once.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.orchestrator import run  # noqa: E402


def _resolve_cfg_path() -> Path:
    new_path = ROOT / "exp" / "config" / "legacy" / "main.yaml"
    if new_path.exists():
        return new_path
    return ROOT / "exp" / "config" / "main.yaml"


def _to_orchestrator_cfg(legacy_cfg: dict) -> dict:
    methods = {}
    for model in legacy_cfg["models"]:
        methods[model.lower()] = {
            "args": {
                "model": model,
                "lr": legacy_cfg.get("lr", 1e-3),
                "batch_size": legacy_cfg.get("batch_size", 128),
                "epochs": legacy_cfg.get("epochs", 10),
            },
        }
    tasks = [{"name": ds.lower(), "dataset": ds, "transform": None, "power": 0.0} for ds in legacy_cfg["datasets"]]
    return {
        "experiment": legacy_cfg.get("experiment", "Main"),
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
    raise SystemExit(run(cfg_dict=_to_orchestrator_cfg(legacy_cfg)))


if __name__ == "__main__":
    main()
