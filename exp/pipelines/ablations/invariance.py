"""Sweep image classifiers over augmentation types and strengths."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.orchestrator import run


def _build_cfg(base: dict, transform: str, power: float) -> dict:
    tasks = [
        {"name": f"{ds.lower()}_{transform}_p{power}", "dataset": ds, "transform": transform, "power": power}
        for ds in base["datasets"]
    ]
    return {
        "experiment": base["experiment"],
        "device": base["device"],
        "num_workers": base.get("num_workers", 0),
        "seeds": base["seeds"],
        "tasks": tasks,
        "methods": base["methods"],
    }


def main():
    cfg_path = ROOT / "exp" / "config" / "ablations" / "invariance.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rc = 0
    for tspec in cfg["transformations"]:
        for power in tspec["powers"]:
            rc |= run(cfg_dict=_build_cfg(cfg, tspec["name"], power))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
