"""Persformer sweep over different filtration direction subsets."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.orchestrator import run


def _build_cfg(base: dict, filter_idx: list[int]) -> dict:
    tasks = [{"name": ds.lower(), "dataset": ds, "transform": None, "power": 0.0} for ds in base["datasets"]]
    return {
        "experiment": base["experiment"] + "_" + "-".join(str(x) for x in filter_idx),
        "device": base["device"],
        "num_workers": base.get("num_workers", 0),
        "seeds": base["seeds"],
        "tasks": tasks,
        "common_persistence": {"idx": filter_idx, "eps": base["common_persistence"]["eps"]},
        "methods": base["methods"],
    }


def main():
    cfg_path = ROOT / "exp" / "config" / "ablations" / "n_filters.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rc = 0
    for filter_idx in cfg["filters"]:
        rc |= run(cfg_dict=_build_cfg(cfg, filter_idx))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
