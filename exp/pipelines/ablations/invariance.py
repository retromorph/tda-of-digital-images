"""Sweep image-classifier runs over augmentation strengths (see config/invariance.yaml)."""

import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]

def _resolve_cfg_path():
    new_path = ROOT / "exp" / "config" / "ablations" / "invariance.yaml"
    if new_path.exists():
        return new_path
    return ROOT / "exp" / "config" / "invariance.yaml"

with open(_resolve_cfg_path(), "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

models = cfg.get("models", ["MLP"])
transforms_cfg = cfg["transformations"]

for seed, dataset, model, tspec in product(
    cfg["seeds"], cfg["datasets"], models, transforms_cfg
):
    name = tspec["name"]
    for power in tspec["std"]:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "exp" / "runners" / f"run_{model.lower()}.py"),
                "--experiment",
                cfg["experiment"],
                "--dataset",
                dataset,
                "--seed",
                str(seed),
                "--device",
                str(cfg["device"]),
                "--batch_size",
                str(cfg["batch_size"]),
                "--lr",
                str(cfg["lr"]),
                "--epochs",
                str(cfg["epochs"]),
                "--transform",
                name,
                "--power",
                str(power),
            ],
            cwd=str(ROOT),
            check=False,
        )
