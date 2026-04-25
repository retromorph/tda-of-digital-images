"""Sweep image-classifier runs over augmentation strengths (see config/invariance.yaml)."""

import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "exp" / "config" / "invariance.yaml", "r") as f:
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
                str(ROOT / "exp" / f"run_{model.lower()}.py"),
                "--experiment",
                cfg["experiment"],
                "--dataset",
                dataset,
                "--seed",
                str(seed),
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
