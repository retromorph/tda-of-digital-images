import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "exp" / "config" / "n_filters.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

for seed, dataset, filter_idx in product(cfg["seeds"], cfg["datasets"], cfg["filters"]):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "exp" / "run_persformer.py"),
            "--experiment",
            "N_filters",
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
            "--device",
            str(0),
            "--idx",
        ]
        + [str(x) for x in filter_idx],
        cwd=str(ROOT),
        check=False,
    )
