import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "exp" / "config" / "directions.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

for seed, dataset, idx in product(cfg["seeds"], cfg["datasets"], cfg["filters"]):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "exp" / "runners" / "run_persformer.py"),
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
            "--device",
            str(0),
            "--idx",
        ]
        + [str(item) for item in idx],
        cwd=str(ROOT),
        check=False,
    )
