import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "exp" / "config" / "directions.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

for seed, dataset, idx in product(cfg["seeds"], cfg["datasets"], cfg["filters"]):
    flag = 0 if len(idx) == 0 else 1

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "exp" / "run_phtx.py"),
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
            "--flag",
            str(flag),
            "--device",
            str(0),
            "--idx",
        ]
        + [str(item) for item in idx],
        cwd=str(ROOT),
        check=False,
    )
