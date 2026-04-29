import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "exp" / "config" / "main.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

for seed, dataset, model in product(cfg["seeds"], cfg["datasets"], cfg["models"]):
    cmd = [
        sys.executable,
        str(ROOT / "exp" / "runners" / "run_{}.py".format(model.lower())),
        "--experiment",
        cfg["experiment"],
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--batch_size",
        str(cfg["batch_size"]),
    ]
    if "lr" in cfg:
        cmd.extend(["--lr", str(cfg["lr"])])
    if "epochs" in cfg:
        cmd.extend(["--epochs", str(cfg["epochs"])])
    subprocess.run(cmd, cwd=str(ROOT), check=False)
