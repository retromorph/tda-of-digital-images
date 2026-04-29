import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "exp" / "config" / "fixed_encoders_smoke.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


def run_matrix(section_name, params):
    runner = params["runner"]
    keys = [k for k in params.keys() if k != "runner"]
    values = [params[k] for k in keys]

    for dataset, seed, combo in product(cfg["datasets"], cfg["seeds"], product(*values)):
        cmd = [
            sys.executable,
            str(ROOT / "exp" / "runners" / runner),
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
            "--num_workers",
            str(cfg.get("num_workers", 0)),
            "--max_train",
            str(cfg["max_train"]),
            "--max_val",
            str(cfg["max_val"]),
            "--max_test",
            str(cfg["max_test"]),
        ]

        for k, v in zip(keys, combo):
            cmd.extend([f"--{k}", str(v)])

        print("Running", section_name, "|", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=False)


run_matrix("persistence_image", cfg["persistence_image"])
run_matrix("persistence_landscape", cfg["persistence_landscape"])
run_matrix("persistence_silhouette", cfg["persistence_silhouette"])
