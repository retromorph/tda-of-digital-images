import numpy as np
import subprocess
from itertools import product

seeds = [0]
aggs = ["mean", "max"]
filtration_idx = [
    range(0, 8, 1),
    range(0, 16, 2),
]

for seed, idx, agg in product(seeds, filtration_idx, aggs):
    subprocess.run(
        ["python3",
        "run.py",
        "--d_model", str(96),
        "--d_hidden", str(96),
        "--agg", agg,
        "--idx"] + [str(x) for x in idx] + 
        ["--seed", str(seed),
        "--epochs", str(200)
    ])