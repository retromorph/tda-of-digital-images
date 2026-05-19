"""num_latents ablation for LatentPersformer (M8, Series 2.1).

Sweeps ``num_latents ∈ {4, 8, 16, 32, 64, 128, 256, 512}`` on D1+D3 with the
default LatentPersformer config from M7. Used to identify the knee point
(diminishing returns vs cost) and to defend the M7 default choice.
"""

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.orchestrator import run  # noqa: E402


def _build_cfg(base: dict, num_latents: int) -> dict:
    methods = {}
    for name, m in base["methods"].items():
        args = dict(m.get("args", {}))
        args["num_latents"] = num_latents
        methods[name] = {"args": args}
    tasks = [
        {"name": f"{ds.lower()}__nl{num_latents}", "dataset": ds, "transform": None, "power": 0.0}
        for ds in base["datasets"]
    ]
    return {
        "experiment": f"{base['experiment']}/nl{num_latents}",
        "device": base.get("device", "cpu"),
        "num_workers": base.get("num_workers", 0),
        "seeds": base["seeds"],
        "tasks": tasks,
        "common_persistence": base["common_persistence"],
        "logging": base.get("logging", {}),
        "budget": base["budget"],
        "methods": methods,
    }


def main() -> int:
    cfg_path = ROOT / "exp" / "config" / "ablations" / "latent_size.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rc = 0
    for nl in cfg["num_latents"]:
        sub = _build_cfg(copy.deepcopy(cfg), int(nl))
        rc |= run(cfg_dict=sub)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
