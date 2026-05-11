"""Per-direction Persformer sweep over all individual and paired filtration directions."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.ablations.n_filters import _build_cfg
from exp.pipelines.orchestrator import run


def main():
    cfg_path = ROOT / "exp" / "config" / "ablations" / "directions.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rc = 0
    for filter_idx in cfg["filters"]:
        rc |= run(cfg_dict=_build_cfg(cfg, filter_idx))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
