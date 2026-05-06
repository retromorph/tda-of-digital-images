"""Per-direction Persformer sweep, on top of the orchestrator."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.ablations.n_filters import _build_orchestrator_cfg  # noqa: E402
from exp.pipelines.orchestrator import run  # noqa: E402


def _resolve_cfg_path() -> Path:
    new_path = ROOT / "exp" / "config" / "ablations" / "directions.yaml"
    if new_path.exists():
        return new_path
    return ROOT / "exp" / "config" / "directions.yaml"


def main():
    with open(_resolve_cfg_path(), "r", encoding="utf-8") as f:
        legacy_cfg = yaml.load(f, Loader=yaml.FullLoader)
    rc = 0
    for filter_idx in legacy_cfg["filters"]:
        rc |= run(cfg_dict=_build_orchestrator_cfg(legacy_cfg, filter_idx))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
