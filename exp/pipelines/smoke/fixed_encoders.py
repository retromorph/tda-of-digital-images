"""Smoke test for fixed encoders, ported on top of the unified orchestrator.

Builds an orchestrator-shaped config that wires `persistent_image`,
`persistent_landscape` and `persistent_silhouette` methods. Subset truncation
(`max_train` / `max_val` / `max_test` from the legacy script) is not yet
supported by the unified runner, so this smoke test runs a single epoch on the
full dataset; reduce `epochs` further or pick a small dataset to keep it cheap.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.orchestrator import run  # noqa: E402

_METHOD_NAMES = ("persistence_image", "persistence_landscape", "persistence_silhouette")


def _resolve_cfg_path() -> Path:
    new_path = ROOT / "exp" / "config" / "smoke" / "fixed_encoders_smoke.yaml"
    if new_path.exists():
        return new_path
    return ROOT / "exp" / "config" / "fixed_encoders_smoke.yaml"


def _method_args(legacy_cfg: dict, encoder_section: str) -> dict:
    section = legacy_cfg.get(encoder_section, {})
    args = {
        "lr": legacy_cfg.get("lr", 1e-3),
        "batch_size": legacy_cfg.get("batch_size", 128),
        "epochs": legacy_cfg.get("epochs", 1),
    }
    # Encoder-specific knobs come as singleton lists in the legacy config; flatten them.
    for key, value in section.items():
        if key == "runner":
            continue
        if isinstance(value, list):
            value = value[0] if value else None
        args[key] = value
    return args


def _build_orchestrator_cfg(legacy_cfg: dict) -> dict:
    method_name_map = {
        "persistence_image": "persistent_image",
        "persistence_landscape": "persistent_landscape",
        "persistence_silhouette": "persistent_silhouette",
    }
    methods = {}
    for legacy_section in _METHOD_NAMES:
        if legacy_section not in legacy_cfg:
            continue
        methods[method_name_map[legacy_section]] = {"args": _method_args(legacy_cfg, legacy_section)}
    tasks = [{"name": ds.lower(), "dataset": ds, "transform": None, "power": 0.0} for ds in legacy_cfg["datasets"]]
    return {
        "experiment": legacy_cfg.get("experiment", "FixedEncodersSmoke"),
        "device": legacy_cfg.get("device", "cpu"),
        "num_workers": legacy_cfg.get("num_workers", 0),
        "seeds": legacy_cfg["seeds"],
        "tasks": tasks,
        "methods": methods,
        "__source": str(_resolve_cfg_path()),
    }


def main():
    with open(_resolve_cfg_path(), "r", encoding="utf-8") as f:
        legacy_cfg = yaml.load(f, Loader=yaml.FullLoader)
    raise SystemExit(run(cfg_dict=_build_orchestrator_cfg(legacy_cfg)))


if __name__ == "__main__":
    main()
