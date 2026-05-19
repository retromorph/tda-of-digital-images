"""Main 3D paper comparison (M7, Series 1.2).

Launches two orchestrator sweeps from a shared base config:

  1. converge mode — early-stop on val loss (patience=20, max 200 epochs)
  2. wallclock mode — fixed 45-min budget with Pareto snapshots @ 5/15/45 min

Run:
    uv run python -m exp.pipelines.main_3d_paper            # both modes
    uv run python -m exp.pipelines.main_3d_paper --mode converge
    uv run python -m exp.pipelines.main_3d_paper --mode wallclock --dry_run
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.orchestrator import run  # noqa: E402


def _build_cfg(base: dict, mode_name: str) -> dict:
    """Compose a per-mode orchestrator config (tasks/methods Cartesian)."""
    mode = base["modes"][mode_name]
    scheduler = mode.get("scheduler", "warmup_cosine")
    early = mode.get("early_stop") or {}

    methods = {}
    for name, m in base["methods"].items():
        args = dict(m.get("args", {}))
        # Wallclock mode uses constant LR (scheduler='none' via warmup_epochs=0)
        # so the Pareto-curve snapshots aren't confounded by where in the cosine
        # schedule we read out.
        if scheduler == "none":
            args["warmup_epochs"] = 0
        # Push per-mode early-stop into each method (orchestrator reads from raw_args).
        args["early_stop_patience"] = int(early.get("patience", 0))
        args["early_stop_min_delta"] = float(early.get("min_delta", 0.0))
        methods[name] = {"args": args}

    tasks = [
        {"name": f"{ds.lower()}__{mode_name}", "dataset": ds, "transform": None, "power": 0.0}
        for ds in base["datasets"]
    ]
    return {
        "experiment": f"{base['experiment']}/{mode_name}",
        "device": base.get("device", "cpu"),
        "num_workers": base.get("num_workers", 0),
        "seeds": base["seeds"],
        "tasks": tasks,
        "common_persistence": base["common_persistence"],
        "logging": base.get("logging", {}),
        "budget": mode["budget"],
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["converge", "wallclock", "both"], default="both")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--only_method", default=None)
    parser.add_argument("--only_task", default=None)
    parser.add_argument("--config", default=str(ROOT / "exp" / "config" / "benchmark" / "main_3d_paper.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)

    modes = ["converge", "wallclock"] if args.mode == "both" else [args.mode]
    rc = 0
    for m in modes:
        sub = _build_cfg(copy.deepcopy(base), m)
        rc |= run(
            cfg_dict=sub,
            dry_run=args.dry_run,
            only_method=args.only_method,
            only_task=args.only_task,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
