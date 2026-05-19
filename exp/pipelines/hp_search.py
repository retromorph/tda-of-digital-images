"""Hyperparameter grid search for the main 3D paper benchmark (M6).

Sweeps a small lr × weight_decay grid for each method on a single calibration
dataset (D1=DEEPORE-3D, seed=42), selects best by val loss, and writes
``best_hp.yaml`` consumed by ``exp/config/benchmark/main_3d_paper.yaml`` (M7).

Per the paper Methods section: HP is tuned on D1 only and held fixed across
datasets to avoid per-dataset cherry-picking.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.pipelines.orchestrator import run  # noqa: E402

DEFAULT_LR_GRID = [1e-4, 3e-4, 1e-3]
DEFAULT_WD_GRID = [0.0, 1e-4]


def _hp_tag(lr: float, wd: float) -> str:
    return f"lr{lr:g}_wd{wd:g}".replace("0.", "")


def _build_cfg(base: dict, lr: float, wd: float) -> dict:
    """Compose an orchestrator config that runs every method at one HP combo."""
    methods = {}
    for name, base_method in base["methods"].items():
        m = dict(base_method)
        m["args"] = dict(m.get("args", {}))
        m["args"]["lr"] = lr
        m["args"]["weight_decay"] = wd
        methods[name] = m

    tasks = [
        {"name": ds.lower(), "dataset": ds, "transform": None, "power": 0.0}
        for ds in base["datasets"]
    ]
    cfg = {
        "experiment": base["experiment"] + "_" + _hp_tag(lr, wd),
        "device": base.get("device", "cpu"),
        "num_workers": base.get("num_workers", 0),
        "seeds": base["seeds"],
        "tasks": tasks,
        "methods": methods,
    }
    if "common_persistence" in base:
        cfg["common_persistence"] = base["common_persistence"]
    if "logging" in base:
        cfg["logging"] = base["logging"]
    if "budget" in base:
        cfg["budget"] = base["budget"]
    return cfg


def _select_best_per_method(experiment_prefix: str, methods: list[str], output_path: Path) -> None:
    """Query MLflow for all HP-search runs and pick the lowest val-loss config per method."""
    import mlflow

    client = mlflow.tracking.MlflowClient()
    best: dict[str, dict] = {}
    for exp in client.search_experiments():
        if not exp.name.startswith(experiment_prefix):
            continue
        for run in client.search_runs([exp.experiment_id], max_results=10000):
            method = run.data.params.get("method") or run.data.tags.get("method")
            if method is None:
                continue
            if method not in methods:
                continue
            loss_val = run.data.metrics.get("loss_val_best") or run.data.metrics.get("loss_val")
            if loss_val is None:
                continue
            lr = float(run.data.params.get("lr", "nan"))
            wd = float(run.data.params.get("weight_decay", "nan"))
            cand = {"lr": lr, "weight_decay": wd, "loss_val": float(loss_val), "run_id": run.info.run_id}
            if method not in best or cand["loss_val"] < best[method]["loss_val"]:
                best[method] = cand

    out = {"best_hp": {m: {k: v[k] for k in ("lr", "weight_decay")} for m, v in best.items()}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(out, sort_keys=True))
    print(f"Wrote best HP per method to {output_path}")
    for m, v in best.items():
        print(f"  {m}: lr={v['lr']:.1e}, wd={v['weight_decay']:.1e}, loss_val={v['loss_val']:.4f}")


def main() -> int:
    cfg_path = ROOT / "exp" / "config" / "hp_search" / "main.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    lr_grid = cfg.get("lr_grid", DEFAULT_LR_GRID)
    wd_grid = cfg.get("wd_grid", DEFAULT_WD_GRID)

    rc = 0
    for lr, wd in itertools.product(lr_grid, wd_grid):
        sub = _build_cfg(cfg, lr, wd)
        rc |= run(cfg_dict=sub)

    output_path = Path(cfg.get("output", "exp/config/hp_search/best_hp.yaml"))
    _select_best_per_method(cfg["experiment"], list(cfg["methods"].keys()), ROOT / output_path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
