import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Keys consumed by training/optim/scheduler/budget — they should never end up in
# `model.args` or `encoder.args`. Add new training-level fields here.
_TRAIN_KEYS = {
    "model",
    "lr",
    "batch_size",
    "epochs",
    "weight_decay",
    "warmup_epochs",
    "eta_min",
    "grad_accum",
    "max_grad_norm",
    "early_stop_patience",
    "early_stop_min_delta",
    "scheduler",
}
_ENCODER_METHODS = {
    "persistent_image": {
        "encoder_name": "persistence_image",
        "model_name": "PERSISTENCE_CNN2D",
        "encoder_keys": {"resolution", "sigma2", "weighting", "weight_power"},
    },
    "persistent_landscape": {
        "encoder_name": "persistence_landscape",
        "model_name": "PERSISTENCE_CNN1D",
        "encoder_keys": {"num_layers", "resolution"},
    },
    "persistent_silhouette": {
        "encoder_name": "persistence_silhouette",
        "model_name": "PERSISTENCE_CNN1D",
        "encoder_keys": {"resolution", "weighting", "weight_power"},
    },
}


def _resolve_cfg_path(cfg_path: Path) -> Path:
    if cfg_path.exists():
        return cfg_path
    candidates = [
        ROOT / "exp" / "config" / "benchmark" / cfg_path.name,
        ROOT / "exp" / "config" / "ablations" / cfg_path.name,
        ROOT / "exp" / "config" / "smoke" / cfg_path.name,
        ROOT / "exp" / "config" / "legacy" / cfg_path.name,
        ROOT / "exp" / "config" / cfg_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return cfg_path


def _to_override(key, value):
    if isinstance(value, (dict, list, bool)) or value is None:
        payload = json.dumps(value)
    else:
        payload = str(value)
    return f"{key}={payload}"


def _split_method_args(method_name: str, raw_args: dict) -> tuple[dict, dict]:
    """Return (model_args, encoder_args). Training-level keys are dropped here."""
    non_train = {k: v for k, v in raw_args.items() if k not in _TRAIN_KEYS}
    encoder_cfg = _ENCODER_METHODS.get(method_name)
    if encoder_cfg is None:
        return non_train, {}
    encoder_args = {k: v for k, v in non_train.items() if k in encoder_cfg["encoder_keys"]}
    model_args = {k: v for k, v in non_train.items() if k not in encoder_cfg["encoder_keys"]}
    return model_args, encoder_args


def _build_overrides(cfg, task, method_name, method_cfg, seed):
    """Compose the full override list for one (method, task, seed) combination.

    Encoder + model + training overrides are produced together so that
    `model.args` is written exactly once.
    """
    raw_args = dict(method_cfg.get("args", {}))
    model_args, encoder_args = _split_method_args(method_name, raw_args)
    encoder_cfg = _ENCODER_METHODS.get(method_name)
    if encoder_cfg is not None:
        # PERSISTENCE_CNN1D needs `in_channels` to match landscape `num_layers`.
        if method_name == "persistent_landscape" and "num_layers" in encoder_args:
            model_args["in_channels"] = int(encoder_args["num_layers"])
        model_name = encoder_cfg["model_name"]
    else:
        model_name = raw_args.get("model", method_name)

    experiment_root = cfg["experiment"]
    overrides = [
        _to_override("seed", seed),
        _to_override("device", cfg["device"]),
        _to_override("num_workers", cfg.get("num_workers", 0)),
        _to_override("dataset.name", task["dataset"]),
        _to_override("dataset.test_time.transform", task.get("transform", None)),
        _to_override("dataset.test_time.power", task.get("power", 0.0)),
        _to_override("model.name", model_name),
        _to_override("model.args", model_args),
        _to_override("training.batch_size", raw_args.get("batch_size", 128)),
        _to_override("training.optimizer.lr", raw_args.get("lr", 1e-3)),
        _to_override("training.optimizer.weight_decay", raw_args.get("weight_decay", 0.0)),
        _to_override("training.scheduler.warmup_epochs", raw_args.get("warmup_epochs", 0)),
        _to_override("training.scheduler.eta_min", raw_args.get("eta_min", 0.0)),
        _to_override(
            "training.scheduler.name",
            "warmup_cosine" if raw_args.get("warmup_epochs", 0) > 0 else "none",
        ),
        _to_override("training.budget.value", raw_args.get("epochs", 10)),
        _to_override("training.grad_accum", raw_args.get("grad_accum", 1)),
        _to_override("training.max_grad_norm", raw_args.get("max_grad_norm", 0.0)),
        _to_override("training.early_stop.patience", raw_args.get("early_stop_patience", 0)),
        _to_override("training.early_stop.min_delta", raw_args.get("early_stop_min_delta", 0.0)),
        _to_override("logging.experiment", f"{experiment_root}/{task['name']}"),
    ]

    common = cfg.get("common_persistence", {}) or {}
    if "idx" in common:
        overrides.append(_to_override("filtration.diagram_idx", common["idx"]))
    if "eps" in common:
        overrides.append(_to_override("filtration.args.eps", common["eps"]))

    if encoder_cfg is not None:
        overrides.append(_to_override("encoder.name", encoder_cfg["encoder_name"]))
        overrides.append(_to_override("encoder.args", encoder_args))

    return overrides


def _log_manifest_to_mlflow(cfg, cfg_path, manifest, total, failures):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("EXP_ARTIFACTS")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "orchestrator_manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        run_name = "orchestrator_manifest_{}_{}".format(cfg["experiment"], int(time.time()))
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("experiment_root", str(cfg["experiment"]))
            mlflow.log_param("config_path", str(cfg_path))
            mlflow.log_metric("total_runs", total)
            mlflow.log_metric("failures", failures)
            mlflow.log_artifact(str(out_path), artifact_path="orchestrator")
            return mlflow.get_artifact_uri("orchestrator")


def run(cfg_path=None, *, cfg_dict=None, dry_run=False, only_method=None, only_task=None, inproc=False):
    """Run a sweep described by `cfg_dict` or by a YAML config at `cfg_path`.

    Either `cfg_path` or `cfg_dict` must be provided. When `cfg_dict` is given,
    its `__source` key (if present) is used as a manifest label; otherwise the
    label falls back to `<inline>`.
    """
    if cfg_dict is None:
        if cfg_path is None:
            raise ValueError("Either cfg_path or cfg_dict must be provided.")
        cfg_path = _resolve_cfg_path(Path(cfg_path))
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict(cfg_dict)
        cfg_path = Path(str(cfg.pop("__source", "<inline>")))

    methods = cfg["methods"]
    tasks = cfg["tasks"]
    seeds = cfg["seeds"]
    manifest = []
    total = 0
    failures = 0

    for seed, task, (method_name, method_cfg) in product(seeds, tasks, methods.items()):
        if only_method is not None and method_name != only_method:
            continue
        if only_task is not None and task["name"] != only_task:
            continue

        overrides = _build_overrides(cfg, task, method_name, method_cfg, seed)

        cmd = [sys.executable, str(ROOT / "runner.py"), "--config", str(ROOT / "exp" / "config" / "unified" / "default.yaml")]
        if inproc:
            cmd.append("--inproc")
        for item in overrides:
            cmd.extend(["--override", item])

        total += 1
        print("[{}/?] method={}, task={}, seed={}".format(total, method_name, task["name"], seed))
        print(" ".join(cmd))

        started = time.time()
        if dry_run:
            code = 0
        else:
            proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
            code = proc.returncode
        elapsed = time.time() - started

        manifest.append(
            {
                "config_path": str(cfg_path),
                "experiment_root": cfg["experiment"],
                "task_name": task["name"],
                "dataset": task["dataset"],
                "method": method_name,
                "seed": seed,
                "elapsed_s": elapsed,
                "exit_code": code,
                "command": cmd,
                "overrides": overrides,
            }
        )
        if code != 0:
            failures += 1

    artifact_uri = _log_manifest_to_mlflow(cfg, cfg_path, manifest, total, failures)
    print("\nLogged manifest artifact: {}".format(artifact_uri))
    print("Runs: {}, failures: {}".format(total, failures))
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified orchestrator for benchmark sweeps.")
    parser.add_argument("--config", default=str(ROOT / "exp" / "config" / "benchmark" / "preliminary_clean.yaml"))
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only_method", default=None)
    parser.add_argument("--only_task", default=None)
    parser.add_argument("--inproc", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    raise SystemExit(
        run(
            cfg_path=Path(args.config),
            dry_run=args.dry_run,
            only_method=args.only_method,
            only_task=args.only_task,
            inproc=args.inproc,
        )
    )
