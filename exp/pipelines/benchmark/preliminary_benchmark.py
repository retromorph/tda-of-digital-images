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

ROOT = Path(__file__).resolve().parents[3]


def _to_cli_flag(k):
    return "--{}".format(str(k))


def _append_arg(cmd, key, value):
    if isinstance(value, bool):
        cmd.append(_to_cli_flag(key) if value else "--no-{}".format(str(key)))
        return
    if isinstance(value, list):
        if len(value) == 0:
            return
        cmd.append(_to_cli_flag(key))
        cmd.extend([str(v) for v in value])
        return
    if value is None:
        return
    cmd.extend([_to_cli_flag(key), str(value)])


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


def build_cmd(cfg, task, method_name, method_cfg, seed):
    cmd = [
        sys.executable,
        str(ROOT / "exp" / "runners" / method_cfg["runner"]),
    ]

    experiment_name = "{}_{}".format(cfg["experiment"], task["name"])
    cmd.extend(["--experiment", experiment_name])
    cmd.extend(["--dataset", task["dataset"]])
    cmd.extend(["--seed", str(seed)])
    cmd.extend(["--device", str(cfg["device"])])
    cmd.extend(["--num_workers", str(cfg.get("num_workers", 0))])

    common_persistence = cfg.get("common_persistence", {})
    for k in ("idx", "eps"):
        if k in common_persistence:
            _append_arg(cmd, k, common_persistence[k])

    transform = task.get("transform", None)
    power = task.get("power", 0.0)
    if transform is not None:
        cmd.extend(["--transform", str(transform)])
    if power is not None:
        cmd.extend(["--power", str(power)])

    if "project" in method_cfg:
        cmd.extend(["--project", str(method_cfg["project"])])

    method_args = method_cfg.get("args", {})
    for k, v in method_args.items():
        _append_arg(cmd, k, v)

    return cmd


def _log_manifest_to_mlflow(cfg, cfg_path, manifest, total, failures):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("EXP_ARTIFACTS")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "preliminary_manifest.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)

        run_name = "preliminary_manifest_{}_{}".format(cfg["experiment"], int(time.time()))
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("experiment_root", str(cfg["experiment"]))
            mlflow.log_param("config_path", str(cfg_path))
            mlflow.log_metric("total_runs", total)
            mlflow.log_metric("failures", failures)
            mlflow.log_artifact(str(out_path), artifact_path="preliminary")
            return mlflow.get_artifact_uri("preliminary")


def run(cfg_path, dry_run=False, only_method=None, only_task=None):
    cfg_path = _resolve_cfg_path(Path(cfg_path))
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

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

        cmd = build_cmd(cfg, task, method_name, method_cfg, seed)
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

        row = {
            "config_path": str(cfg_path),
            "experiment_root": cfg["experiment"],
            "task_name": task["name"],
            "dataset": task["dataset"],
            "transform": task.get("transform", None),
            "power": task.get("power", 0.0),
            "method": method_name,
            "seed": seed,
            "runner": method_cfg["runner"],
            "project": method_cfg.get("project", None),
            "elapsed_s": elapsed,
            "exit_code": code,
            "command": cmd,
        }
        manifest.append(row)

        if code != 0:
            failures += 1

    artifact_uri = _log_manifest_to_mlflow(cfg, cfg_path, manifest, total, failures)
    print("\nLogged manifest artifact: {}".format(artifact_uri))
    print("Runs: {}, failures: {}".format(total, failures))
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified preliminary benchmark runner for persistence-based methods."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "exp" / "config" / "benchmark" / "preliminary_clean.yaml"),
        help="Path to YAML benchmark config.",
    )
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only_method", default=None, help="Run only one method key from config.")
    parser.add_argument("--only_task", default=None, help="Run only one task name from config.")
    args = parser.parse_args()

    raise SystemExit(
        run(
            cfg_path=Path(args.config),
            dry_run=args.dry_run,
            only_method=args.only_method,
            only_task=args.only_task,
        )
    )
