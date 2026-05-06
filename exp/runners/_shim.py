import argparse
import json
import warnings
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner import run_with_overrides


def _parse_unknown(unknown: list[str]) -> dict:
    parsed = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:].replace("-", "_")
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            value = unknown[i + 1]
            i += 2
        else:
            value = True
            i += 1
        parsed[key] = value
    return parsed


def _s(value):
    return json.dumps(value) if isinstance(value, (list, dict, bool)) else str(value)


def run_legacy_shim(
    *,
    model_name: str,
    input_kind: str,
    encoder_name: str | None = None,
    model_arg_keys: set[str] | None = None,
    encoder_arg_keys: set[str] | None = None,
):
    warnings.warn(
        "Legacy runner is deprecated; forwarding to unified runner.py.",
        DeprecationWarning,
        stacklevel=2,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "exp" / "config" / "unified" / "default.yaml"))
    parser.add_argument("--dataset", default="MNIST")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--experiment", default="Test")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--transform", default=None)
    parser.add_argument("--power", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--idx", type=int, nargs="+", default=[0, 2, 4, 6, 7, 9, 11, 13, 16])
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--model", default=model_name)
    args, unknown = parser.parse_known_args()

    unknown_kv = _parse_unknown(unknown)
    model_keys = model_arg_keys or set()
    enc_keys = encoder_arg_keys or set()
    model_args = {}
    encoder_args = {}
    for k, v in unknown_kv.items():
        if k in enc_keys:
            encoder_args[k] = v
        elif k in model_keys:
            model_args[k] = v

    overrides = [
        f"seed={args.seed}",
        f"device={args.device}",
        f"num_workers={args.num_workers}",
        f"dataset.name={args.dataset}",
        f"dataset.test_time.transform={_s(args.transform)}",
        f"dataset.test_time.power={args.power}",
        f"model.name={args.model}",
        f"training.batch_size={args.batch_size}",
        f"training.optimizer.name=adamw",
        f"training.optimizer.lr={args.lr}",
        f"training.optimizer.weight_decay={args.weight_decay}",
        "training.scheduler.name=warmup_cosine" if args.warmup_epochs > 0 else "training.scheduler.name=none",
        f"training.scheduler.warmup_epochs={args.warmup_epochs}",
        f"training.scheduler.eta_min={args.eta_min}",
        f"training.budget.kind=epochs",
        f"training.budget.value={args.epochs}",
        f"logging.experiment={args.experiment}",
    ]
    if input_kind in {"diagram", "encoded"}:
        overrides.extend(
            [
                "filtration.name=pht_directional",
                f"filtration.diagram_idx={_s(args.idx)}",
                f"filtration.args.eps={args.eps}",
            ]
        )
    if encoder_name is not None:
        overrides.append(f"encoder.name={encoder_name}")
    for k, v in model_args.items():
        overrides.append(f"model.args.{k}={_s(v)}")
    for k, v in encoder_args.items():
        overrides.append(f"encoder.args.{k}={_s(v)}")
    run_with_overrides(args.config, overrides)
