"""Sequence-length scaling benchmark (M1, Series 1.1).

For each (method, N) measures forward and forward+backward time and peak GPU
memory on synthetic persistence diagrams of length N. The output CSV is the
data source for the central paper figure showing that Persformer scales
quadratically (and OOMs at large N) while LinearPersformer and
LatentPersformer stay flat.

Run:
    uv run python -m exp.pipelines.scaling_seqlen --config exp/config/scaling/seqlen.yaml
or single-point smoke:
    uv run python -m exp.pipelines.scaling_seqlen --N 1024 --method LATENT_PERSFORMER --device cpu --reps 5
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.base import get_model_spec  # noqa: E402
from src.persistence.synthetic import random_diagram_batch  # noqa: E402


METHODS_DEFAULT = ["PERSFORMER", "LINEAR_PERSFORMER", "LATENT_PERSFORMER", "PHTS"]
N_DEFAULT = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        # MPS exposes only current_allocated_memory; manually track via a baseline.
        pass


def _peak_mem_mb(device: torch.device, baseline_mb: float = 0.0) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / 2**20
    if device.type == "mps":
        return float(torch.mps.current_allocated_memory()) / 2**20 - baseline_mb
    return 0.0


def _build_model(method: str, device: torch.device) -> torch.nn.Module:
    """Construct a model with fixed-size config (independent of dataset)."""
    spec = get_model_spec(method)

    class _Meta:
        task = "regression"
        n_classes = None

    meta = _Meta()
    if method == "PERSFORMER":
        model = spec.build(meta, d_model=128, d_hidden=512, num_layers=4, num_heads=8)
    elif method == "LINEAR_PERSFORMER":
        model = spec.build(meta, d_model=128, intermediate_size=512, num_hidden_layers=4, num_attention_heads=8)
    elif method == "LATENT_PERSFORMER":
        model = spec.build(
            meta,
            d_model=128,
            d_latents=256,
            num_latents=64,
            num_blocks=1,
            num_self_attends_per_block=4,
            num_self_attention_heads=8,
            num_cross_attention_heads=8,
            pooling="attn",
        )
    elif method == "PHTS":
        model = spec.build(meta, d_model=128, d_hidden=256)
    else:
        raise ValueError(f"Unknown method {method!r}")
    return model.to(device)


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _benchmark_one(
    method: str,
    n_points: int,
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    reps: int,
) -> dict:
    """Forward + backward timing at fixed (method, N). Returns a result row."""
    row = {
        "method": method,
        "N": n_points,
        "batch_size": batch_size,
        "fwd_ms": float("nan"),
        "bwd_ms": float("nan"),
        "peak_mem_mb": float("nan"),
        "params": 0,
        "oom": False,
        "error": "",
    }
    try:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = _build_model(method, device)
        row["params"] = _count_params(model)

        X, mask = random_diagram_batch(batch_size, n_points, device=device)

        # Warmup
        model.train()
        for _ in range(warmup):
            y = model(X, mask)
            loss = (y * y).sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
        _device_sync(device)
        _reset_peak_mem(device)
        baseline = _peak_mem_mb(device)

        # Forward-only timing
        model.eval()
        fwd_times = []
        with torch.no_grad():
            for _ in range(reps):
                _device_sync(device)
                t0 = time.perf_counter()
                _ = model(X, mask)
                _device_sync(device)
                fwd_times.append((time.perf_counter() - t0) * 1000.0)

        # Forward + backward timing
        model.train()
        bwd_times = []
        for _ in range(reps):
            _device_sync(device)
            t0 = time.perf_counter()
            y = model(X, mask)
            loss = (y * y).sum()
            loss.backward()
            _device_sync(device)
            bwd_times.append((time.perf_counter() - t0) * 1000.0)
            model.zero_grad(set_to_none=True)

        row["fwd_ms"] = sum(fwd_times) / len(fwd_times) / batch_size
        row["bwd_ms"] = sum(bwd_times) / len(bwd_times) / batch_size
        row["peak_mem_mb"] = _peak_mem_mb(device, baseline_mb=0.0)
    except torch.cuda.OutOfMemoryError:
        row["oom"] = True
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            row["oom"] = True
        else:
            row["error"] = str(e)[:300]
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def run_sweep(
    *,
    methods: Iterable[str],
    n_values: Iterable[int],
    device_str: str,
    batch_size: int,
    warmup: int,
    reps: int,
    output_csv: Path,
) -> None:
    device = torch.device(device_str)
    rows = []
    for method in methods:
        oom_locked = False
        for N in n_values:
            if oom_locked:
                rows.append(
                    {
                        "method": method,
                        "N": N,
                        "batch_size": batch_size,
                        "fwd_ms": float("nan"),
                        "bwd_ms": float("nan"),
                        "peak_mem_mb": float("nan"),
                        "params": 0,
                        "oom": True,
                        "error": "lock-after-prior-OOM",
                    }
                )
                continue
            print(f"[{method}] N={N} ...", flush=True)
            row = _benchmark_one(method, N, device=device, batch_size=batch_size, warmup=warmup, reps=reps)
            print(
                f"  fwd={row['fwd_ms']:.3f} ms/sample, bwd={row['bwd_ms']:.3f} ms/sample, "
                f"peak_mem={row['peak_mem_mb']:.1f} MB, params={row['params']}, "
                f"oom={row['oom']}{(' err=' + row['error']) if row['error'] else ''}",
                flush=True,
            )
            rows.append(row)
            if row["oom"]:
                oom_locked = True

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--output", default="results/scaling_seqlen.csv")
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        methods = cfg.get("methods", METHODS_DEFAULT)
        n_values = cfg.get("N", N_DEFAULT)
        device_str = cfg.get("device", args.device)
        batch_size = int(cfg.get("batch_size", args.batch_size))
        warmup = int(cfg.get("warmup", args.warmup))
        reps = int(cfg.get("reps", args.reps))
        output = Path(cfg.get("output", args.output))
    else:
        methods = [args.method] if args.method else METHODS_DEFAULT
        n_values = [args.N] if args.N else N_DEFAULT
        device_str = args.device
        batch_size = args.batch_size
        warmup = args.warmup
        reps = args.reps
        output = Path(args.output)

    run_sweep(
        methods=methods,
        n_values=n_values,
        device_str=device_str,
        batch_size=batch_size,
        warmup=warmup,
        reps=reps,
        output_csv=output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
