import os
import random
import subprocess
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.logger import MLFlowLogger
from src.utils import get_mlflow_tracking_uri

from fvcore.nn import FlopCountAnalysis


def seed_everything(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    return torch.device(device_name)


def safe_num_workers(requested: int) -> int:
    if os.name == "posix" and os.uname().sysname.lower() == "darwin" and requested > 0:
        print("macOS spawn safety: overriding num_workers {} -> 0".format(requested))
        return 0
    return requested


def make_dataloader(dataset, batch_size, *, shuffle=False, collate_fn=None, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def build_experiment_name(experiment: str, dataset: str, method: str) -> str:
    return "{}/{}/{}".format(experiment, dataset, method)


def infer_output_dim(meta):
    if getattr(meta, "task", "classification") == "regression":
        return 1
    return int(getattr(meta, "n_classes"))


def _git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return commit
    except Exception:
        return "unknown"


def _count_trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _flops_per_sample(model, batch, with_mask=False):
    if FlopCountAnalysis is None:
        return 0
    try:
        if with_mask:
            x, mask, _ = batch
            flops = FlopCountAnalysis(model, (x[:1], mask[:1]))
        else:
            x, _ = batch
            flops = FlopCountAnalysis(model, x[:1])
        return int(flops.total())
    except Exception:
        return 0


def build_collate(task):
    from src.datasets.types import collate_fn

    return partial(collate_fn, task=task)


def build_mlflow_logger(args, method_name, task_name, model=None, sample_batch=None, forward_takes_mask=False):
    tracking_uri = get_mlflow_tracking_uri()
    experiment_name = build_experiment_name(args.experiment, task_name, method_name)
    run_name = "{}-{}-seed{}".format(method_name, task_name, args.seed)
    tags = {
        "dataset": task_name,
        "model": method_name,
        "seed": args.seed,
        "task": getattr(args, "task", "classification"),
        "git_commit": _git_commit(),
    }
    logger = MLFlowLogger(
        url=tracking_uri,
        project=experiment_name,
        params=vars(args),
        run_name=run_name,
        tags=tags,
    )
    compute_metrics = {
        "params_count": _count_trainable_params(model) if model is not None else 0,
        "flops_per_sample": _flops_per_sample(model, sample_batch, with_mask=forward_takes_mask)
        if (model is not None and sample_batch is not None)
        else 0,
        "peak_gpu_mem_mb": 0.0,
        "train_wallclock_s": 0.0,
    }
    logger.log(compute_metrics, step=0)
    return logger


def update_runtime_metrics(logger, started_at_s: float, device: torch.device):
    elapsed = max(0.0, time.time() - started_at_s)
    peak_gpu_mem_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    logger.log(
        {
            "train_wallclock_s": elapsed,
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
        },
        step=0,
    )
