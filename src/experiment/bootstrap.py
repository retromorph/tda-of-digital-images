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

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:  # pragma: no cover - optional dep
    FlopCountAnalysis = None


def seed_everything(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name) -> torch.device:
    """Accept device as string ('cpu' / 'mps' / 'cuda:0') or as numeric index (legacy CLI).

    Numeric values are interpreted as CUDA device indexes when CUDA is available,
    otherwise they fall back to CPU so that legacy `--device 0` usage doesn't crash
    on CPU-only laptops.
    """
    if isinstance(device_name, int):
        return torch.device(f"cuda:{device_name}") if torch.cuda.is_available() else torch.device("cpu")
    text = str(device_name).strip()
    if text.lstrip("-").isdigit():
        idx = int(text)
        return torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(text)


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
    """Returns (flops_per_sample, status). status ∈ {ok, unavailable, failed}."""
    if FlopCountAnalysis is None:
        return 0, "unavailable"
    try:
        if with_mask:
            x, mask, _ = batch
            flops = FlopCountAnalysis(model, (x[:1], mask[:1]))
        else:
            x, _ = batch
            flops = FlopCountAnalysis(model, x[:1])
        return int(flops.total()), "ok"
    except Exception:
        return 0, "failed"


def build_collate(task, idx=None, eps=None, sin_encoding_config=None):
    from src.datasets.types import collate_fn

    return partial(collate_fn, task=task, idx=idx, eps=eps, sin_encoding_config=sin_encoding_config)


def select_dataset_output_by_input_kind(input_kind: str) -> str:
    if input_kind == "flat":
        return "1d"
    return "2d"


def select_forward_takes_mask(input_kind: str) -> bool:
    return input_kind == "diagram"


def build_mlflow_logger(
    *,
    experiment_name: str,
    run_name: str,
    params: dict | None,
    tags: dict | None,
    model=None,
    sample_batch=None,
    forward_takes_mask: bool = False,
):
    """Create a MLflow run, log static params/tags and one-shot compute counters.

    Compute counters logged here at step 0:
        - params_count
        - flops_per_sample (best-effort; tag `flops_status` records ok|unavailable|failed)
    Wallclock and peak memory are logged later by `update_runtime_metrics`.
    """
    tracking_uri = get_mlflow_tracking_uri()
    base_tags = {"git_commit": _git_commit()}
    base_tags.update(tags or {})

    logger = MLFlowLogger(
        url=tracking_uri,
        project=experiment_name,
        params=params,
        run_name=run_name,
        tags=base_tags,
    )

    if model is not None:
        params_count = _count_trainable_params(model)
        flops_value, flops_status = (
            _flops_per_sample(model, sample_batch, with_mask=forward_takes_mask)
            if sample_batch is not None
            else (0, "unavailable")
        )
        logger.log({"params_count": params_count, "flops_per_sample": flops_value}, step=0)
        logger.log_tags({"flops_status": flops_status})
    return logger


def update_runtime_metrics(logger, started_at_s: float, device: torch.device, *, step: int = 0):
    elapsed = max(0.0, time.time() - started_at_s)
    peak_gpu_mem_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    logger.log(
        {
            "train_wallclock_s": elapsed,
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
        },
        step=step,
    )
