import torch

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import PersistenceDatasetConfig, get_persistence_dataset
from src.experiment import (
    build_collate,
    build_mlflow_logger,
    infer_output_dim,
    make_dataloader,
    resolve_device,
    safe_num_workers,
    seed_everything,
    update_runtime_metrics,
)
from src.models.deepsets import DeepSets
from src.trainer import Trainer
import time

_DEFAULT_IDX = list(range(0, 16 + 1, 2))

parser = argparse.ArgumentParser(
    description="DeepSets on persistence diagrams (PHTS)",
    usage="uv run python exp/runners/run_phts.py --dataset MNIST --idx 0 4",
)

group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="MNIST")
group.add_argument("--idx", help="Direction filters", type=int, nargs="+", default=_DEFAULT_IDX)
group.add_argument("--eps", type=float, help="Persistence epsilon threshold", default=0.0)
group.add_argument("--transform", help="Test-time transform name", default=None)
group.add_argument("--power", type=float, help="Test-time transform strength", default=0.0)

group.add_argument("--model", help="Model", default="PHTS")
group.add_argument("--d_model", type=int, help="Embedding dim", default=192)
group.add_argument("--d_hidden", type=int, help="Hidden dim", default=256)
group.add_argument("--dropout", type=float, help="Dropout", default=0.05)
group.add_argument("--activation", help="Activation (GELU, ReLU, ...)", default="GELU")
group.add_argument("--alpha", type=float, help="Activation alpha", default=0.0)

group.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
group.add_argument("--batch_size", type=int, help="Batch size", default=128)
group.add_argument("--epochs", type=int, help="Number of epochs", default=250)

group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=str, required=True, help="Device name (e.g. cpu, mps, cuda:0)")
group.add_argument("--project", help="MLflow project prefix", default="PHTModels")
group.add_argument("--experiment", help="Experiment name", default="Test")

group.add_argument("--num_workers", type=int, help="DataLoader workers", default=0)

args = parser.parse_args()
args.num_workers = safe_num_workers(args.num_workers)
seed_everything(args.seed)
device = resolve_device(args.device)

dataset_train, dataset_val, dataset_test, meta = get_persistence_dataset(
    PersistenceDatasetConfig(
        dataset_str=args.dataset,
        seed=args.seed,
        idx=args.idx,
        eps=args.eps,
        transform_str=args.transform,
        power=args.power,
    )
)
args.task = meta.task
collate = build_collate(meta.task)
dataloader_train = make_dataloader(dataset_train, args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
dataloader_val = make_dataloader(dataset_val, args.batch_size, collate_fn=collate, num_workers=args.num_workers)
dataloader_test = make_dataloader(dataset_test, args.batch_size, collate_fn=collate, num_workers=args.num_workers)

model = DeepSets(9, infer_output_dim(meta), args.d_model, args.d_hidden, args.dropout, args.activation, args.alpha)

print(
    "Data:\t\t {}, idx=[{}], eps={}, transform={}, power={}".format(
        args.dataset, ", ".join(map(str, args.idx)), args.eps, args.transform, args.power
    )
)
print(
    "Model:\t\t {}, d_model={}, d_hidden={}, dropout={}, activation={}".format(
        args.model, args.d_model, args.d_hidden, args.dropout, args.activation
    )
)
print("Optimization:\t lr={}, batch size={}, seed={}, device={}".format(args.lr, args.batch_size, args.seed, args.device))

logger = build_mlflow_logger(
    args,
    method_name=args.model,
    task_name=args.dataset,
    model=model,
    sample_batch=next(iter(dataloader_train)),
    forward_takes_mask=True,
)
trainer = Trainer(model, device, logger, task=meta.task, forward_takes_mask=True)
started_at = time.time()
trainer.fit(
    dataloader_train,
    dataloader_val,
    dataloader_test,
    lr=args.lr,
    n_epochs=args.epochs,
    desc="{}, {}".format(args.model, args.seed),
    close_logger=False,
)
update_runtime_metrics(logger, started_at, device)
logger.end()
