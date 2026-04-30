import torch

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import ImageDatasetConfig, get_image_dataset
from src.experiment import (
    build_mlflow_logger,
    infer_output_dim,
    make_dataloader,
    resolve_device,
    safe_num_workers,
    seed_everything,
    update_runtime_metrics,
)
from src.models.vit import ViT
from src.trainer import Trainer
import time

parser = argparse.ArgumentParser(
    description="ViT (HuggingFace) on image tensors",
    usage="uv run python exp/runners/run_vit.py --dataset MNIST",
)

group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="MNIST")
group.add_argument("--transform", help="Dataset transformation", default=None)
group.add_argument("--power", type=float, help="Dataset transformation strength", default=0.0)

group.add_argument("--model", help="Model", default="ViT")
group.add_argument("--d_model", type=int, help="ViT hidden size", default=128)
group.add_argument("--d_hidden", type=int, help="ViT MLP intermediate size", default=128)
group.add_argument("--n_heads", type=int, help="Attention heads", default=8)
group.add_argument("--n_blocks", type=int, help="Transformer layers", default=2)
group.add_argument("--patch_size", type=int, help="Patch size", default=7)
group.add_argument("--dropout", type=float, help="Dropout", default=0.1)

group.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
group.add_argument("--batch_size", type=int, help="Batch size", default=128)
group.add_argument("--epochs", type=int, help="Number of epochs", default=50)

group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=str, required=True, help="Device name (e.g. cpu, mps, cuda:0)")

group.add_argument("--project", help="MLflow project prefix", default="ImageModels")
group.add_argument("--experiment", help="Experiment name", default="Test")

group.add_argument("--num_workers", type=int, help="DataLoader workers", default=6)

args = parser.parse_args()
args.num_workers = safe_num_workers(args.num_workers)
seed_everything(args.seed)
device = resolve_device(args.device)

images_train, images_val, images_test, meta = get_image_dataset(
    ImageDatasetConfig(
        dataset_str=args.dataset,
        seed=args.seed,
        transform_str=args.transform,
        power=args.power,
        output="2d",
    )
)
dataloader_train = make_dataloader(images_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
dataloader_val = make_dataloader(images_val, args.batch_size, num_workers=args.num_workers)
dataloader_test = make_dataloader(images_test, args.batch_size, num_workers=args.num_workers)
args.task = meta.task

model = ViT(
    meta.dim,
    infer_output_dim(meta),
    args.d_model,
    args.d_hidden,
    args.n_heads,
    args.n_blocks,
    args.patch_size,
    dropout=args.dropout,
)

print("Data:\t\t {}, transform={}, power={}".format(args.dataset, args.transform, args.power))
print(
    "Model:\t\t d_model={}, d_hidden={}, n_heads={}, n_blocks={}, patch_size={}, dropout={}".format(
        args.d_model, args.d_hidden, args.n_heads, args.n_blocks, args.patch_size, args.dropout
    )
)
print("Optimization:\t lr={}, batch size={}, seed={}, device={}".format(args.lr, args.batch_size, args.seed, args.device))

logger = build_mlflow_logger(args, method_name=args.model, task_name=args.dataset, model=model)
trainer = Trainer(model, device, logger, task=meta.task)
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
