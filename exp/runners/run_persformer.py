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
from src.models.persformer import Persformer
from src.trainer import Trainer
import time

parser = argparse.ArgumentParser(
    description="Persformer on persistence diagrams",
    usage="uv run python exp/runners/run_persformer.py --dataset MNIST --idx 0 4",
)

group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="BLOBS")
group.add_argument("--idx", help="Direction filters", type=int, nargs="+", default=[0, 2, 4, 6, 7, 9, 11, 13, 16])
group.add_argument("--eps", type=float, help="Persistence epsilon threshold", default=0.02)
group.add_argument("--transform", help="Test-time transform name", default=None)
group.add_argument("--power", type=float, help="Test-time transform strength", default=0.0)

group.add_argument("--model", help="Model", default="PERSFORMER")
group.add_argument("--d_model", type=int, help="Model dim", default=128)
group.add_argument("--d_hidden", type=int, help="FFN hidden dim (encoder)", default=512)
group.add_argument("--num_heads", type=int, help="Attention heads", default=8)
group.add_argument("--num_layers", type=int, help="Transformer encoder layers", default=5)
group.add_argument("--encoder_dropout", type=float, help="Dropout in encoder + pooling attention", default=0.0)
group.add_argument("--decoder_dropout", type=float, help="Dropout in MLP decoder", default=0.2)
group.add_argument(
    "--decoder_hidden_dims",
    type=str,
    help="Comma-separated hidden widths for MLP decoder (e.g. 256,256,64)",
    default="256,256,64",
)
group.add_argument(
    "--pooling_heads",
    type=int,
    default=None,
    help="MHA pooling heads (default: same as --num_heads)",
)
group.add_argument("--norm", help="Post-encoder norm: layer, batch, or none", default="layer")
group.add_argument("--activation", help="Transformer activation (GELU, ReLU, ...)", default="GELU")
group.add_argument("--alpha", type=float, help="Activation alpha", default=0.0)
group.add_argument(
    "--norm_first",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Transformer norm_first (--norm_first / --no-norm_first)",
)

group.add_argument("--lr", type=float, help="Peak learning rate (after warmup)", default=1e-3)
group.add_argument("--weight_decay", type=float, help="AdamW weight decay", default=1e-4)
group.add_argument("--warmup_epochs", type=int, help="Linear LR warmup length", default=10)
group.add_argument(
    "--eta_min",
    type=float,
    help="Minimum LR after cosine decay (absolute; internally passed as eta_min/lr to trainer)",
    default=1e-6,
)
group.add_argument("--batch_size", type=int, help="Batch size", default=32)
group.add_argument("--epochs", type=int, help="Number of epochs", default=1000)

group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=str, required=True, help="Device name (e.g. cpu, mps, cuda:0)")
group.add_argument("--experiment", help="Experiment name", default="Test")

group.add_argument("--num_workers", type=int, help="DataLoader workers", default=2)

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
collate = build_collate(meta.task, idx=args.idx, eps=args.eps)
dataloader_train = make_dataloader(dataset_train, args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
dataloader_val = make_dataloader(dataset_val, args.batch_size, collate_fn=collate, num_workers=args.num_workers)
dataloader_test = make_dataloader(dataset_test, args.batch_size, collate_fn=collate, num_workers=args.num_workers)

norm_kw = None if str(args.norm).lower() in ("none", "null", "") else args.norm
decoder_dims = tuple(int(x.strip()) for x in args.decoder_hidden_dims.split(",") if x.strip())

# trainer expects eta_min as a ratio (LR_min / LR_peak) for scheduler setup
# argparse eta_min is absolute min LR; convert to ratio here
eta_min_ratio = args.eta_min / args.lr if args.lr > 0 else 0.0

model = Persformer(
    transform=None,
    d_in=9,
    d_out=infer_output_dim(meta),
    d_model=args.d_model,
    d_hidden=args.d_hidden,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    norm=norm_kw,
    encoder_dropout=args.encoder_dropout,
    decoder_hidden_dims=decoder_dims,
    decoder_dropout=args.decoder_dropout,
    activation=args.activation,
    alpha=args.alpha,
    norm_first=args.norm_first,
    pooling_heads=args.pooling_heads,
)

print(
    "Data:\t\t {}, idx=[{}], eps={}, transform={}, power={}".format(
        args.dataset, ", ".join(map(str, args.idx)), args.eps, args.transform, args.power
    )
)
print(
    "Model:\t\t {}, d_model={}, d_hidden={}, num_heads={}, num_layers={}, enc_do={}, dec_do={}, dec_h={}, norm={}, act={}, pre_ln={}, pool_h={}".format(
        args.model,
        args.d_model,
        args.d_hidden,
        args.num_heads,
        args.num_layers,
        args.encoder_dropout,
        args.decoder_dropout,
        decoder_dims,
        args.norm,
        args.activation,
        int(args.norm_first),
        args.pooling_heads if args.pooling_heads is not None else args.num_heads,
    )
)
print(
    "Optimization:\t lr={}, wd={}, warmup={}, eta_min={}, batch={}, epochs={}, seed={}, device={}".format(
        args.lr,
        args.weight_decay,
        args.warmup_epochs,
        args.eta_min,
        args.batch_size,
        args.epochs,
        args.seed,
        device,
    )
)

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
    weight_decay=args.weight_decay,
    warmup_epochs=args.warmup_epochs,
    eta_min=eta_min_ratio,
    scheduler="warmup_cosine",
    close_logger=False,
)
update_runtime_metrics(logger, started_at, device)
logger.end()
