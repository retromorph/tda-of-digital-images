import argparse
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import PersistenceDatasetConfig, collate_fn, get_persistence_dataset
from src.logger import MLFlowLogger
from src.models.latent_persformer import LatentPersformer
from src.trainer import TrainerPersformer
from src.utils import get_mlflow_tracking_uri

parser = argparse.ArgumentParser(
    description="LatentPersformer on persistence diagrams",
    usage="uv run python exp/runners/run_latent_persformer.py --dataset MNIST --idx 0 4",
)

group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="BLOBS")
group.add_argument("--idx", help="Direction filters", type=int, nargs="+", default=[0, 2, 4, 6, 7, 9, 11, 13, 16])
group.add_argument("--eps", type=float, help="Persistence epsilon threshold", default=0.02)
group.add_argument("--transform", help="Test-time transform name", default=None)
group.add_argument("--power", type=float, help="Test-time transform strength", default=0.0)

group.add_argument("--model", help="Model", default="LATENT_PERSFORMER")
group.add_argument("--d_model", type=int, help="Input model dim", default=128)
group.add_argument("--d_latents", type=int, help="Latent dim", default=256)
group.add_argument("--num_latents", type=int, help="Number of latent vectors", default=128)
group.add_argument("--num_blocks", type=int, help="Perceiver blocks", default=1)
group.add_argument("--num_self_attends_per_block", type=int, help="Self-attention layers per block", default=4)
group.add_argument("--num_self_attention_heads", type=int, help="Self-attention heads", default=8)
group.add_argument("--num_cross_attention_heads", type=int, help="Cross-attention heads", default=8)
group.add_argument("--cross_attention_widening_factor", type=int, help="Cross-attention FFN widening", default=1)
group.add_argument("--self_attention_widening_factor", type=int, help="Self-attention FFN widening", default=1)
group.add_argument("--dropout", type=float, help="Perceiver attention dropout", default=0.1)
group.add_argument("--decoder_dropout", type=float, help="Dropout in MLP decoder", default=0.2)
group.add_argument(
    "--decoder_hidden_dims",
    type=str,
    help="Comma-separated hidden widths for MLP decoder (e.g. 256,256,64)",
    default="256,256,64",
)
group.add_argument("--activation", help="Activation (GELU, ReLU, ...)", default="GELU")

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
group.add_argument("--device", type=int, help="CUDA device id", default=0)
group.add_argument("--experiment", help="Experiment name", default="Test")
group.add_argument("--num_workers", type=int, help="DataLoader workers", default=2)

args = parser.parse_args()
if sys.platform == "darwin" and args.num_workers > 0:
    print("macOS spawn safety: overriding num_workers {} -> 0".format(args.num_workers))
    args.num_workers = 0

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

mlflow_url = get_mlflow_tracking_uri()
mlflow_project = "PERSFORMER_{}".format(args.experiment)

device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.device("cpu")

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
dataloader_train = DataLoader(
    dataset_train, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers
)
dataloader_val = DataLoader(dataset_val, args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
dataloader_test = DataLoader(dataset_test, args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

decoder_dims = tuple(int(x.strip()) for x in args.decoder_hidden_dims.split(",") if x.strip())
eta_min_ratio = args.eta_min / args.lr if args.lr > 0 else 0.0

model = LatentPersformer(
    transform=None,
    d_in=9,
    d_out=meta.n_classes,
    d_model=args.d_model,
    d_latents=args.d_latents,
    num_latents=args.num_latents,
    num_blocks=args.num_blocks,
    num_self_attends_per_block=args.num_self_attends_per_block,
    num_self_attention_heads=args.num_self_attention_heads,
    num_cross_attention_heads=args.num_cross_attention_heads,
    cross_attention_widening_factor=args.cross_attention_widening_factor,
    self_attention_widening_factor=args.self_attention_widening_factor,
    dropout=args.dropout,
    decoder_hidden_dims=decoder_dims,
    decoder_dropout=args.decoder_dropout,
    activation=args.activation,
)

print(
    "Data:\t\t {}, idx=[{}], eps={}, transform={}, power={}".format(
        args.dataset, ", ".join(map(str, args.idx)), args.eps, args.transform, args.power
    )
)
print(
    "Model:\t\t {}, d_model={}, d_latents={}, n_latents={}, blocks={}, self/blk={}, self_h={}, cross_h={}, do={}, dec_do={}, dec_h={}".format(
        args.model,
        args.d_model,
        args.d_latents,
        args.num_latents,
        args.num_blocks,
        args.num_self_attends_per_block,
        args.num_self_attention_heads,
        args.num_cross_attention_heads,
        args.dropout,
        args.decoder_dropout,
        decoder_dims,
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

logger = MLFlowLogger(mlflow_url, mlflow_project, vars(args))
trainer = TrainerPersformer(model, device, logger)
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
)
