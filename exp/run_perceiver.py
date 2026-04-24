import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import warnings

warnings.filterwarnings("ignore")

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_pht_dataset, collate_fn
from torch.utils.data import DataLoader

from src.models.perceiver import PerceiverPHT
from src.trainer import TrainerPHTX
from src.logger import MLFlowLogger

import argparse


parser = argparse.ArgumentParser(
    description="Perceiver on persistence diagrams",
    usage="python3 run_perceiver.py --dataset MNIST --idx 0 4",
)

# dataset
group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="MNIST")
group.add_argument("--idx", help="Filters", type=int, nargs="+", default=[0, 2, 4, 6, 7, 9, 11, 13, 16])
group.add_argument("--eps", type=float, help="Eps", default=0.02)
group.add_argument("--transform", help="Dataset transformation", default=None)
group.add_argument("--power", type=float, help="Dataset transformation power", default=0.0)

# model
group.add_argument("--model", help="Model", default="PERCEIVER")
group.add_argument("--d_model", type=int, help="Input embedding dim", default=128)
group.add_argument("--d_latents", type=int, help="Latent dim", default=256)
group.add_argument("--num_latents", type=int, help="Number of latent vectors", default=128)
group.add_argument("--num_blocks", type=int, help="Number of blocks", default=1)
group.add_argument("--num_self_attends_per_block", type=int, help="Self-attention layers per block", default=6)
group.add_argument("--num_self_attention_heads", type=int, help="Self-attention heads", default=8)
group.add_argument("--num_cross_attention_heads", type=int, help="Cross-attention heads", default=8)
group.add_argument("--dropout", type=float, help="Attention dropout", default=0.0)

# optimization
group.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
group.add_argument("--batch_size", type=int, help="Batch size", default=128)
group.add_argument("--epochs", type=int, help="Number of epochs", default=20)

# experiment
group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=int, help="Device id", default=0)
group.add_argument("--experiment", help="Experiment", default="Test")

# workers
group.add_argument("--num_workers", type=int, help="N workers", default=2)

args = parser.parse_args()

# randomness
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# mlflow
mlflow_url = "http://192.168.31.240:10001"
mlflow_project = "PHTX_{}".format(args.experiment)

# device
device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# data
dataset_train, dataset_val, dataset_test, meta = get_pht_dataset(
    args.dataset, args.seed, args.idx, args.eps, args.transform, args.power
)
dataloader_train = DataLoader(
    dataset_train, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers
)
dataloader_val = DataLoader(dataset_val, args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
dataloader_test = DataLoader(dataset_test, args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

# model
model = PerceiverPHT(
    d_in=9,
    d_out=meta.n_classes,
    d_model=args.d_model,
    d_latents=args.d_latents,
    num_latents=args.num_latents,
    num_blocks=args.num_blocks,
    num_self_attends_per_block=args.num_self_attends_per_block,
    num_self_attention_heads=args.num_self_attention_heads,
    num_cross_attention_heads=args.num_cross_attention_heads,
    dropout=args.dropout,
)

print(
    "Data:\t\t {}, idx=[{}], eps={}, transform={}, power={}".format(
        args.dataset, ", ".join(map(str, args.idx)), args.eps, args.transform, args.power
    )
)
print(
    "Model:\t\t {}, d_model={}, d_latents={}, num_latents={}, blocks={}, self/blk={}, self_h={}, cross_h={}, dropout={}".format(
        args.model,
        args.d_model,
        args.d_latents,
        args.num_latents,
        args.num_blocks,
        args.num_self_attends_per_block,
        args.num_self_attention_heads,
        args.num_cross_attention_heads,
        args.dropout,
    )
)
print("Optimization:\t lr={}, batch size={}, seed={}, device={}".format(args.lr, args.batch_size, args.seed, args.device))

# fit
logger = MLFlowLogger(mlflow_url, mlflow_project, vars(args))
trainer = TrainerPHTX(model, device, logger)
trainer.fit(dataloader_train, dataloader_val, dataloader_test, lr=args.lr, n_epochs=args.epochs, desc="{}, {}".format(args.model, args.seed))
