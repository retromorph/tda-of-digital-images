import random
import numpy as np
import torch

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import collate_fn, get_pht_dataset
from src.logger import MLFlowLogger
from src.models.phtx import PersistentHomologyTransformer
from src.trainer import TrainerPHTX
from src.utils import get_mlflow_tracking_uri
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="Persistent Homology Transformer (PHTX)",
    usage="uv run python exp/run_phtx.py --dataset MNIST --idx 0 4",
)

group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="BLOBS")
group.add_argument("--idx", help="Direction filters", type=int, nargs="+", default=[0, 2, 4, 6, 7, 9, 11, 13, 16])
group.add_argument("--eps", type=float, help="Persistence epsilon threshold", default=0.02)
group.add_argument("--transform", help="Test-time transform name", default=None)
group.add_argument("--power", type=float, help="Test-time transform strength", default=0.0)

group.add_argument("--model", help="Model", default="PHTX")
group.add_argument("--d_model", type=int, help="Model dim", default=128)
group.add_argument("--d_hidden", type=int, help="FFN hidden dim", default=192)
group.add_argument("--num_heads", type=int, help="Attention heads", default=8)
group.add_argument("--num_layers", type=int, help="Transformer encoder layers", default=2)
group.add_argument("--dropout", type=float, help="Dropout", default=0.1)
group.add_argument("--norm", help="Post-encoder norm: layer, batch, or none", default="layer")
group.add_argument("--agg", help="Diagram pooling: mean or max", default="max")
group.add_argument("--activation", help="Transformer activation (GELU, ReLU, ...)", default="GELU")
group.add_argument("--alpha", type=float, help="Activation alpha", default=0.0)
group.add_argument(
    "--norm_first",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Transformer norm_first (--norm_first / --no-norm_first)",
)

group.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
group.add_argument("--batch_size", type=int, help="Batch size", default=128)
group.add_argument("--epochs", type=int, help="Number of epochs", default=20)

group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=int, help="CUDA device id", default=0)
group.add_argument("--experiment", help="Experiment name", default="Test")

group.add_argument("--num_workers", type=int, help="DataLoader workers", default=2)

args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

mlflow_url = get_mlflow_tracking_uri()
mlflow_project = "PHTX_{}".format(args.experiment)

device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.device("cpu")

dataset_train, dataset_val, dataset_test, meta = get_pht_dataset(
    args.dataset, args.seed, args.idx, args.eps, args.transform, args.power
)
dataloader_train = DataLoader(
    dataset_train, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers
)
dataloader_val = DataLoader(dataset_val, args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
dataloader_test = DataLoader(dataset_test, args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

norm_kw = None if str(args.norm).lower() in ("none", "null", "") else args.norm
model = PersistentHomologyTransformer(
    None,
    9,
    meta.n_classes,
    args.d_model,
    args.d_hidden,
    args.num_heads,
    args.num_layers,
    args.agg,
    norm_kw,
    args.dropout,
    args.activation,
    args.alpha,
    args.norm_first,
)

print(
    "Data:\t\t {}, idx=[{}], eps={}, transform={}, power={}".format(
        args.dataset, ", ".join(map(str, args.idx)), args.eps, args.transform, args.power
    )
)
print(
    "Model:\t\t {}, d_model={}, d_hidden={}, num_heads={}, num_layers={}, dropout={}, agg={}, norm={}, act={}, pre_ln={}".format(
        args.model,
        args.d_model,
        args.d_hidden,
        args.num_heads,
        args.num_layers,
        args.dropout,
        args.agg,
        args.norm,
        args.activation,
        int(args.norm_first),
    )
)
print("Optimization:\t lr={}, batch size={}, seed={}, device={}".format(args.lr, args.batch_size, args.seed, args.device))

logger = MLFlowLogger(mlflow_url, mlflow_project, vars(args))
trainer = TrainerPHTX(model, device, logger)
trainer.fit(
    dataloader_train,
    dataloader_val,
    dataloader_test,
    lr=args.lr,
    n_epochs=args.epochs,
    desc="{}, {}".format(args.model, args.seed),
)
