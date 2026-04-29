import random
import numpy as np
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
from src.logger import MLFlowLogger
from src.models.mlp import MLP
from src.trainer import Trainer
from src.utils import get_mlflow_tracking_uri, save_checkpoint
from torch.utils.data import DataLoader

# set parser
parser = argparse.ArgumentParser(
    description="MLP baseline on flattened image pixels",
    usage="uv run python exp/runners/run_mlp.py --dataset MNIST",
)

# dataset
group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="MNIST")
group.add_argument("--transform", help="Dataset transformation (noise, blur, affine, perspective, rotation)", default=None)
group.add_argument("--power", type=float, help="Dataset transformation strength", default=0.0)

# model
group.add_argument("--model", help="Model", default="MLP")
group.add_argument("--d_hidden", type=int, help="MLP hidden dim", default=256)
group.add_argument("--num_layers", type=int, help="Number of hidden blocks (>=1)", default=2)
group.add_argument("--dropout", type=float, help="Dropout", default=0.1)
group.add_argument("--activation", help="Activation (GELU, ReLU, ELU, LeakyReLU, CELU)", default="ReLU")
group.add_argument("--alpha", type=float, help="Activation alpha (LeakyReLU/CELU)", default=0.0)

# optimization
group.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
group.add_argument("--batch_size", type=int, help="Batch size", default=128)
group.add_argument("--epochs", type=int, help="Number of epochs", default=5)

# experiment
group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=str, required=True, help="Device name (e.g. cpu, mps, cuda:0)")

# logs, checkpoints
group.add_argument("--project", help="MLflow project prefix", default="ImageModels")
group.add_argument("--experiment", help="Experiment name", default="Test")
group.add_argument(
    "--checkpoint",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save checkpoint (--checkpoint / --no-checkpoint)",
)

# workers
group.add_argument("--num_workers", type=int, help="DataLoader workers", default=4)

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
mlflow_project = "{}_{}".format(args.project, args.experiment)

device = torch.device(args.device)

images_train, images_val, images_test, meta = get_image_dataset(
    ImageDatasetConfig(
        dataset_str=args.dataset,
        seed=args.seed,
        transform_str=args.transform,
        power=args.power,
        output="1d",
    )
)
dataloader_train = DataLoader(images_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
dataloader_val = DataLoader(images_val, args.batch_size, num_workers=args.num_workers)
dataloader_test = DataLoader(images_test, args.batch_size, num_workers=args.num_workers)

model = MLP(
    meta.dim**2,
    meta.n_classes,
    args.d_hidden,
    args.dropout,
    num_layers=args.num_layers,
    activation=args.activation,
    alpha=args.alpha,
)

print("Data:\t\t {}, transform={}, power={}".format(args.dataset, args.transform, args.power))
print(
    "Model:\t\t d_hidden={}, num_layers={}, dropout={}, activation={}".format(
        args.d_hidden, args.num_layers, args.dropout, args.activation
    )
)
print("Optimization:\t lr={}, batch size={}, seed={}, device={}".format(args.lr, args.batch_size, args.seed, args.device))

logger = MLFlowLogger(mlflow_url, mlflow_project, vars(args))
trainer = Trainer(model, device, logger)
trainer.fit(
    dataloader_train,
    dataloader_val,
    dataloader_test,
    lr=args.lr,
    n_epochs=args.epochs,
    desc="{}, {}".format(args.model, args.seed),
)

if args.checkpoint:
    save_checkpoint(model, trainer.optimizer, args)
