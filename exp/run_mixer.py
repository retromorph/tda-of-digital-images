import random
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_image_dataset
from torch.utils.data import DataLoader

from src.models.mlp import MLP
from src.trainer import Trainer
from src.logger import MLFlowLogger

from utils import save_checkpoint

import argparse

# set parser
parser = argparse.ArgumentParser(
    description="Persistent Homology Transformer",
    usage="python3 run_mlp.py --dataset MNIST"
)

# dataset
group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="MNIST")
group.add_argument("--transform", help="Dataset transformation", default=None)
group.add_argument("--power", type=float, help="Dataset transformation power", default=0.0)

# model
group.add_argument("--model", help="Model", default="Mixer")
group.add_argument("--d_hidden", type=int, help="MLP hidden dim", default=256)
group.add_argument("--num_layers", type=int, help="Number of layers", default=2)
group.add_argument("--dropout", type=int, help="Dropout", default=0.1)
group.add_argument("--activation", help="Activation", default="ReLU")
group.add_argument("--norm_first", type=bool, help="Norm first?", default=True)

# optimization
group.add_argument("--opt", help="Optimization method", default="AdamW")
group.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
group.add_argument("--batch_size", type=int, help="Batch size", default=128)
group.add_argument("--epochs", type=int, help="Number of epochs", default=5)

# experiment
group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=int, help="Device id", default=0)

# logs, checkpoints
group.add_argument("--project", help="Project", default="PHTX_Alpha")
group.add_argument("--experiment", help="Experiment", default="Test")
group.add_argument("--checkpoint", type=bool, help="Save checkpoint?", default=False)

# workers
group.add_argument("--num_workers", type=int, help="N workers", default=4)

# parse command line
args = parser.parse_args()

# randomness
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# random state
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# mflow
mlflow_url = "http://192.168.31.240:10001"
mlflow_project = "{}_{}".format(args.project, args.experiment)

# device
device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# data
images_train, images_val, images_test, meta = get_image_dataset(args.dataset, args.seed, args.transform, args.power, output="1d")
dataloader_train = DataLoader(images_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
dataloader_val = DataLoader(images_val, args.batch_size, num_workers=args.num_workers)
dataloader_test = DataLoader(images_test, args.batch_size, num_workers=args.num_workers)

# model
model = MLP(meta.dim**2, meta.n_classes, args.d_hidden, args.dropout, args.norm_first, args.activation)

print("Data:\t\t {}, transform={}, power={}".format(args.dataset, args.transform, args.power))
print("Model:\t\t d_hidden={}, num_layers={}, dropout={}, f={}, pre_ln={}".format(args.d_hidden, args.num_layers, args.dropout, args.activation, int(args.norm_first)))
print("Optimization:\t lr={}, batch size={}, seed={}, device={}".format(args.lr, args.batch_size, args.seed, args.device))

# fit
logger = MLFlowLogger(mlflow_url, mlflow_project, vars(args))
trainer = Trainer(model, device, logger)
trainer.fit(dataloader_train, dataloader_val, dataloader_test, lr=args.lr, n_epochs=args.epochs, desc="{}, {}".format(args.model, args.seed))

# save checkpoint
if args.checkpoint:
    save_checkpoint(model, trainer.optimizer, args)