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

from src.models.capsnet import FinalCapsNet
from src.trainer import TrainerCapsNet
from src.logger import MLFlowLogger

import argparse

# set parser
parser = argparse.ArgumentParser(
    description="Persistent Homology Transformer",
    usage="python3 run_resnet.py --dataset MNIST"
)

# dataset
group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="MNIST")
group.add_argument("--transform", help="Dataset transformation", default=None)
group.add_argument("--power", type=float, help="Dataset transformation power", default=0.0)

# model
group.add_argument("--model", help="Model", default="CapsNet")
# group.add_argument("--d_hidden", type=int, help="Hidden dim", default=32)
# group.add_argument("--n_blocks", type=int, help="Number of residual blocks", default=2)
# group.add_argument("--dropout", type=int, help="Dropout", default=0.0)
# group.add_argument("--activation", help="Activation", default="ReLU")
# group.add_argument("--norm_first", type=bool, help="Norm first?", default=False)

# optimization
group.add_argument("--opt", help="Optimization method", default="AdamW")
group.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
group.add_argument("--batch_size", type=int, help="Batch size", default=128)
group.add_argument("--epochs", type=int, help="Number of epochs", default=50)

# experiment
group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=int, help="Device id", default=0)
group.add_argument("--experiment", help="Experiment", default="Test")

# workers
group.add_argument("--num_workers", type=int, help="N workers", default=0)

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
mlflow_project = "PHTX_{}".format(args.experiment)

# device
device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# data
images_train, images_val, images_test, meta = get_image_dataset(args.dataset, args.seed, args.transform, args.power, output="2d")
dataloader_train = DataLoader(images_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
dataloader_val = DataLoader(images_val, args.batch_size, num_workers=args.num_workers)
dataloader_test = DataLoader(images_test, args.batch_size, num_workers=args.num_workers)

# model
model = FinalCapsNet() # TODO: assumes 28x28 input

print("Data:\t\t {}, transform={}, power={}".format(args.dataset, args.transform, args.power))
# print("Model:\t\t d_hidden={}, n_blocks={}".format(args.d_hidden, args.n_blocks, args.dropout, args.activation, int(args.norm_first)))
print("Optimization:\t lr={}, batch size={}, seed={}, device={}".format(args.lr, args.batch_size, args.seed, args.device))

# fit
logger = MLFlowLogger(mlflow_url, mlflow_project, vars(args))
trainer = TrainerCapsNet(model, device, logger)
trainer.fit(dataloader_train, dataloader_val, dataloader_test, lr=args.lr, n_epochs=args.epochs, desc="{}, {}".format(args.model, args.seed))