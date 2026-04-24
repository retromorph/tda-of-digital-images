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

from src.data import get_pht_dataset
from torch.utils.data import DataLoader

from src.models.phtx import PersistentHomologyTransformer
from src.trainer import TrainerPHTX
from src.logger import MLFlowLogger

import pickle
import argparse

# set parser
parser = argparse.ArgumentParser(
    description="Persistent Homology Transformer",
    usage="python3 run_phtx.py --dataset MNIST --idx 0 4"
)

# dataset
group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="MNIST")
group.add_argument("--idx", help="Filters", type=int, nargs="+", default=range(0, 17))
group.add_argument("--eps", type=float, help="Eps", default=0.02)
group.add_argument("--transform", help="Dataset transformation", default=None)
group.add_argument("--power", type=float, help="Dataset transformation power", default=0.0)

# model
group.add_argument("--model", help="Model", default="PHTX")
group.add_argument("--d_model", type=int, help="Model dim", default=128)
group.add_argument("--d_hidden", type=int, help="MLP hidden dim", default=128)
group.add_argument("--num_heads", type=int, help="Number of heads", default=8)
group.add_argument("--num_layers", type=int, help="Number of layers", default=2)
group.add_argument("--dropout", type=int, help="Dropout", default=0.1)
group.add_argument("--norm", help="Normalization", default="layer")
group.add_argument("--agg", help="Aggregation", default="max")
group.add_argument("--activation", help="Activation", default="GELU")
group.add_argument("--alpha", type=float, help="Alpha", default=0.0)
group.add_argument("--norm_first", type=bool, help="Norm first?", default=True)

# optimization
group.add_argument("--opt", help="Optimization method", default="AdamW")
group.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
group.add_argument("--batch_size", type=int, help="Batch size", default=128)
group.add_argument("--epochs", type=int, help="Number of epochs", default=200)

# experiment
group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=int, help="Device id", default=0)
group.add_argument("--experiment", help="Experiment", default="Test")

# workers
group.add_argument("--num_workers", type=int, help="N workers", default=2) # best=2

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
mlflow_project = "PHTX_debug"

# device
device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# data
dim = 28
n_classes = 10
# images_train, images_val, images_test, diagrams_train, diagrams_val, diagrams_test, dim, n_classes = get_dataset(args.dataset, args.seed, args.transform, args.power, output="2d")
# images_train = SubsetWithSamples(images_train)
# images_val = SubsetWithSamples(images_val)
# dataset_train = PersistenceTransformDataset(images_train, diagrams_train, torch.tensor(args.idx), args.eps)
# dataset_val = PersistenceTransformDataset(images_val, diagrams_val, torch.tensor(args.idx), args.eps)
# dataset_test = PersistenceTransformDataset(images_test, diagrams_test, torch.tensor(args.idx), args.eps)

D_train = pickle.load(open("./data/diagrams/MNIST_D_train.pkl", "rb"))
D_val = pickle.load(open("./data/diagrams/MNIST_D_val.pkl", "rb"))
D_test = pickle.load(open("./data/diagrams/MNIST_D_test.pkl", "rb"))

y_train = pickle.load(open("./data/diagrams/MNIST_y_train.pkl", "rb"))
y_val = pickle.load(open("./data/diagrams/MNIST_y_val.pkl", "rb"))
y_test = pickle.load(open("./data/diagrams/MNIST_y_test.pkl", "rb"))

dataset_train = PersistenceTransformDataset2(D_train, y_train, torch.tensor(args.idx), args.eps)
dataset_val = PersistenceTransformDataset2(D_val, y_val, torch.tensor(args.idx), args.eps)
dataset_test = PersistenceTransformDataset2(D_test, y_test, torch.tensor(args.idx), args.eps)

dataloader_train = DataLoader(dataset_train, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers) 
dataloader_val = DataLoader(dataset_val, args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
dataloader_test = DataLoader(dataset_test, args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

# model
model = PersistentHomologyTransformer(None, 9, n_classes, args.d_model, args.d_hidden, args.num_heads, args.num_layers, args.agg, args.norm, args.dropout, args.activation, args.alpha, args.norm_first)

print("Data:\t\t {}, idx=[{}], eps={}, transform={}, power={}".format(args.dataset, ", ".join(map(str, args.idx)), args.eps, args.transform, args.power))
print("Model:\t\t {}, d_model={}, d_hidden={}, num_heads={}, num_layers={}, dropout={}, agg={}, norm={}, f={}, pre_ln={}".format(args.model, args.d_model, args.d_hidden, args.num_heads, args.num_layers, args.dropout, args.agg, args.norm, args.activation, int(args.norm_first)))
print("Optimization:\t lr={}, batch size={}, seed={}, device={}".format(args.lr, args.batch_size, args.seed, args.device))

# fit
logger = MLFlowLogger(mlflow_url, mlflow_project, vars(args))
trainer = TrainerPHTX(model, device, logger)
trainer.fit(dataloader_train, dataloader_val, dataloader_test, lr=args.lr, n_epochs=args.epochs, desc="{}, {}".format(args.model, args.seed))