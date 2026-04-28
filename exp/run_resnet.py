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

from src.datasets import ImageDatasetConfig, get_image_dataset
from src.logger import MLFlowLogger
from src.models.resnet import ResNet
from src.trainer import Trainer
from src.utils import get_mlflow_tracking_uri
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="Small ResNet on image tensors",
    usage="uv run python exp/run_resnet.py --dataset MNIST",
)

group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", help="Dataset", default="MNIST")
group.add_argument("--transform", help="Dataset transformation", default=None)
group.add_argument("--power", type=float, help="Dataset transformation strength", default=0.0)

group.add_argument("--model", help="Model", default="ResNet")
group.add_argument(
    "--d_hidden",
    type=int,
    help="Base channel width (wider trunk uses 2x and 4x this)",
    default=32,
)

group.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
group.add_argument("--batch_size", type=int, help="Batch size", default=128)
group.add_argument("--epochs", type=int, help="Number of epochs", default=50)

group.add_argument("--seed", type=int, help="Seed", default=0)
group.add_argument("--device", type=int, help="CUDA device id", default=0)

group.add_argument("--project", help="MLflow project prefix", default="ImageModels")
group.add_argument("--experiment", help="Experiment name", default="Test")

group.add_argument("--num_workers", type=int, help="DataLoader workers", default=4)

args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

mlflow_url = get_mlflow_tracking_uri()
mlflow_project = "{}_{}".format(args.project, args.experiment)

device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.device("cpu")

images_train, images_val, images_test, meta = get_image_dataset(
    ImageDatasetConfig(
        dataset_str=args.dataset,
        seed=args.seed,
        transform_str=args.transform,
        power=args.power,
        output="2d",
    )
)
dataloader_train = DataLoader(images_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
dataloader_val = DataLoader(images_val, args.batch_size, num_workers=args.num_workers)
dataloader_test = DataLoader(images_test, args.batch_size, num_workers=args.num_workers)

model = ResNet(in_channels=1, out_channels=args.d_hidden, d_output=meta.n_classes)

print("Data:\t\t {}, transform={}, power={}".format(args.dataset, args.transform, args.power))
print("Model:\t\t base_channels={}".format(args.d_hidden))
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
