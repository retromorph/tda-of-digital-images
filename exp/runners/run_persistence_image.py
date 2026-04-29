import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import PersistenceDatasetConfig, get_persistence_dataset
from src.fixed_encoders import EncodedFeatureDataset, PersistenceImageEncoder
from src.logger import MLFlowLogger
from src.models.persistence_cnn2d import PersistenceCNN2D
from src.trainer import Trainer
from src.utils import get_mlflow_tracking_uri


parser = argparse.ArgumentParser(
    description="Persistence Image + small 2D CNN classifier",
    usage="uv run python exp/runners/run_persistence_image.py --dataset MNIST",
)
group = parser.add_argument_group("Algorithm configuration")
group.add_argument("--dataset", default="MNIST")
group.add_argument("--idx", type=int, nargs="+", default=[0, 2, 4, 6, 7, 9, 11, 13, 16])
group.add_argument("--eps", type=float, default=0.02)
group.add_argument("--transform", default=None)
group.add_argument("--power", type=float, default=0.0)
group.add_argument("--resolution", type=int, default=28)
group.add_argument("--sigma2", type=float, default=1.0)
group.add_argument("--weighting", default="linear")
group.add_argument("--weight_power", type=float, default=1.0)
group.add_argument("--base_channels", type=int, default=16)
group.add_argument("--dropout", type=float, default=0.1)
group.add_argument("--lr", type=float, default=0.0003)
group.add_argument("--batch_size", type=int, default=128)
group.add_argument("--epochs", type=int, default=20)
group.add_argument("--seed", type=int, default=0)
group.add_argument(
    "--device",
    type=str,
    required=True,
    help="Device name (e.g. cpu, mps, cuda:0)",
)
group.add_argument("--project", default="FixedEncoders")
group.add_argument("--experiment", default="Smoke")
group.add_argument("--num_workers", type=int, default=0)
group.add_argument("--max_train", type=int, default=None)
group.add_argument("--max_val", type=int, default=None)
group.add_argument("--max_test", type=int, default=None)
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
encoder = PersistenceImageEncoder(
    resolution=args.resolution,
    sigma2=args.sigma2,
    weighting=args.weighting,
    weight_power=args.weight_power,
)
base_cache_config = {
    "dataset_str": args.dataset,
    "seed": args.seed,
    "idx": args.idx,
    "eps": args.eps,
    "transform_str": args.transform,
    "power": args.power,
    "fractions": [5 / 6, 1 / 6],
}

train_features = EncodedFeatureDataset(dataset_train, encoder, split="train", base_cache_config=base_cache_config)
val_features = EncodedFeatureDataset(dataset_val, encoder, split="val", base_cache_config=base_cache_config)
test_features = EncodedFeatureDataset(dataset_test, encoder, split="test", base_cache_config=base_cache_config)

if args.max_train is not None:
    train_features = Subset(train_features, range(min(args.max_train, len(train_features))))
if args.max_val is not None:
    val_features = Subset(val_features, range(min(args.max_val, len(val_features))))
if args.max_test is not None:
    test_features = Subset(test_features, range(min(args.max_test, len(test_features))))

dataloader_train = DataLoader(train_features, args.batch_size, shuffle=True, num_workers=args.num_workers)
dataloader_val = DataLoader(val_features, args.batch_size, num_workers=args.num_workers)
dataloader_test = DataLoader(test_features, args.batch_size, num_workers=args.num_workers)

model = PersistenceCNN2D(
    d_output=meta.n_classes,
    in_channels=1,
    base_channels=args.base_channels,
    dropout=args.dropout,
)

print("Data:\t\t {}, idx=[{}], eps={}".format(args.dataset, ", ".join(map(str, args.idx)), args.eps))
print(
    "Encoder:\t PI, resolution={}, sigma2={}, weighting={}, weight_power={}".format(
        args.resolution, args.sigma2, args.weighting, args.weight_power
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
    desc="PI-{}, {}".format(args.dataset, args.seed),
)
