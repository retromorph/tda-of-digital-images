import random
import numpy as np
import torch

from itertools import product
from tqdm import tqdm

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import PersistenceDatasetConfig, get_persistence_dataset

# randomness
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# random state
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

seeds = [0, 1, 2, 3, 4]
datasets = ["EMNIST-L"]
transforms = {
    None: [0.0],
}

for seed, dataset, transform in product(seeds, datasets, transforms):
    for power in transforms[transform]:
        print("Dataset: {}, transform: {}, power: {}, seed: {}".format(dataset, transform, power, seed))
        get_persistence_dataset(
            PersistenceDatasetConfig(
                dataset_str=dataset,
                seed=seed,
                transform_str=transform,
                power=power,
            )
        )