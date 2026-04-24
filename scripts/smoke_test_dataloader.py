import random
import numpy as np
import torch

from src.data import get_image_dataset, get_pht_dataset, collate_fn
from torch.utils.data import DataLoader

# randomness
seed = 4
idx = [0, 4]
eps = 0.02

# get image datasets
# dataset_train, dataset_val, dataset_test, meta = get_image_dataset("BLOBS", seed, output="2d")

# print(dataset_train.data.shape)

# print(dataset_train.targets.unique())
# print(dataset_val.targets.shape)
# print(dataset_test.targets.shape)

# # get PHT datasets
dataset_pht_train, dataset_pht_val, dataset_pht_test, _ = get_pht_dataset("MNIST", seed)
dataloader = DataLoader(dataset_pht_train, batch_size=4, shuffle=False, collate_fn=collate_fn)

X, X_mask, Y = next(iter(dataloader))

print("X", X.shape)
print("M", X_mask.shape)

print(X_mask)

# print(dataset_pht_train.data.shape, dataset_pht_train.targets.shape)
# print(dataset_pht_val.data.shape, dataset_pht_val.targets.shape)
# print(dataset_pht_test.data.shape, dataset_pht_test.targets.shape)
