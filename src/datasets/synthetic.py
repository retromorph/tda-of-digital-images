import numpy as np
import porespy as ps
import torch

from tqdm import tqdm

from src.datasets.types import ImageDataset

def get_blobs_dataset(train=True):
    x_train, y_train, x_test, y_test = get_blobs()
    if train:
        return ImageDataset(x_train, y_train)
    return ImageDataset(x_test, y_test)


def get_blobs():
    n_classes, samples_per_class, samples_per_class_test = 2, 30000, 5000
    params = {
        "0": {"porosity": 0.75, "blobiness": 0.501},
        "1": {"porosity": 0.75, "blobiness": 0.5},
    }

    x_train = torch.zeros((60000, 64, 64))
    x_test = torch.zeros((10000, 64, 64))
    y_train = torch.zeros((60000))
    y_test = torch.zeros((10000))

    for class_id in range(n_classes):
        porosity = params[f"{class_id}"]["porosity"]
        blobiness = params[f"{class_id}"]["blobiness"]

        for i in tqdm(range(samples_per_class)):
            img = ps.generators.blobs(
                shape=[64, 64],
                porosity=porosity,
                blobiness=blobiness,
                seed=np.random.randint(0, 1000),
            )
            x_train[class_id * samples_per_class + i] = torch.from_numpy(img.astype(int))
            y_train[class_id * samples_per_class + i] = class_id

        for i in tqdm(range(samples_per_class_test)):
            img = ps.generators.blobs(
                shape=[64, 64],
                porosity=porosity,
                blobiness=blobiness,
                seed=np.random.randint(0, 1000),
            )
            x_test[class_id * samples_per_class_test + i] = torch.from_numpy(img.astype(int))
            y_test[class_id * samples_per_class_test + i] = class_id

    return x_train, y_train.long(), x_test, y_test.long()
