import numpy as np
import torch

from types import SimpleNamespace

from tqdm import tqdm

import os
import pickle

import porespy as ps

from torch.utils.data import Dataset
from torchvision.datasets import MNIST, KMNIST, EMNIST, FashionMNIST
from torchvision.transforms.v2 import Compose, ToDtype, Lambda
from torchvision.transforms.v2 import GaussianBlur, GaussianNoise, RandomAffine, RandomPerspective, RandomRotation
from torchvision.transforms import InterpolationMode as IM

from src.transforms_dir import Direction

from src.persistence import pht

alphas = list(np.linspace(0, 360, 16+1)[:-1])
f = Direction(alphas, agg="add")
def pht_apply(img):
    return pht(f(img), img, pos=alphas)


def collate_fn(batch):

    # get len of a batch and len of each diagram in a batch
    n_batch = len(batch)
    d_lengths = [int(torch.argmax(D[:,0])) for D, y_ in batch]
    
    # set batch tensor to the max length of a diagram in a batch
    Ds = torch.ones([n_batch, max(d_lengths), 9]) * 0.
    D_masks = torch.zeros([n_batch, max(d_lengths)]).bool()
    ys = torch.zeros(n_batch).long()
    
    # populate targets, diagrams and their masks
    for i, (D, y) in enumerate(batch):
        Ds[i][:d_lengths[i]] = D[:d_lengths[i]]
        D_masks[i][d_lengths[i]:] = True
        ys[i] = y
    
    return Ds, D_masks, ys


class PersistenceTransformDataset(Dataset):

    def __init__(self, diagrams, targets, idx=None, eps=None):
        self.targets = targets
        idx = torch.tensor(idx) if idx is not None else None

        # get diagrams as list of tensors
        D = torch.ones([len(diagrams), max(map(len, diagrams))+1, 9]) * torch.inf

        # select points according to eps and idx
        for i, dgm in enumerate(diagrams):

            # eps
            if eps is not None:
                eps_idx = (dgm[:,1] - dgm[:,0]) >= eps
                dgm = dgm[eps_idx]

            # directions
            dgm_direction = torch.clone(dgm[:,-1])

            # sin
            # dgm[:,3] = torch.sin(dgm[:,3] * (torch.pi / 90))

            # idx
            if idx is not None:

                dgm_idx = torch.isin(dgm_direction, idx)
                dgm = dgm[dgm_idx]

                dgm_angle = torch.clone(dgm[:,-2]) # WAS torch.clone(dgm[:,-2])

                D[i,:len(dgm),:4] = dgm[:,:4]

                # position encoding
                D[i,:len(dgm),4] = torch.sin(dgm_angle * (torch.pi / 40))
                D[i,:len(dgm),5] = torch.sin(dgm_angle * (torch.pi / 90))
                D[i,:len(dgm),6] = torch.sin(dgm_angle * (torch.pi / 180))
                D[i,:len(dgm),7] = torch.sin(dgm_angle * (torch.pi / 130))
                D[i,:len(dgm),8] = torch.sin(dgm_angle * (torch.pi / 360))
            else:
                D[i,:len(dgm),:4] = dgm[:,:4]

        # cut to the largest diagram accross all dataset
        if idx is not None:
            max_len = torch.argmax(D[:,:,0], axis=1).max()
            D = D[:,:max_len+1] # leave at least one inf value!

        self.data = D

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class ImageDataset(Dataset):
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        X = self.data[idx]
        if self.transform:
            X = self.transform(X)
        y = self.targets[idx]

        return X, y
    

def get_dataset_cfg(dataset_str):

    datasets = {
        "MNIST": {
            "dataset_train_val": lambda: MNIST(root="./data/image", train=True, download=True),
            "dataset_test": lambda: MNIST(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10)
        },
        "KMNIST": {
            "dataset_train_val": lambda: KMNIST(root="./data/image", train=True, download=True),
            "dataset_test": lambda: KMNIST(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10)
        },
        "EMNIST-B": {
            "dataset_train_val": lambda: EMNIST(root="./data/image", split="balanced", train=True, download=True),
            "dataset_test": lambda: EMNIST(root="./data/image", split="balanced", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=47)
        },
        "EMNIST-L": {
            "dataset_train_val": lambda: EMNIST(root="./data/image", split="letters", train=True, download=True),
            "dataset_test": lambda: EMNIST(root="./data/image", split="letters", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=26)
        },
        "FMNIST": {
            "dataset_train_val": lambda: FashionMNIST(root="./data/image", train=True, download=True),
            "dataset_test": lambda: FashionMNIST(root="./data/image", train=False, download=True),
            "meta": SimpleNamespace(dim=28, n_classes=10)
        },
        "BLOBS": {
            "dataset_train_val": lambda: get_blobs_datasets(train=True),
            "dataset_test": lambda: get_blobs_datasets(train=False),
            "meta": SimpleNamespace(dim=64, n_classes=2)
        },
    }

    if dataset_str not in datasets.keys():
      raise ValueError("Supported datasets are '{}'.".format("', '".join(datasets.keys())))
    
    return datasets[dataset_str]


def get_blobs_datasets(train=True): 
    X_train_blobs, y_train_blobs, X_test_blobs, y_test_blobs = get_blobs()
    return ImageDataset(X_train_blobs, y_train_blobs) if train else ImageDataset(X_test_blobs, y_test_blobs)

def get_blobs():
    n_classes, samples_per_class, samples_per_class_test = 2, 30000, 5000
    params = {
        "0": {"porosity": 0.75, "blobiness": 0.501},
        "1": {"porosity": 0.75, "blobiness": 0.5}}

    X_train_blobs = torch.zeros((60000, 64, 64))
    X_test_blobs = torch.zeros((10000, 64, 64))
    y_train_blobs = torch.zeros((60000))
    y_test_blobs = torch.zeros((10000))

    for class_id in range(n_classes):
        porosity = params[f"{class_id}"]["porosity"]
        blobiness = params[f"{class_id}"]["blobiness"]

        for i in tqdm(range(samples_per_class)):
            img = ps.generators.blobs(shape=[64, 64],
                                    porosity=porosity,
                                    blobiness=blobiness,
                                    seed=np.random.randint(0, 1000))
            X_train_blobs[class_id * samples_per_class + i] = torch.from_numpy(img.astype(int))
            y_train_blobs[class_id * samples_per_class + i] = class_id

        for i in tqdm(range(samples_per_class_test)):
            img = ps.generators.blobs(shape=[64, 64],
                                    porosity=porosity,
                                    blobiness=blobiness,
                                    seed=np.random.randint(0, 1000))
            X_test_blobs[class_id * samples_per_class_test + i] = torch.from_numpy(img.astype(int))
            y_test_blobs[class_id * samples_per_class_test + i] = class_id

    return X_train_blobs, y_train_blobs.long(), X_test_blobs, y_test_blobs.long()


def get_transform(transform_str, power=0.0):

    transforms = {
        "affine": lambda power: RandomAffine(0, (power / 28, power / 28), interpolation=IM.BILINEAR),
        "noise": lambda power: GaussianNoise(sigma=power),
        "blur": lambda power: GaussianBlur(kernel_size=5, sigma=(float(power), float(power))),
        "perspective": lambda power: RandomPerspective(distortion_scale=power, p=1.0, interpolation=IM.BILINEAR),
        "rotation": lambda power: RandomRotation(degrees=power, interpolation=IM.BILINEAR),
    }

    if transform_str not in transforms.keys():
      raise ValueError("Supported transforms are '{}'.".format("', '".join(transforms.keys())))
    
    return transforms[transform_str](power)


def get_image_dataset(dataset_str, seed, transform_str=None, power=0.0, fractions=[5/6, 1/6], output="2d"):

    if output not in ["1d", "2d"]:
      raise ValueError("Supported outputs are '1d' or '2d'.")

    # transform
    transform_base = [Lambda(lambda x: x / 255), ToDtype(torch.float32)]
    transform_invariance = [get_transform(transform_str, power)] if transform_str is not None else []
    transform_flatten = [Lambda(lambda x: torch.flatten(x, 1))] if output=="1d" else []

    # train/val, test transforms
    transform_train_val = Compose(transform_base + transform_flatten)
    transform_test = Compose(transform_base + transform_invariance + transform_flatten)

    # datasets
    dataset_cfg = get_dataset_cfg(dataset_str)
    dataset_train_val_ = dataset_cfg["dataset_train_val"]()
    dataset_test_ = dataset_cfg["dataset_test"]()
    meta = dataset_cfg["meta"]

    # fractions
    f_train, _f_val = fractions
    n_train = int(len(dataset_train_val_) * f_train)

    # train, val, test as tensors
    train_val_random_idx = torch.randperm(len(dataset_train_val_), generator=torch.Generator().manual_seed(seed))

    X_train = transform_train_val(dataset_train_val_.data[train_val_random_idx[:n_train]].unsqueeze(1))
    y_train = dataset_train_val_.targets[train_val_random_idx[:n_train]]
    X_val = transform_train_val(dataset_train_val_.data[train_val_random_idx[n_train:]].unsqueeze(1))
    y_val = dataset_train_val_.targets[train_val_random_idx[n_train:]]
    X_test = transform_test(dataset_test_.data.unsqueeze(1))
    y_test = dataset_test_.targets

    # train, val, test datasets
    dataset_train = ImageDataset(X_train, y_train)
    dataset_val = ImageDataset(X_val, y_val)
    dataset_test = ImageDataset(X_test, y_test)

    if dataset_str=="EMNIST-L":
        y_train -= 1
        y_val -= 1
        y_test -= 1

    return dataset_train, dataset_val, dataset_test, meta


def get_pht_dataset(dataset_str, seed, idx=None, eps=None, transform_str=None, power=0.0, fractions=[5/6, 1/6]):

    # get image train/val/test datasets
    dataset_train, dataset_val, dataset_test, meta = get_image_dataset(dataset_str, seed, transform_str, power, fractions, "2d")
    
    # get labels
    y_train = dataset_train.targets
    y_val = dataset_val.targets
    y_test = dataset_test.targets

    # if datasets do not exist, create
    tstr = "{}-{}_".format(transform_str, power) if transform_str is not None else ""
    train_filename = "./data/diagrams/{}/{}_train_seed-{}.pkl".format(dataset_str, dataset_str, seed)
    val_filename = "./data/diagrams/{}/{}_val_seed-{}.pkl".format(dataset_str, dataset_str, seed)
    test_filename = "./data/diagrams/{}/{}_test_{}seed-{}.pkl".format(dataset_str, dataset_str, tstr, seed)

    # if at least single file missing
    if not (os.path.isfile(train_filename) and os.path.isfile(val_filename) and os.path.isfile(test_filename)):
        # print("TRAIN or VAL or TEST MISSING")

        print("Computing PHT of a dataset, this can take several minutes...")
        os.makedirs("./data/diagrams/{}".format(dataset_str), exist_ok=True)

        # if train or val missing
        if not (os.path.isfile(train_filename) and os.path.isfile(val_filename)):
            # print("TRAIN or VAL MISSING")
            # D_train = pool.map(pht_apply, dataset_train.data)
            
            D_train = []
            for item in tqdm(dataset_train.data):
                D_train.append(pht_apply(item))

            pickle.dump((D_train, y_train), open(train_filename, "wb"))
            print("Train complete")

            # D_val = pool.map(pht_apply, dataset_val.data)

            D_val = []
            for item in tqdm(dataset_val.data):
                D_val.append(pht_apply(item))

            pickle.dump((D_val, y_val), open(val_filename, "wb"))
            print("Val complete")
        
        else:
            # print("TRAIN, VAL PRESENT")
            (D_train, y_train) = pickle.load(open(train_filename, "rb"))
            (D_val, y_val) = pickle.load(open(val_filename, "rb"))

        # if test missing
        if not os.path.isfile(test_filename): 
            # print("TEST MISSING")
            # if transform_str is None:
            #     D_test = pool.map(pht_apply, dataset_test.data)
            # else:
            D_test = []
            for item in tqdm(dataset_test.data):
                D_test.append(pht_apply(item))
            
            pickle.dump((D_test, y_test), open(test_filename, "wb"))
            print("Test complete")
        
        else:
            # print("TEST PRESENT")
            (D_test, y_test) = pickle.load(open(test_filename, "rb"))
    
    else:
        # print("TRAIN, VAL, TEST PRESENT")
        (D_train, y_train) = pickle.load(open(train_filename, "rb"))
        (D_val, y_val) = pickle.load(open(val_filename, "rb"))
        (D_test, y_test) = pickle.load(open(test_filename, "rb"))

    if dataset_str=="EMNIST-L":
        y_train -= 1
        y_val -= 1
        y_test -= 1

    # PHT train/val/test datasets
    dataset_pht_train = PersistenceTransformDataset(D_train, y_train, idx=idx, eps=eps)
    dataset_pht_val = PersistenceTransformDataset(D_val, y_val, idx=idx, eps=eps)
    dataset_pht_test = PersistenceTransformDataset(D_test, y_test, idx=idx, eps=eps)

    return dataset_pht_train, dataset_pht_val, dataset_pht_test, meta