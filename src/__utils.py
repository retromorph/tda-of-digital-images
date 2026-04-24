import torch
import torch.nn as nn

import pickle

from torchvision.datasets import MNIST, EMNIST, KMNIST, FashionMNIST as FMNIST, CIFAR10
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize, Lambda
from torchvision.transforms.v2 import GaussianNoise, GaussianBlur, RandomAffine, RandomPerspective, RandomRotation
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, random_split


def get_dataset(dataset, seed, transform=None, power=0.0, output="1d"):

   if transform=="Noise":
      T = GaussianNoise(sigma=power)
   elif transform=="Affine":
      T = RandomAffine(0, (power/28, power/28), interpolation=InterpolationMode.BILINEAR)
   elif transform=="Perspective":
      T = RandomPerspective(distortion_scale=power, p=1.0, interpolation=InterpolationMode.BILINEAR)
   elif transform=="Rotation":
      T = RandomRotation(degrees=power, interpolation=InterpolationMode.BILINEAR)
   elif transform is None:
      T = None
   else:
      raise ValueError("Supported transforms are 'Affine', 'Noise', 'Perspective', 'Rotation'.")

   if output not in ["1d", "2d"]:
      raise ValueError("Supported outputs are '1d' or '2d'.")
   
   T_base = [ToImage(), ToDtype(torch.float32, scale=True)]
   T_flatten = Lambda(lambda x: torch.flatten(x))
   
   transform_train = Compose(T_base + ([T_flatten] if output=="1d" else []))
   transform_test = Compose(T_base + ([T] if T is not None else []) + ([T_flatten] if output=="1d" else []))

   transform_test_str = "" if transform is None else "_{}-{}".format(transform, power)

   datasets = {
      "MNIST": {
         "images_train": MNIST(root="./data/image", train=True, download=True, transform=transform_train),
         "images_test": MNIST(root="./data/image", train=False, download=True, transform=transform_test),
         "diagrams_dir_train": pickle.load(open("./data/MNIST_D_train_dir.pkl", "rb")),
         "diagrams_dir_test": pickle.load(open("./data/MNIST_D_test_dir{}.pkl".format(transform_test_str), "rb")),
         "dim": (28, 28),
         "n_classes": 10
      }
   }

   if dataset not in datasets.keys():
      raise ValueError("Supported datasets are '{}'.".format("', '".join(datasets.keys())))

   generator_images = torch.Generator().manual_seed(seed)
   generator_diagrams = torch.Generator().manual_seed(seed)

   images_train, images_val = random_split(datasets[dataset]["images_train"], [50000, 10000], generator_images)
   diagrams_train, diagrams_val = random_split(datasets[dataset]["diagrams_dir_train"], [50000, 10000], generator_diagrams)
   images_test = datasets[dataset]["images_test"]
   diagrams_test = datasets[dataset]["diagrams_dir_test"]

   dim = datasets[dataset]["dim"]
   n_classes = datasets[dataset]["n_classes"]

   return images_train, images_val, images_test, diagrams_train, diagrams_val, diagrams_test, dim, n_classes


def get_activation(str, alpha=0.0):
   activations = {
      "GELU": nn.GELU(),
      "ELU": nn.ELU(),
      "ReLU": nn.ReLU(),
      "LeakyReLU": nn.LeakyReLU(alpha),
      "CELU": nn.CELU(alpha)
   }
   
   if str not in activations.keys():
      raise ValueError("Supported activations are '{}'.".format("', '".join(activations.keys())))

   return activations[str]


def argmin(lst):
  return lst.index(min(lst))