import math
from typing import Any, Dict

import numpy as np

import torch
import torch.nn as nn
from torchvision.transforms.v2 import Transform

from scipy.ndimage import grey_dilation, distance_transform_edt


def dilation_filter(img, size=3):
    return grey_dilation(img, size=size)

def sedt_filter(img, threshold=0.5): # TODO: separate Binarize and SEDT transforms?
    img /= img.max() # TODO: images are already normalized
    positive_edt = distance_transform_edt(img > threshold) # TODO: why threshold?
    negative_edt = distance_transform_edt(img <= threshold)
    sedt = positive_edt - negative_edt
    return torch.Tensor(sedt)

def direction_filter(img, alpha):
    if len(img.shape)==3:
        width, height = img.shape[1], img.shape[2]
    elif len(img.shape)==2:
        width, height = img.shape[0], img.shape[1]
    else:
        raise ValueError()

    X = (math.cos(alpha) - (np.arange(0, width) - (width / 2 - 0.5)) / (width * math.sqrt(2))) * math.cos(alpha) / 2
    Y = (math.sin(alpha) - (np.arange(0, height) - (height / 2 - 0.5)) / (height * math.sqrt(2))) * math.sin(alpha) / 2
    direction_filter = X.reshape(-1, 1) + Y.reshape(1, -1)
    return torch.Tensor(np.multiply(direction_filter, img))


class SoftThreshold(Transform):
    """SoftThreshold transform."""
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        x_ = torch.clone(inpt)
        x_[x_ < self.threshold] = 0
        return x_
    

class HardThreshold(Transform):
    """SoftThreshold transform."""
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        x_ = torch.clone(inpt)
        x_[x_ < self.threshold] = 0.
        x_[x_ >= self.threshold] = 1.
        return x_


class Id(Transform):
    """Identity transform."""
    def __init__(self):
        super().__init__()
        self.weight = None

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt
    
class Dilate(Transform):
    """Dilation transform."""
    def __init__(self, size=2):
        super().__init__()
        self.size = size
        self.weight = None

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return torch.Tensor(dilation_filter(inpt, self.size))

class SEDT(Transform):
    """Apply the signed Euclidean distance transform (SEDT) to a binary image."""
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.weight = threshold # TODO: None?

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return sedt_filter(inpt, self.threshold) 

class Direction(Transform):
    """Transform an image with the direction transform."""
    def __init__(self, alphas):
        super().__init__()
        self.alphas = alphas
        self.weight = alphas

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        output = torch.zeros(len(self.alphas), inpt.shape[-2], inpt.shape[-1])
        for i, alpha in enumerate(self.alphas):
            output[i] = direction_filter(inpt, alpha)
        return output

class Convolve2d(Transform):
    """Convolve a 2D image."""
    def __init__(self, dims, kernel_size=3, random_state=None):
        super().__init__()
        self.random_state = random_state

        self.filters = nn.Sequential(*[
            nn.Conv2d(dims[i], dims[i+1], kernel_size) for i in range(len(dims)-1)
        ])

        self.weight = []
        for filter in self.filters:
            self.weight.append(filter.weight)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.filters(inpt)