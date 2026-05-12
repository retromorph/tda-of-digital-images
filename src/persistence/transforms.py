import math
import torch
import numpy as np

from typing import Any, Dict
from torchvision.transforms.v2 import Transform
from scipy.ndimage import rotate


def direction_filter(img, alpha, agg="mult"):
    if len(img.shape) == 3:
        width, _height = img.shape[1], img.shape[2]
    elif len(img.shape) == 2:
        width, _height = img.shape[0], img.shape[1]
    else:
        raise ValueError()

    # Canvas must be wide enough that a rotated filter covers the full image.
    # sqrt(2) * width is the minimum diagonal needed; bump by 1 when the
    # excess is odd so padding is symmetric on both sides.
    padded = math.ceil(width * math.sqrt(2))
    if (padded - width) % 2 != 0:
        padded += 1
    pad = (padded - width) // 2

    img_out = np.zeros((padded, padded))
    img_out[pad:pad + width, pad:pad + width] = img
    filter_hor = np.repeat(np.linspace(1, 0, padded), padded).reshape(padded, padded).T
    filter_dir = rotate(filter_hor, alpha, reshape=False)[pad:pad + width, pad:pad + width]

    if agg == "mult":
        g = np.multiply
    elif agg == "add":
        g = np.add
    elif agg == "min":
        g = np.minimum
    elif agg == "max":
        g = np.maximum
    else:
        raise ValueError("Aggregation function must be one of 'mult', 'add', 'min', or 'max'.")

    return torch.Tensor(g(filter_dir, img))


class Direction(Transform):
    """Transform an image with the direction transform."""

    def __init__(self, alphas, agg="mult", add_sublevel=False):
        super().__init__()
        self.agg = agg
        self.alphas = alphas
        self.weight = alphas
        self.add_sublevel = add_sublevel

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        sub_dim = 1 if self.add_sublevel else 0
        output = torch.zeros(len(self.alphas) + sub_dim, inpt.shape[-2], inpt.shape[-1])
        for i, alpha in enumerate(self.alphas):
            output[i] = direction_filter(inpt, alpha, self.agg)
        if self.add_sublevel:
            output[-1] = inpt
        return output
