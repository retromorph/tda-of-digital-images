import torch
import numpy as np

from typing import Any, Dict
from torchvision.transforms.v2 import Transform
from scipy.ndimage import rotate

def direction_filter(img, alpha, agg="mult"):
    if len(img.shape)==3:
        width, height = img.shape[1], img.shape[2]
    elif len(img.shape)==2:
        width, _height = img.shape[0], img.shape[1]
    else:
        raise ValueError()
        
    if width==28:
        img_out = np.zeros((40, 40))
        img_out[6:34, 6:34] = img
        filter_hor = np.repeat(np.linspace(1, 0, 40), 40).reshape(40, 40).T
        filter_dir = (rotate(filter_hor, alpha, reshape=False))[6:34,:][:,6:34]
    elif width==32:
        img_out = np.zeros((46, 46))
        img_out[7:39, 7:39] = img
        filter_hor = np.repeat(np.linspace(1, 0, 46), 46).reshape(46, 46).T
        filter_dir = (rotate(filter_hor, alpha, reshape=False))[7:39,:][:,7:39]
    elif width==64:
        img_out = np.zeros((92, 92))
        img_out[14:78, 14:78] = img
        filter_hor = np.repeat(np.linspace(1, 0, 92), 92).reshape(92, 92).T
        filter_dir = (rotate(filter_hor, alpha, reshape=False))[14:78,:][:,14:78]
    else:
        raise ValueError("Only sizes of 28x28, 32x32, and 64x64 pixels are supported.")
        
    if agg=="mult":
        g = np.multiply
    elif agg=="add":
        g = np.add
    elif agg=="min":
        g = np.minimum
    elif agg=="max":
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
