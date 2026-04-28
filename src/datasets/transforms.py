import torch

from torchvision.transforms import InterpolationMode as IM
from torchvision.transforms.v2 import (
    Compose,
    GaussianBlur,
    GaussianNoise,
    Lambda,
    RandomAffine,
    RandomPerspective,
    RandomRotation,
    ToDtype,
)


def get_transform(transform_str, power=0.0):
    transforms = {
        "affine": lambda p: RandomAffine(0, (p / 28, p / 28), interpolation=IM.BILINEAR),
        "noise": lambda p: GaussianNoise(sigma=p),
        "blur": lambda p: GaussianBlur(kernel_size=5, sigma=(float(p), float(p))),
        "perspective": lambda p: RandomPerspective(distortion_scale=p, p=1.0, interpolation=IM.BILINEAR),
        "rotation": lambda p: RandomRotation(degrees=p, interpolation=IM.BILINEAR),
    }

    if transform_str not in transforms:
        raise ValueError("Supported transforms are '{}'.".format("', '".join(transforms.keys())))
    return transforms[transform_str](power)


def build_image_transforms(transform_str=None, power=0.0, output="2d"):
    transform_base = [Lambda(lambda x: x / 255), ToDtype(torch.float32)]
    transform_invariance = [get_transform(transform_str, power)] if transform_str is not None else []
    transform_flatten = [Lambda(lambda x: torch.flatten(x, 1))] if output == "1d" else []

    transform_train_val = Compose(transform_base + transform_flatten)
    transform_test = Compose(transform_base + transform_invariance + transform_flatten)
    return transform_train_val, transform_test
