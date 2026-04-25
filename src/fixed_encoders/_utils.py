import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


def diagram_tensor_to_pairs(diagram: torch.Tensor) -> np.ndarray:
    """Convert padded diagram tensor into a clean Nx2 (birth, death) ndarray."""
    if diagram.ndim != 2 or diagram.shape[1] < 2:
        raise ValueError("Expected diagram tensor with shape [N, >=2].")

    pairs = diagram[:, :2]
    finite = torch.isfinite(pairs).all(dim=1)
    pairs = pairs[finite]
    if pairs.numel() == 0:
        return np.zeros((0, 2), dtype=np.float64)

    persistence = pairs[:, 1] - pairs[:, 0]
    pairs = pairs[persistence > 0]
    if pairs.numel() == 0:
        return np.zeros((0, 2), dtype=np.float64)

    return pairs.detach().cpu().numpy().astype(np.float64)


def _weight_none(point: np.ndarray) -> float:
    return 1.0


def _weight_linear(point: np.ndarray) -> float:
    return float(max(point[1] - point[0], 0.0))


@dataclass
class PowerWeight:
    power: float

    def __call__(self, point: np.ndarray) -> float:
        return float(max(point[1] - point[0], 0.0) ** self.power)


def make_weighting(weighting: str = "none", weight_power: float = 1.0) -> Callable[[np.ndarray], float]:
    if weighting == "none":
        return _weight_none
    if weighting == "linear":
        return _weight_linear
    if weighting == "power":
        return PowerWeight(weight_power)
    raise ValueError("Supported weighting modes are 'none', 'linear', and 'power'.")


def sigma2_to_bandwidth(sigma2: float) -> float:
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    return math.sqrt(sigma2)
