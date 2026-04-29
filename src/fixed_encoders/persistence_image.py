import numpy as np
import torch
from gudhi.representations import PersistenceImage

from ._utils import diagram_tensor_to_pairs, make_weighting, sigma2_to_bandwidth


class PersistenceImageEncoder:
    def __init__(self, resolution: int = 28, sigma2: float = 1.0, weighting: str = "none", weight_power: float = 1.0):
        self.resolution = resolution
        self.sigma2 = sigma2
        self.weighting = weighting
        self.weight_power = weight_power

        self._transformer = PersistenceImage(
            bandwidth=sigma2_to_bandwidth(sigma2),
            weight=make_weighting(weighting, weight_power),
            resolution=[resolution, resolution],
            im_range=[np.nan, np.nan, np.nan, np.nan],
        )

    def cache_config(self) -> dict:
        return {
            "resolution": self.resolution,
            "sigma2": self.sigma2,
            "weighting": self.weighting,
            "weight_power": self.weight_power,
        }

    def __call__(self, diagram: torch.Tensor) -> torch.Tensor:
        pairs = diagram_tensor_to_pairs(diagram)
        if len(pairs) == 0:
            return torch.zeros((1, self.resolution, self.resolution), dtype=torch.float32)

        encoded = self._transformer.fit_transform([pairs])[0]
        encoded = encoded.reshape(self.resolution, self.resolution)
        return torch.from_numpy(encoded).float().unsqueeze(0)
