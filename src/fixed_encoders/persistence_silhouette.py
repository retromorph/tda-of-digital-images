import torch
from gudhi.representations import Silhouette

from ._utils import diagram_tensor_to_pairs, make_weighting


class PersistenceSilhouetteEncoder:
    def __init__(self, resolution: int = 100, weighting: str = "none", weight_power: float = 1.0):
        self.resolution = resolution
        self.weighting = weighting
        self.weight_power = weight_power
        self._transformer = Silhouette(
            weight=make_weighting(weighting, weight_power),
            resolution=resolution,
        )

    def cache_config(self) -> dict:
        return {
            "resolution": self.resolution,
            "weighting": self.weighting,
            "weight_power": self.weight_power,
        }

    def __call__(self, diagram: torch.Tensor) -> torch.Tensor:
        pairs = diagram_tensor_to_pairs(diagram)
        if len(pairs) == 0:
            return torch.zeros((1, self.resolution), dtype=torch.float32)

        encoded = self._transformer.fit_transform([pairs])[0]
        return torch.from_numpy(encoded).float().unsqueeze(0)
