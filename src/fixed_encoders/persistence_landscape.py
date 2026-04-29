import torch
from gudhi.representations import Landscape

from ._utils import diagram_tensor_to_pairs


class PersistenceLandscapeEncoder:
    def __init__(self, num_layers: int = 5, resolution: int = 100):
        self.num_layers = num_layers
        self.resolution = resolution
        self._transformer = Landscape(num_landscapes=num_layers, resolution=resolution)

    def cache_config(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "resolution": self.resolution,
        }

    def __call__(self, diagram: torch.Tensor) -> torch.Tensor:
        pairs = diagram_tensor_to_pairs(diagram)
        if len(pairs) == 0:
            return torch.zeros((self.num_layers, self.resolution), dtype=torch.float32)

        encoded = self._transformer.fit_transform([pairs])[0]
        encoded = encoded.reshape(self.num_layers, self.resolution)
        return torch.from_numpy(encoded).float()
