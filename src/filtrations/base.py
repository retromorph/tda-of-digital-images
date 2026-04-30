from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class Diagram:
    points: torch.Tensor
    schema: list[str]


class Filtration(Protocol):
    def __call__(self, image: torch.Tensor) -> Diagram: ...
