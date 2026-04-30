import torch

from src.filtrations.base import Diagram
from src.persistence import sublevel_persistence
from src.registry import FILTRATIONS


@FILTRATIONS("sublevel")
def sublevel(params=None):
    eps = None if params is None else params.get("eps")

    def apply(image):
        dgms = sublevel_persistence(image, eps=eps, pos=None, sort="persistence")
        if len(dgms) == 0:
            points = torch.zeros((0, 6), dtype=torch.float32)
        else:
            out = []
            for dgm in dgms:
                base = torch.zeros((len(dgm), 6), dtype=torch.float32)
                base[:, :3] = dgm[:, :3]
                base[:, 3] = 1.0
                base[:, 4] = 0.0
                base[:, 5] = 0.0
                out.append(base)
            points = torch.cat(out, dim=0)
        return Diagram(
            points=points,
            schema=["birth", "death", "dim", "sublevel", "direction_alpha", "direction_idx"],
        )

    return apply
