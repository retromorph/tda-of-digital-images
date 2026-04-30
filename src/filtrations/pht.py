import numpy as np

from src.filtrations.base import Diagram
from src.persistence import pht
from src.persistence.transforms import Direction
from src.registry import FILTRATIONS


@FILTRATIONS("pht_directional")
def pht_directional(params=None):
    cfg = dict(params or {})
    alphas = cfg.get("alphas", list(np.linspace(0, 360, 16 + 1)[:-1]))
    agg = cfg.get("agg", "add")
    direction = Direction(alphas, agg=agg)

    def apply(image):
        points = pht(direction(image), image, pos=alphas)
        return Diagram(
            points=points,
            schema=["birth", "death", "dim", "sublevel", "direction_alpha", "direction_idx"],
        )

    return apply
