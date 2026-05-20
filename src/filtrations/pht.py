import numpy as np

from src.filtrations.base import Diagram
from src.persistence import pht
from src.persistence.transforms import Direction, Direction3D, fibonacci_sphere
from src.registry import FILTRATIONS


def _normalize_2d(image):
    if image.dim() == 2:
        return image
    if image.dim() == 3 and image.shape[0] == 1:
        return image[0]
    raise ValueError(
        f"pht_directional(ndim=2) expects (H, W) or (1, H, W), got {tuple(image.shape)}"
    )


def _normalize_3d(image):
    if image.dim() == 3:
        return image
    if image.dim() == 4 and image.shape[0] == 1:
        return image[0]
    raise ValueError(
        f"pht_directional(ndim=3) expects (D, H, W) or (1, D, H, W), got {tuple(image.shape)}"
    )


@FILTRATIONS("pht_directional")
def pht_directional(params=None):
    cfg = dict(params or {})
    ndim = int(cfg.get("ndim", 2))
    agg = cfg.get("agg", "add")
    eps = cfg.get("eps")

    if ndim == 2:
        alphas = cfg.get("alphas", list(np.linspace(0, 360, 16 + 1)[:-1]))
        direction = Direction(alphas, agg=agg)
        pos = list(alphas)

        def apply(image):
            img = _normalize_2d(image)
            stack = direction(img)
            base = img.unsqueeze(0)
            points = pht(stack, base, pos=pos, eps=eps)
            return Diagram(
                points=points,
                schema=["birth", "death", "dim", "sublevel", "direction_alpha", "direction_idx"],
            )

        return apply

    if ndim == 3:
        n_directions = int(cfg.get("n_directions", 16))
        unit_vecs, phi_deg = fibonacci_sphere(n_directions)
        direction = Direction3D(unit_vecs, agg=agg)
        pos = phi_deg.tolist()

        def apply(image):
            vol = _normalize_3d(image)
            stack = direction(vol)
            base = vol.unsqueeze(0)
            points = pht(stack, base, pos=pos, eps=eps)
            return Diagram(
                points=points,
                schema=["birth", "death", "dim", "sublevel", "direction_alpha", "direction_idx"],
            )

        return apply

    raise ValueError(f"pht_directional: ndim must be 2 or 3, got {ndim}")
