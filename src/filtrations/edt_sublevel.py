import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from src.filtrations.base import Diagram
from src.filtrations.pht_classical import _binary_mask
from src.persistence import sublevel_persistence
from src.registry import FILTRATIONS


@FILTRATIONS("edt_sublevel")
def edt_sublevel(params=None):
    """Sublevel filtration of the negative Euclidean distance transform on the
    binarized pore phase (Moon 2019 style for porous-media permeability).

    Filtration value per pixel:
      pore cells   -> -EDT(x) (negative distance to nearest matrix boundary)
      matrix cells -> 0

    Sublevel sets sweep pore centers first (deepest pore bodies), grow toward
    pore boundaries, then matrix appears at level 0. H0 features = connected
    pore bodies merging through throats; H1 features = loops in the pore
    network filled when throats close.
    """
    cfg = dict(params or {})
    binarize = cfg.get("binarize", "otsu")
    pore_phase = cfg.get("pore_phase", "dark")
    eps = cfg.get("eps")
    normalize = cfg.get("normalize", False)
    phase = cfg.get("phase", "pore")  # "pore" | "matrix" | "both"
    ndim = int(cfg.get("ndim", 2))

    if phase not in ("pore", "matrix", "both"):
        raise ValueError(f"phase must be 'pore', 'matrix', or 'both', got {phase!r}")
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    def _channel(mask_np: np.ndarray, normalize_value: float) -> np.ndarray:
        edt = distance_transform_edt(mask_np).astype(np.float32)
        if normalize and normalize_value > 0:
            edt = edt / normalize_value
        # Matrix cells have edt=0; -edt = 0. Pore cells have edt>0; -edt < 0.
        # Sublevel sweeps from deepest pore (-edt very negative) to matrix at 0.
        return -edt

    def apply(image):
        if ndim == 2:
            if image.dim() == 2:
                img = image
            elif image.dim() == 3 and image.shape[0] == 1:
                img = image[0]
            else:
                raise ValueError(
                    f"edt_sublevel(ndim=2) expects (H, W) or (1, H, W), got {tuple(image.shape)}"
                )
        else:
            if image.dim() == 3:
                img = image
            elif image.dim() == 4 and image.shape[0] == 1:
                img = image[0]
            else:
                raise ValueError(
                    f"edt_sublevel(ndim=3) expects (D, H, W) or (1, D, H, W), got {tuple(image.shape)}"
                )
        mask = _binary_mask(img, binarize, pore_phase).numpy()
        diag_px = float(np.sqrt(sum(s * s for s in mask.shape)))

        channels = []
        if phase in ("pore", "both"):
            channels.append(_channel(mask, diag_px))
        if phase in ("matrix", "both"):
            channels.append(_channel(~mask, diag_px))

        stack = torch.from_numpy(np.stack(channels, axis=0))
        dgms = sublevel_persistence(
            stack, eps=eps, pos=None, sort="persistence", inf="remove"
        )
        if not dgms:
            points = torch.zeros((0, 6), dtype=torch.float32)
        else:
            out = []
            for k, dgm in enumerate(dgms):
                n = len(dgm)
                base = torch.zeros((n, 6), dtype=torch.float32)
                base[:, :3] = dgm[:, :3].to(torch.float32)
                base[:, 3] = 1.0  # marks this filtration family (vs PHT directional)
                base[:, 4] = 0.0  # no angle
                base[:, 5] = float(k)  # 0 = pore channel, 1 = matrix channel (if phase=both)
                out.append(base)
            points = torch.cat(out, dim=0)

        return Diagram(
            points=points,
            schema=["birth", "death", "dim", "sublevel", "direction_alpha", "direction_idx"],
        )

    return apply
