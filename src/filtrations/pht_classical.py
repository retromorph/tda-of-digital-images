import numpy as np
import torch

from src.filtrations.base import Diagram
from src.persistence import sublevel_persistence
from src.persistence.transforms import fibonacci_sphere
from src.registry import FILTRATIONS


def _otsu_threshold(arr: np.ndarray) -> float:
    flat = arr.ravel().astype(np.float64)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0
    hist, edges = np.histogram(finite, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return float(centers.mean())
    p /= total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom = np.where(denom == 0, 1e-12, denom)
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.argmax(sigma_b2))])


def _binary_mask(img: torch.Tensor, mode: str, pore_phase: str) -> torch.Tensor:
    if mode == "none":
        return torch.ones(img.shape, dtype=torch.bool)
    if mode != "otsu":
        raise ValueError(f"binarize must be 'otsu' or 'none', got {mode!r}")
    arr = img.detach().cpu().numpy()
    thresh = _otsu_threshold(arr)
    if pore_phase == "dark":
        mask = arr < thresh
    elif pore_phase == "bright":
        mask = arr > thresh
    else:
        raise ValueError("pore_phase must be 'dark' or 'bright'.")
    return torch.from_numpy(mask)


def _height_function_stack_2d(mask: torch.Tensor, alphas_deg) -> torch.Tensor:
    H, W = mask.shape
    ys = torch.arange(H, dtype=torch.float32).unsqueeze(1) / max(H - 1, 1)
    xs = torch.arange(W, dtype=torch.float32).unsqueeze(0) / max(W - 1, 1)
    n = len(alphas_deg)
    stack = torch.empty((n, H, W), dtype=torch.float32)
    any_pore = bool(mask.any().item())
    for k, alpha_deg in enumerate(alphas_deg):
        a = float(alpha_deg) * float(np.pi) / 180.0
        h = xs * float(np.cos(a)) + ys * float(np.sin(a))
        if any_pore:
            sentinel = float(h[mask].max().item()) + 1.0
        else:
            sentinel = 1.0
        stack[k] = torch.where(mask, h, torch.full_like(h, sentinel))
    return stack


def _height_function_stack_3d(mask: torch.Tensor, unit_vectors: np.ndarray) -> torch.Tensor:
    D, H, W = mask.shape
    zs = torch.arange(D, dtype=torch.float32).view(D, 1, 1) / max(D - 1, 1)
    ys = torch.arange(H, dtype=torch.float32).view(1, H, 1) / max(H - 1, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, W) / max(W - 1, 1)
    K = unit_vectors.shape[0]
    stack = torch.empty((K, D, H, W), dtype=torch.float32)
    any_pore = bool(mask.any().item())
    for k in range(K):
        vx, vy, vz = float(unit_vectors[k, 0]), float(unit_vectors[k, 1]), float(unit_vectors[k, 2])
        h = vx * xs + vy * ys + vz * zs
        h = h.expand(D, H, W).contiguous()
        if any_pore:
            sentinel = float(h[mask].max().item()) + 1.0
        else:
            sentinel = 1.0
        stack[k] = torch.where(mask, h, torch.full_like(h, sentinel))
    return stack


@FILTRATIONS("pht_classical")
def pht_classical(params=None):
    cfg = dict(params or {})
    ndim = int(cfg.get("ndim", 2))
    binarize = cfg.get("binarize", "otsu")
    pore_phase = cfg.get("pore_phase", "dark")
    eps = cfg.get("eps")

    if ndim == 2:
        alphas = cfg.get("alphas")
        if alphas is None:
            alphas = list(np.linspace(0.0, 180.0, 16 + 1)[:-1])
        pos = list(alphas)

        def apply(image):
            if image.dim() == 2:
                img2d = image
            elif image.dim() == 3:
                img2d = image[0]
            else:
                raise ValueError(
                    f"pht_classical(ndim=2) expects (H, W) or (1, H, W), got {tuple(image.shape)}"
                )
            mask = _binary_mask(img2d, binarize, pore_phase)
            stack = _height_function_stack_2d(mask, alphas)
            dgms = sublevel_persistence(
                stack, eps=eps, pos=pos, sort="persistence", inf="remove"
            )
            if not dgms:
                points = torch.zeros((0, 6), dtype=torch.float32)
            else:
                points = torch.cat(dgms, dim=0).to(torch.float32)
            return Diagram(
                points=points,
                schema=["birth", "death", "dim", "sublevel", "direction_alpha", "direction_idx"],
            )

        return apply

    if ndim == 3:
        n_directions = int(cfg.get("n_directions", 16))
        unit_vecs, phi_deg = fibonacci_sphere(n_directions)
        pos = phi_deg.tolist()

        def apply(image):
            if image.dim() == 3:
                vol = image
            elif image.dim() == 4 and image.shape[0] == 1:
                vol = image[0]
            else:
                raise ValueError(
                    f"pht_classical(ndim=3) expects (D, H, W) or (1, D, H, W), got {tuple(image.shape)}"
                )
            mask = _binary_mask(vol, binarize, pore_phase)
            stack = _height_function_stack_3d(mask, unit_vecs)
            dgms = sublevel_persistence(
                stack, eps=eps, pos=pos, sort="persistence", inf="remove"
            )
            if not dgms:
                points = torch.zeros((0, 6), dtype=torch.float32)
            else:
                points = torch.cat(dgms, dim=0).to(torch.float32)
            return Diagram(
                points=points,
                schema=["birth", "death", "dim", "sublevel", "direction_alpha", "direction_idx"],
            )

        return apply

    raise ValueError(f"pht_classical: ndim must be 2 or 3, got {ndim}")
