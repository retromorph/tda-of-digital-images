import math
import torch
import numpy as np

from typing import Any, Dict, Sequence
from torchvision.transforms.v2 import Transform
from scipy.ndimage import rotate


def fibonacci_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` unit vectors on S^2 via the golden-ratio spiral.

    Returns ``(unit_vectors[n, 3], azimuth_deg[n])`` where ``azimuth_deg`` is
    the in-plane angle ``atan2(vy, vx)`` wrapped to ``[0, 360)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    k = np.arange(n, dtype=np.float64)
    # z evenly spaced in (-1, 1), inclusive at the poles for n=1.
    if n == 1:
        z = np.array([0.0])
    else:
        z = 1.0 - 2.0 * k / (n - 1)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phi = k * golden
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    unit = np.stack([x, y, z], axis=1).astype(np.float64)
    azimuth = np.degrees(np.arctan2(y, x)) % 360.0
    return unit, azimuth.astype(np.float64)


def _agg_fn(agg: str):
    if agg == "mult":
        return np.multiply
    if agg == "add":
        return np.add
    if agg == "min":
        return np.minimum
    if agg == "max":
        return np.maximum
    raise ValueError("Aggregation function must be one of 'mult', 'add', 'min', or 'max'.")


def direction_filter(img, alpha, agg="mult"):
    if len(img.shape) == 3:
        width, _height = img.shape[1], img.shape[2]
    elif len(img.shape) == 2:
        width, _height = img.shape[0], img.shape[1]
    else:
        raise ValueError()

    # Canvas must be wide enough that a rotated filter covers the full image.
    # sqrt(2) * width is the minimum diagonal needed; bump by 1 when the
    # excess is odd so padding is symmetric on both sides.
    padded = math.ceil(width * math.sqrt(2))
    if (padded - width) % 2 != 0:
        padded += 1
    pad = (padded - width) // 2

    img_out = np.zeros((padded, padded))
    img_out[pad:pad + width, pad:pad + width] = img
    filter_hor = np.repeat(np.linspace(1, 0, padded), padded).reshape(padded, padded).T
    filter_dir = rotate(filter_hor, alpha, reshape=False)[pad:pad + width, pad:pad + width]

    if agg == "mult":
        g = np.multiply
    elif agg == "add":
        g = np.add
    elif agg == "min":
        g = np.minimum
    elif agg == "max":
        g = np.maximum
    else:
        raise ValueError("Aggregation function must be one of 'mult', 'add', 'min', or 'max'.")

    return torch.Tensor(g(filter_dir, img))


class Direction(Transform):
    """Transform an image with the direction transform."""

    def __init__(self, alphas, agg="mult", add_sublevel=False):
        super().__init__()
        self.agg = agg
        self.alphas = alphas
        self.weight = alphas
        self.add_sublevel = add_sublevel

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        sub_dim = 1 if self.add_sublevel else 0
        output = torch.zeros(len(self.alphas) + sub_dim, inpt.shape[-2], inpt.shape[-1])
        for i, alpha in enumerate(self.alphas):
            output[i] = direction_filter(inpt, alpha, self.agg)
        if self.add_sublevel:
            output[-1] = inpt
        return output


def direction_filter_3d(img: torch.Tensor, unit_vec: Sequence[float], agg: str = "add") -> torch.Tensor:
    """3D analogue of ``direction_filter``: build a normalized linear height
    ``h(z,y,x) = vx*x + vy*y + vz*z`` over a unit cube and combine it with the
    volume via ``agg``.

    ``img`` is ``(D, H, W)`` or ``(1, D, H, W)``. Output is ``(D, H, W)``.
    """
    if img.dim() == 4:
        if img.shape[0] != 1:
            raise ValueError(f"direction_filter_3d expects single-channel volume, got shape {tuple(img.shape)}")
        vol = img[0]
    elif img.dim() == 3:
        vol = img
    else:
        raise ValueError(f"direction_filter_3d expects 3D/4D tensor, got shape {tuple(img.shape)}")

    vx, vy, vz = float(unit_vec[0]), float(unit_vec[1]), float(unit_vec[2])
    D, H, W = vol.shape
    zs = np.linspace(0.0, 1.0, D, dtype=np.float64).reshape(D, 1, 1)
    ys = np.linspace(0.0, 1.0, H, dtype=np.float64).reshape(1, H, 1)
    xs = np.linspace(0.0, 1.0, W, dtype=np.float64).reshape(1, 1, W)
    h = vx * xs + vy * ys + vz * zs
    # Normalize h to [0, 1] so the dynamic range matches the 2D ramp.
    h_min = min(0.0, vx) + min(0.0, vy) + min(0.0, vz)
    h_max = max(0.0, vx) + max(0.0, vy) + max(0.0, vz)
    span = h_max - h_min
    if span > 1e-12:
        h = (h - h_min) / span
    else:
        h = np.zeros_like(h)
    # 2D `direction_filter` uses a 1->0 ramp; mirror that here.
    h = 1.0 - h
    h_t = torch.as_tensor(np.broadcast_to(h, vol.shape).copy(), dtype=vol.dtype)
    g = _agg_fn(agg)
    return torch.as_tensor(g(h_t.numpy(), vol.detach().cpu().numpy()), dtype=vol.dtype)


class Direction3D(Transform):
    """Volumetric directional transform.

    Given a list of unit vectors on S^2, produce a ``(K, D, H, W)`` stack
    where each channel is the chosen ``agg`` of the volume with a linear
    height function aligned with that direction.
    """

    def __init__(self, unit_vectors, agg: str = "add", add_sublevel: bool = False):
        super().__init__()
        self.unit_vectors = np.asarray(unit_vectors, dtype=np.float64)
        if self.unit_vectors.ndim != 2 or self.unit_vectors.shape[1] != 3:
            raise ValueError(f"unit_vectors must have shape (K, 3), got {self.unit_vectors.shape}")
        self.agg = agg
        self.add_sublevel = add_sublevel

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if inpt.dim() == 4:
            if inpt.shape[0] != 1:
                raise ValueError(f"Direction3D expects single-channel volume, got shape {tuple(inpt.shape)}")
            vol = inpt[0]
        elif inpt.dim() == 3:
            vol = inpt
        else:
            raise ValueError(f"Direction3D expects 3D/4D tensor, got shape {tuple(inpt.shape)}")
        K = self.unit_vectors.shape[0]
        sub_dim = 1 if self.add_sublevel else 0
        output = torch.zeros(K + sub_dim, *vol.shape, dtype=vol.dtype)
        for i, v in enumerate(self.unit_vectors):
            output[i] = direction_filter_3d(vol, v, self.agg)
        if self.add_sublevel:
            output[-1] = vol
        return output
