"""Synthetic persistence diagrams for the sequence-length scaling benchmark (M1).

Produces tensors directly in the shape produced by ``PersistenceDataset`` +
``collate_fn``: ``(B, N, 9)`` where the per-point channels are
``[birth, death, hom_dim, angle, sin_5_features...]``. The mask is all-zeros
(no padding) so attention pays its full quadratic cost on the requested ``N``.
"""

from __future__ import annotations

import math

import torch


def random_diagram_batch(
    batch_size: int,
    n_points: int,
    d_in: int = 9,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(X, mask)`` mimicking persistence diagrams of fixed length.

    Channels follow the repo convention:
        ``[..., 0]``: birth ~ U(0, 1)
        ``[..., 1]``: death = birth + Exp(2.0) (always > birth)
        ``[..., 2]``: homology dim ~ Bernoulli(0.4) (0 or 1)
        ``[..., 3]``: angle ~ U(0, 2π)
        ``[..., 4:9]``: sin(angle / d) for d ∈ {40, 90, 180, 130, 360}

    All positions valid → ``mask`` is all-False of shape ``(B, N)``.
    """
    if d_in != 9:
        raise ValueError(f"Synthetic diagrams hard-coded for d_in=9; got {d_in}.")

    g = generator
    if g is None:
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
    # Build on CPU for deterministic generators (MPS doesn't support per-device generators well), then move.
    birth = torch.rand((batch_size, n_points), generator=g)
    death = birth + (-torch.log(1.0 - torch.rand((batch_size, n_points), generator=g) + 1e-9) / 2.0)
    hom_dim = (torch.rand((batch_size, n_points), generator=g) < 0.4).to(torch.float32)
    angle = torch.rand((batch_size, n_points), generator=g) * (2.0 * math.pi)
    sin_dens = (40.0, 90.0, 180.0, 130.0, 360.0)
    sin_feats = torch.stack(
        [torch.sin(angle * (torch.pi / d)) for d in sin_dens],
        dim=-1,
    )

    X = torch.cat(
        [
            birth.unsqueeze(-1),
            death.unsqueeze(-1),
            hom_dim.unsqueeze(-1),
            angle.unsqueeze(-1),
            sin_feats,
        ],
        dim=-1,
    ).to(dtype=dtype)

    mask = torch.zeros((batch_size, n_points), dtype=torch.bool)
    return X.to(device), mask.to(device)
