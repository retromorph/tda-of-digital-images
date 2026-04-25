"""
LatentPersformer: Persformer variant with a Perceiver latent encoder.

Input tensor per point (from ``PersistenceTransformDataset`` / ``collate_fn``) has
``d_in=9`` channels with semantics:

- ``[..., 0]``: birth time
- ``[..., 1]``: death time
- ``[..., 2]``: homology dimension
- ``[..., 3]``: filtration / direction angle
- ``[..., 4:9]``: engineered positional encodings of the angle

Padding mask follows repo convention: ``True`` means padded (ignore).
"""

import torch
import torch.nn as nn
from transformers import PerceiverConfig, PerceiverModel


def _hf_hidden_act(name: str) -> str:
    mapping = {
        "gelu": "gelu",
        "relu": "relu",
        "silu": "silu",
        "swish": "silu",
    }
    key = str(name).strip().lower()
    return mapping.get(key, "gelu")


class LatentPersformer(nn.Module):
    """Perceiver-based encoder + latent pooling + MLP decoder."""

    def __init__(
        self,
        transform=None,
        d_in=9,
        d_out=2,
        d_model=128,
        d_latents=256,
        num_latents=128,
        num_blocks=1,
        num_self_attends_per_block=4,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        cross_attention_widening_factor=1,
        self_attention_widening_factor=1,
        dropout=0.1,
        decoder_hidden_dims=(256, 256, 64),
        decoder_dropout=0.2,
        activation="GELU",
    ):
        super().__init__()

        self.linear_in = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        cfg = PerceiverConfig(
            d_model=d_model,
            d_latents=d_latents,
            num_latents=num_latents,
            num_blocks=num_blocks,
            num_self_attends_per_block=num_self_attends_per_block,
            num_self_attention_heads=num_self_attention_heads,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_widening_factor=cross_attention_widening_factor,
            self_attention_widening_factor=self_attention_widening_factor,
            hidden_act=_hf_hidden_act(activation),
            attention_probs_dropout_prob=dropout,
        )
        self.encoder = PerceiverModel(cfg)

        dec_layers = []
        prev = d_latents
        for h in decoder_hidden_dims:
            dec_layers += [
                nn.LayerNorm(prev),
                nn.Linear(prev, h),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
            ]
            prev = h
        dec_layers += [nn.LayerNorm(prev), nn.Linear(prev, d_out)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, X, mask):
        """
        Args:
            X: ``(batch, seq, d_in)`` diagram features.
            mask: ``(batch, seq)`` bool with ``True`` on padded positions.
        """
        X = self.linear_in(X)
        attention_mask = (~mask).to(dtype=X.dtype)
        out = self.encoder(inputs=X, attention_mask=attention_mask)
        latents = out.last_hidden_state
        pooled = latents.mean(dim=1)
        return self.decoder(pooled)
