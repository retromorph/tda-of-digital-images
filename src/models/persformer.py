"""
Persformer-style encoder for persistence diagrams.

Input tensor per point (from ``PersistenceTransformDataset`` / ``collate_fn``) has
``d_in=9`` channels with semantics:

- ``[..., 0]``: birth time
- ``[..., 1]``: death time
- ``[..., 2]``: homology dimension (numeric label from the diagram)
- ``[..., 3]``: filtration / direction angle (when using directional PHT)
- ``[..., 4:9]``: engineered positional encodings of the angle (sines at fixed scales)

Padding positions are masked out by ``src_key_padding_mask`` (``True`` = ignore).
"""

import torch
import torch.nn as nn

from src.utils import get_activation


class Persformer(nn.Module):
    """Transformer encoder + multi-head attention pooling + MLP decoder (Persformer-style).

    Expects ``d_in=9`` diagram tokens per ``forward`` docstring / module header (birth, death,
    homology dim, filtration angle, then engineered angle channels).
    """

    def __init__(
        self,
        transform=None,
        d_in=9,
        d_out=2,
        d_model=128,
        d_hidden=512,
        num_heads=8,
        num_layers=5,
        norm="layer",
        encoder_dropout=0.0,
        decoder_hidden_dims=(256, 256, 64),
        decoder_dropout=0.2,
        activation="GELU",
        alpha=0.0,
        norm_first=True,
        pooling_heads=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        ph = num_heads if pooling_heads is None else pooling_heads
        if d_model % ph != 0:
            raise ValueError("d_model must be divisible by pooling_heads (or num_heads if pooling_heads is None).")

        self.linear_in = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        f = get_activation(activation, alpha)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            d_hidden,
            encoder_dropout,
            activation=f,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.post_encoder_norm = None
        if norm == "layer":
            self.post_encoder_norm = nn.LayerNorm(d_model)
        elif norm == "batch":
            self.post_encoder_norm = nn.BatchNorm1d(d_model)

        # Multi-head attention pooling (single learnable query; Set Transformer / Persformer-style)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, ph, dropout=encoder_dropout, batch_first=True
        )

        # MLP decoder: LayerNorm + Linear + GELU + Dropout blocks, then logits
        dec_layers = []
        prev = d_model
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
            X: ``(batch, seq, d_in)`` diagram features (see module docstring for ``d_in=9`` layout).
            mask: ``(batch, seq)`` bool, ``True`` on padded positions (ignored in encoder and pooling).
        """
        X = self.linear_in(X)
        X = self.encoder(X, src_key_padding_mask=mask)

        if self.post_encoder_norm is not None:
            if isinstance(self.post_encoder_norm, nn.BatchNorm1d):
                X = self.post_encoder_norm(X.transpose(1, 2)).transpose(1, 2)
            else:
                X = self.post_encoder_norm(X)

        B = X.size(0)
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, X, X, key_padding_mask=mask)
        pooled = pooled.squeeze(1)

        return self.decoder(pooled)
