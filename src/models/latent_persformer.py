"""
LatentPersformer: Persformer variant with a Perceiver latent encoder.

Input tensor per point (from ``PersistenceDataset`` / ``collate_fn``) has
``d_in=9`` channels with semantics:

- ``[..., 0]``: birth time
- ``[..., 1]``: death time
- ``[..., 2]``: homology dimension
- ``[..., 3]``: filtration / direction angle
- ``[..., 4:9]``: engineered positional encodings of the angle

Padding mask follows repo convention: ``True`` means padded (ignore).

Pooling modes:
- ``"attn"`` (default): learnable query + MultiheadAttention over latent slots.
  Slots keep stable identity across samples, which is required for the latent
  probing and cross-attention interpretability experiments.
- ``"mean"``: mean over latents (legacy; kept for the pooling A/B ablation).
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import PerceiverConfig, PerceiverModel

from src.utils import _hf_hidden_act


class LatentPersformer(nn.Module):
    """Perceiver-based encoder + latent pooling + MLP decoder."""

    def __init__(
        self,
        d_in: int = 9,
        d_out: int = 2,
        d_model: int = 128,
        d_latents: int = 256,
        num_latents: int = 128,
        num_blocks: int = 1,
        num_self_attends_per_block: int = 4,
        num_self_attention_heads: int = 8,
        num_cross_attention_heads: int = 8,
        cross_attention_widening_factor: int = 1,
        self_attention_widening_factor: int = 1,
        dropout: float = 0.1,
        decoder_hidden_dims=(256, 256, 64),
        decoder_dropout: float = 0.2,
        activation: str = "GELU",
        pooling: str = "attn",
        pooling_heads: Optional[int] = None,
    ):
        super().__init__()
        if pooling not in ("mean", "attn"):
            raise ValueError(f"pooling must be 'mean' or 'attn'; got {pooling!r}")
        self.pooling = pooling

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

        if pooling == "attn":
            ph = num_self_attention_heads if pooling_heads is None else pooling_heads
            if d_latents % ph != 0:
                raise ValueError(
                    f"d_latents ({d_latents}) must be divisible by pooling_heads ({ph})."
                )
            self.pool_query = nn.Parameter(torch.zeros(1, 1, d_latents))
            nn.init.normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(
                d_latents, ph, dropout=dropout, batch_first=True
            )
        else:
            self.pool_query = None
            self.pool_attn = None

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
        dec_layers += [nn.Linear(prev, d_out)]
        self.decoder = nn.Sequential(*dec_layers)

        self._collect_aux: bool = False
        self._aux: dict = {}

    def enable_aux(self, flag: bool = True) -> None:
        """Toggle collection of latents and attention weights into ``self.aux``.

        Off by default to avoid memory overhead during training. Probing and
        visualization scripts call ``model.enable_aux(True)`` before forward.
        """
        self._collect_aux = flag
        if not flag:
            self._aux = {}

    @property
    def aux(self) -> dict:
        """Intermediate tensors captured during the last forward when aux is enabled.

        Keys (all detached):
            ``latents``: (B, num_latents, d_latents) — post-Perceiver, pre-pool.
            ``cross_attn``: tuple of (B, cross_heads, num_latents, seq) per block.
            ``self_attn``: tuple of (B, self_heads, num_latents, num_latents)
                per (block, self_attend) step.
        """
        return self._aux

    def forward(self, X, mask):
        """
        Args:
            X: ``(batch, seq, d_in)`` diagram features.
            mask: ``(batch, seq)`` bool with ``True`` on padded positions.
        """
        X = self.linear_in(X)
        attention_mask = (~mask).to(dtype=X.dtype)

        enc_kwargs = dict(
            inputs=X,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if self._collect_aux:
            enc_kwargs["output_attentions"] = True

        out = self.encoder(**enc_kwargs)
        latents = out.last_hidden_state

        if self.pooling == "mean":
            pooled = latents.mean(dim=1)
        else:
            B = latents.size(0)
            q = self.pool_query.expand(B, -1, -1)
            pooled, _ = self.pool_attn(q, latents, latents)
            pooled = pooled.squeeze(1)

        if self._collect_aux:
            cross_attn = getattr(out, "cross_attentions", None)
            self_attn = getattr(out, "attentions", None)
            self._aux = {
                "latents": latents.detach(),
                "cross_attn": tuple(a.detach() for a in cross_attn) if cross_attn is not None else None,
                "self_attn": tuple(a.detach() for a in self_attn) if self_attn is not None else None,
            }

        return self.decoder(pooled)
