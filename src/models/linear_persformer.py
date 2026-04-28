"""
LinearPersformer: Persformer variant with a Nyströmformer encoder.

Input tensor per point (from ``PersistenceDataset`` / ``collate_fn``) has
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
from transformers import NystromformerConfig, NystromformerModel


def _hf_hidden_act(name: str) -> str:
    mapping = {
        "gelu": "gelu",
        "relu": "relu",
        "silu": "silu",
        "swish": "silu",
    }
    key = str(name).strip().lower()
    return mapping.get(key, "gelu")


class LinearPersformer(nn.Module):
    """Nyströmformer-based encoder + attention pooling + MLP decoder."""

    def __init__(
        self,
        transform=None,
        d_in=9,
        d_out=2,
        d_model=128,
        intermediate_size=512,
        num_hidden_layers=5,
        num_attention_heads=8,
        num_landmarks=64,
        encoder_dropout=0.1,
        decoder_hidden_dims=(256, 256, 64),
        decoder_dropout=0.2,
        pooling_heads=None,
        activation="GELU",
    ):
        super().__init__()
        if d_model % num_attention_heads != 0:
            raise ValueError("d_model must be divisible by num_attention_heads.")
        ph = num_attention_heads if pooling_heads is None else pooling_heads
        if d_model % ph != 0:
            raise ValueError("d_model must be divisible by pooling_heads (or num_attention_heads if pooling_heads is None).")

        self.linear_in = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        cfg = NystromformerConfig(
            vocab_size=2,  # effectively unused, inputs_embeds path only
            pad_token_id=0,
            hidden_size=d_model,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            num_landmarks=num_landmarks,
            hidden_dropout_prob=encoder_dropout,
            attention_probs_dropout_prob=encoder_dropout,
            hidden_act=_hf_hidden_act(activation),
            max_position_embeddings=8192,
        )
        self.encoder = NystromformerModel(cfg)
        self.num_landmarks = int(num_landmarks)

        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, ph, dropout=encoder_dropout, batch_first=True
        )

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
            X: ``(batch, seq, d_in)`` diagram features.
            mask: ``(batch, seq)`` bool with ``True`` on padded positions.
        """
        X = self.linear_in(X)
        B, S, D = X.shape

        # Nystromformer expects sequence length compatible with num_landmarks.
        target_len = max(S, 2 * self.num_landmarks)
        pad_len = (self.num_landmarks - (target_len % self.num_landmarks)) % self.num_landmarks
        target_len = target_len + pad_len
        pad_len = target_len - S
        if pad_len > 0:
            X = torch.nn.functional.pad(X, (0, 0, 0, pad_len), value=0.0)
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=True)

        attention_mask = (~mask).long()
        out = self.encoder(inputs_embeds=X, attention_mask=attention_mask)
        encoded = out.last_hidden_state[:, :S, :]
        mask = mask[:, :S]

        B = encoded.size(0)
        q = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, encoded, encoded, key_padding_mask=mask)
        pooled = pooled.squeeze(1)
        return self.decoder(pooled)
