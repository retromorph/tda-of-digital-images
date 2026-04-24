import torch
import torch.nn as nn

from transformers import PerceiverConfig, PerceiverModel


class PerceiverPHT(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 128,
        d_latents: int = 256,
        num_latents: int = 128,
        num_blocks: int = 1,
        num_self_attends_per_block: int = 6,
        num_self_attention_heads: int = 8,
        num_cross_attention_heads: int = 8,
        dropout: float = 0.0,
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
            attention_probs_dropout_prob=dropout,
        )
        self.perceiver = PerceiverModel(cfg)

        self.linear_out = nn.Sequential(
            nn.LayerNorm(d_latents),
            nn.Linear(d_latents, d_out),
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        X: (batch, seq_len, d_in) persistence-diagram points (padded)
        mask: (batch, seq_len) bool mask where True means padding (same convention as src_key_padding_mask)
        """
        X = self.linear_in(X)
        attention_mask = (~mask).to(dtype=X.dtype)
        out = self.perceiver(inputs=X, attention_mask=attention_mask, return_dict=True)
        latents = out.last_hidden_state  # (batch, num_latents, d_latents)
        pooled = latents.mean(dim=1)
        return self.linear_out(pooled)
