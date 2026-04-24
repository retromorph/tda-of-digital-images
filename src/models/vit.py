import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


class ViT(nn.Module):

    def __init__(self, d_input, d_output, d_model, d_ff=None, n_heads=4, n_blocks=2, patch_size=7, n_channels=1, dropout=0.0):
        super().__init__()

        d_ff = d_model*2 if d_ff is None else d_ff
        self.model = ViTForImageClassification(ViTConfig(
            image_size=d_input,
            patch_size=patch_size,
            num_channels=n_channels,
            num_labels=d_output,
            hidden_size=d_model,
            intermediate_size=d_ff,
            num_hidden_layers=n_blocks,
            num_attention_heads=n_heads,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        ))

    def forward(self, X):
        return self.model(X).logits