import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_activation


class LayerNormLinearBlock(nn.Module):

    def __init__(self, d_input, d_output, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_input, d_output)

    def forward(self, X, mask):
        X = self.norm(X)
        X = self.dropout(X)
        X = self.linear(X)
        return X


class DeepSets(nn.Module):

    def __init__(self, d_input, d_output, d_model, d_hidden, dropout=0.0, activation="GELU", alpha=0.0):
        super().__init__()
        self.act = get_activation(activation, alpha)
        self.encoder1 = LayerNormLinearBlock(d_input, d_model, dropout=dropout)
        self.encoder2 = LayerNormLinearBlock(d_model, d_hidden, dropout=dropout)
        self.encoder3 = LayerNormLinearBlock(d_hidden, d_hidden, dropout=dropout)
        self.encoder4 = LayerNormLinearBlock(d_hidden, d_model, dropout=dropout)
        self.decoder1 = LayerNormLinearBlock(d_model, d_model, dropout=dropout)
        self.decoder2 = LayerNormLinearBlock(d_model, d_output, dropout=dropout)

    def _masked_mean(self, X, mask):
        X_masked = X * ~mask.unsqueeze(-1)
        n_masks = torch.sum(~mask, axis=1)
        X_masked_mean = torch.sum(X_masked, axis=1) / n_masks.unsqueeze(-1)
        return X_masked_mean

    def forward(self, X, mask):
        X = self.act(self.encoder1(X, mask))
        X = self.act(self.encoder2(X, mask))
        X = self.act(self.encoder3(X, mask))
        X = self.act(self.encoder4(X, mask))
        X = self._masked_mean(X, mask)
        X = self.act(self.decoder1(X, mask))
        X = self.decoder2(X, mask)
        return X
