import torch
import torch.nn as nn

from src.utils import get_activation

class PersistentHomologyTransformer(nn.Module):
    
    def __init__(self, transform=None, d_in=4, d_out=2, d_model=16, d_hidden=32, num_heads=4, num_layers=2, agg="mean", norm=None, dropout=0.0, activation="GELU", alpha=0.0, norm_first=False):
        super().__init__()
        self.agg = agg
        self.norm = norm
        self.linear_in = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        f = get_activation(activation, alpha)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_hidden, dropout, activation=f, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        if self.norm=="layer":
            self.norm = nn.LayerNorm(d_model)
        elif self.norm=="batch":
            self.norm = nn.BatchNorm1d(d_model)
        self.linear_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_out),
        )
        
    def _masked_mean(self, X, mask):
        X_masked = X * ~mask.unsqueeze(-1)
        n_masks = torch.sum(~mask, axis=1)
        X_masked_mean = torch.sum(X_masked, axis=1) / n_masks.unsqueeze(-1) # torch.mean(X, axis=1)
        return X_masked_mean
    
    def _masked_max(self, X, mask):
        X_masked_max, _ = torch.max(X.masked_fill(mask.unsqueeze(-1), -torch.inf), axis=1)
        return X_masked_max
        
    def forward(self, X, mask):
        X = self.linear_in(X)
        X = self.encoder(X, src_key_padding_mask=mask)
        if self.norm is not None:
            X = self.norm(X)
        if self.agg=="mean":
            X = self._masked_mean(X, mask)
        elif self.agg=="max":
            X = self._masked_max(X, mask)
        X = self.linear_out(X)
        return X