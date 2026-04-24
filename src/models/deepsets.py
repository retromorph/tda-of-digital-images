import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(X, mask, dim=1, replace_mask=-torch.inf):
    X_mask = ~mask.unsqueeze(-1).repeat((1, 1, X.shape[-1]))
    return torch.masked.masked_tensor(X, X_mask).mean(dim).to_tensor(replace_mask)


class MaskedNorm(nn.Module):

    def __init__(self, d_feature, dim=1):
        """
        Args:
            feature_dim (int): Dimension of feature axis
            dim (int, tuple of ints): Aggregation axis(es)

        Memo:
            dim=(1,):  Feature norm
            dim=(2,):  Layer norm
            dim=(1,2): Set norm
        """
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.empty(d_feature, device=torch.device("cuda:0")))
        self.b = nn.Parameter(torch.empty(d_feature, device=torch.device("cuda:0")))
        torch.nn.init.constant_(self.W, 1.)
        torch.nn.init.constant_(self.b, 0.)

    def forward(self, X, mask, replace_mask=torch.inf):
        X_mask = ~mask.unsqueeze(-1).repeat((1, 1, 128)) # X.shape[-1]
        X_masked = torch.masked.masked_tensor(X, X_mask)
        X_mean = X_masked.mean(self.dim).to_tensor(replace_mask)
        X_std = X_masked.std(self.dim).to_tensor(replace_mask)

        if self.dim==1:
            X_mean.unsqueeze_(1)
            X_std.unsqueeze_(1)
        elif self.dim==2:
            X_mean.unsqueeze_(2)
            X_std.unsqueeze_(2)
        else:
            X_mean.unsqueeze_(1).unsqueeze_(2)
            X_std.unsqueeze_(1).unsqueeze_(2)

        X_centered_masked = X_masked - X_mean
        X_normed_masked = X_centered_masked / X_std
        X_out = F.linear(X_normed_masked.to_tensor(replace_mask), torch.diag_embed(self.W), self.b)
        # X_out = X_normed_masked.to_tensor(replace_mask)

        return X_out #, X_normed_masked, X_centered_masked
    

class LayerNormLinearBlock(nn.Module):

    def __init__(self, d_input, d_output, norm_dim=2, norm="layer", dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_input) # , norm_dim
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_input, d_output)

    def forward(self, X, mask):
        X = self.norm(X) # X, mask
        X = self.dropout(X)
        X = self.linear(X)
        return X
    

class BatchNormLinearBlock(nn.Module):

    def __init__(self, d_input, d_output, norm_dim=2, norm="layer", dropout=0.0):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_input) # , norm_dim
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_input, d_output)

    def forward(self, X, mask):
        X = self.norm(X) # X, mask
        X = self.dropout(X)
        X = self.linear(X)
        return X


class DeepSets(nn.Module):

    def __init__(self, d_input, d_output, d_model, d_hidden, norm="layer", dropout=0.0, activation="GELU", alpha=0.0, norm_first=False):
        super().__init__()
        self.alpha = alpha
        self.encoder1 = LayerNormLinearBlock(d_input, d_model, dropout=dropout, norm_dim=(1,2))
        self.encoder2 = LayerNormLinearBlock(d_model, d_hidden, dropout=dropout, norm_dim=(1,2))
        self.encoder3 = LayerNormLinearBlock(d_hidden, d_hidden, dropout=dropout, norm_dim=(1,2))
        self.encoder4 = LayerNormLinearBlock(d_hidden, d_model, dropout=dropout, norm_dim=(1,2))
        self.decoder1 = LayerNormLinearBlock(d_model, d_model, dropout=dropout, norm_dim=(1,2))
        self.decoder2 = LayerNormLinearBlock(d_model, d_output, dropout=dropout, norm_dim=(1,2))

    def _masked_mean(self, X, mask):
        X_masked = X * ~mask.unsqueeze(-1)
        n_masks = torch.sum(~mask, axis=1)
        X_masked_mean = torch.sum(X_masked, axis=1) / n_masks.unsqueeze(-1)
        return X_masked_mean
    
    def _masked_mean2(self, X, mask, dim=1, replace_mask=-torch.inf):
        X_mask = ~mask.unsqueeze(-1).repeat((1, 1, 128)) # X.shape[-1]
        return torch.masked.masked_tensor(X, X_mask).mean(dim).to_tensor(replace_mask)
    
    def _masked_max(self, X, mask):
        X_masked_max, _ = torch.max(X.masked_fill(mask.unsqueeze(-1), -torch.inf), axis=1)
        return X_masked_max

    def forward(self, X, mask):
        X = F.gelu(self.encoder1(X, mask))
        X = F.gelu(self.encoder2(X, mask))
        X = F.gelu(self.encoder3(X, mask))
        X = F.gelu(self.encoder4(X, mask))
        X = self._masked_mean(X, mask)
        X = F.gelu(self.decoder1(X, mask))
        X = self.decoder2(X, mask)
        return X