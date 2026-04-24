import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, d_input, d_output, d_hidden, dropout=0.0, norm_first=True, activation=nn.ReLU):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(d_input),
            nn.Dropout(dropout),
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_output),
        )

    def forward(self, x):
        return self.layers(x)