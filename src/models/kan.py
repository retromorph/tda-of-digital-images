import torch.nn as nn
from ..kan.fastkan import FastKAN


class KAN(nn.Module):

    def __init__(self, d_input, d_output, d_hidden, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(d_input),
            nn.Dropout(dropout),
            FastKAN([d_input] + n_layers * [d_hidden] + [d_output])
        )

    def forward(self, x):
        return self.layers(x)