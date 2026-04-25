import torch.nn as nn

from src.utils import get_activation


class MLP(nn.Module):
    """MLP with batch norm; ``num_layers`` is the number of hidden width blocks (default 2 matches legacy)."""

    def __init__(
        self,
        d_input,
        d_output,
        d_hidden,
        dropout=0.0,
        num_layers=2,
        activation="ReLU",
        alpha=0.0,
    ):
        super().__init__()
        act = get_activation(activation, alpha)
        layers = [
            nn.BatchNorm1d(d_input),
            nn.Dropout(dropout),
            nn.Linear(d_input, d_hidden),
            act,
        ]
        for _ in range(max(num_layers - 1, 0)):
            layers += [
                nn.BatchNorm1d(d_hidden),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_hidden),
                act,
            ]
        layers += [
            nn.BatchNorm1d(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_output),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
