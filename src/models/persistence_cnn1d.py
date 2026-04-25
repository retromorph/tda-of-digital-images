import torch.nn as nn


class PersistenceCNN1D(nn.Module):
    def __init__(self, d_output: int, in_channels: int = 1, base_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 2
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, d_output),
        )

    def forward(self, x):
        return self.head(self.features(x))
