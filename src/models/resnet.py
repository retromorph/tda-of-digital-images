import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, stride),
            nn.ReLU(),
            ConvBlock(out_channels, out_channels, kernel_size, stride=1)
        )
        self.downsample = ConvBlock(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        z = self.layers(x)
        return F.relu(z + self.downsample(x))


class ResNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=64, d_output=10):
        super().__init__()
        c1, c2, c3 = out_channels, out_channels * 2, out_channels * 4

        self.layers = nn.Sequential(
            ConvBlock(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(c1, c2, kernel_size=3, stride=2),
            ResidualBlock(c2, c3, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, d_output),
        )

    def forward(self, x):
        return self.layers(x)