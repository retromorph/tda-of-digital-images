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

        self.layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(out_channels, out_channels*2, kernel_size=3, stride=2),
            ResidualBlock(out_channels*2, out_channels*4, kernel_size=3, stride=2),
            nn.AvgPool2d(7, 1), # wtf?
            nn.Flatten(),
            nn.Linear(out_channels*4, d_output)
        )

    def forward(self, x):
        return self.layers(x)