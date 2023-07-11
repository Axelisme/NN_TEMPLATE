
"""define some nn layer with """

from torch import nn
from torch import Tensor


# 1x1 convolution
class Conv1x1(nn.Module):
    """A 1x1 convolution."""
    def __init__(self, in_channel:int, out_channel:int, stride:int=1, padding:int=0):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x:Tensor) -> Tensor:
        return self.conv(x)


# 3x3 convolution
class Conv3x3(nn.Module):
    """A 3x3 convolution with padding."""
    def __init__(self, in_channel:int, out_channel:int, stride:int=1, padding:int=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x:Tensor) -> Tensor:
        return self.conv(x)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = nn.Sequential(
            Conv1x1(in_channel, out_channel, stride),
            nn.BatchNorm2d(out_channel),
        ) if (stride != 1) or (in_channel != out_channel) else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out