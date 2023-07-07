
"""define a residual block"""

import torch
import torch.nn as nn
from torch import Tensor
import model.CNN.layer_blocks as lb


class ResNet(nn.Module):
    def __init__(self, in_channel, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.conv = lb.Conv3x3(in_channel, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = lb.ResidualBlock(16, 16)
        self.layer2 = lb.ResidualBlock(16, 32, 2)
        self.layer3 = lb.ResidualBlock(32, 64, 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    @torch.compile
    def forward(self, x:Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out
