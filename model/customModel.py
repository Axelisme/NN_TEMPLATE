
"""A custom neural network model."""

import torch
from torch import nn
from torch import Tensor
from config.configClass import Config


class CustomModel(nn.Module):
    def __init__(self, conf:Config):
        """Initialize a neural network model."""
        super(CustomModel, self).__init__()
        self.conf = conf

        self.conv1 = nn.Conv2d(1, 1, 3, padding=1) # 80x80 -> 80x80
        self.flatten = nn.Flatten() # 80x80 -> 6400
        self.fn = nn.Linear(6400, 10) # 6400 -> 10

    @torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fn(x)
        return x

