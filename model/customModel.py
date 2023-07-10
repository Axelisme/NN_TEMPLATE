
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

    #@torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        return NotImplemented

