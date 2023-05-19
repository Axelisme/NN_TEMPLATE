
"""A neural network model."""

import util.utility as ul
from config.configClass import Config
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, config:Config):
        """Initialize a neural network model."""
        super(Model, self).__init__()
        self.config = config

    #@torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        return x

