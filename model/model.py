
"""A neural network model."""

import util.utility as ul
from config.config import Config
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, config:Config):
        """Initialize a neural network model."""
        super(Model, self).__init__()
        self.config = config
        channel1 = 256
        channel2 = 256

        self.flatten = nn.Flatten()
        self.Conv1 = nn.Sequential(
                                    nn.Linear(64, channel1),
                                    nn.BatchNorm1d(channel1),
                                    nn.ReLU()
                                  )
        self.Conv2 = nn.Sequential(
                                    nn.Linear(channel1, channel2),
                                    nn.BatchNorm1d(channel2),
                                    nn.ReLU()
                                  )
        self.Linear = nn.Linear(channel2, config.output_size)

    #@torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        x = self.flatten(x)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Linear(x)
        return x

