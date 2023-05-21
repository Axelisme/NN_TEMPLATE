
"""A custom neural network model."""

import util.utility as ul
from config.configClass import Config
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class CostomModel(nn.Module):
    def __init__(self, config:Config):
        super(CostomModel, self).__init__()

    #@torch.compile
    def forward(self, x:Tensor) -> Tensor:
        out = x
        return out

