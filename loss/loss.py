
"""define a class to compute loss"""

import torch
from torch import nn
from torch import Tensor

class Loss(nn.Module):
    """define a class to compute loss"""
    def __init__(self) -> None:
        super(Loss, self).__init__()

    def forward(self, output: Tensor, label: Tensor) -> Tensor:
        """forward function"""
        return NotImplemented