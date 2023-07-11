
"""define a class to compute loss"""

import torch
from torch import nn
from torch import Tensor


class CustomLoss(nn.Module):
    """define a class to compute loss"""
    def __init__(self) -> None:
        """initialize a loss instance"""
        super(CustomLoss, self).__init__()

    @torch.compile()
    def forward(self, output: Tensor, label: Tensor) -> Tensor:
        """forward function of loss"""
        return NotImplemented
