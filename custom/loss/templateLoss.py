
"""define a class to compute loss"""

import torch
from torch import nn
from torch import Tensor


class TemplateLoss(nn.Module):
    """define a class to compute loss"""
    def __init__(self) -> None:
        """initialize a loss instance"""
        super(TemplateLoss, self).__init__()

    def forward(self, output: Tensor, label: Tensor) -> Tensor:
        """forward function of loss"""
        return output.sum() - label.sum()
