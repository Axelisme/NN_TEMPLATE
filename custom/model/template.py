
"""A custom neural network model."""

import torch
from torch import nn
from torch import Tensor


class TemplateModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128):
        """Initialize a neural network model."""
        super(TemplateModel, self).__init__()

        self.fn1 = nn.Linear(input_size, hidden_size)
        self.fni = nn.ReLU()
        self.fn2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        x = self.fn1(x)
        x = self.fni(x)
        x = self.fn2(x)
        return x

