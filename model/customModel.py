
"""A custom neural network model."""

import torch
from torch import nn
from torch import Tensor


class TemplateModel(nn.Module):
    def __init__(self, input_size, num_classes):
        """Initialize a neural network model."""
        super(TemplateModel, self).__init__()

        self.fn = nn.Linear(input_size, num_classes)

    @torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        x = self.fn(x)
        return x

