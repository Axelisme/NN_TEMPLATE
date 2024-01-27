"""
define a class for evaluating a model by loss
"""

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric, MeanMetric


class LossScore(MeanMetric):
    """define a class to calculate mean loss as score"""

    def __init__(self, criterion: Module):
        super().__init__()
        self.criterion = criterion

    def update(self, output, *other) -> None:
        """update the metric"""
        original_mode = self.criterion.training
        self.criterion.eval()
        with torch.no_grad():
            loss = self.criterion(output, *other).item()
        super().update(loss)
        self.criterion.train(original_mode)
