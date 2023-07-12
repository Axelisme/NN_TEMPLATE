"""
define a class for evaluating a model by loss
"""
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric
from typing import Optional


class LossScore(Metric):
    """define a class to calculate mean loss as score"""
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def __init__(self, criterion: Module):
        super(LossScore, self).__init__()
        self.criterion = criterion
        self.add_state("total_loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_num", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: Tensor, label: Tensor) -> None:
        """update the metric"""
        original_mode = self.criterion.training
        self.criterion.eval()
        with torch.no_grad():
            self.total_loss += self.criterion(output, label).item()
            self.total_num  += label.size(0)  # type: ignore
        self.criterion.train(original_mode)

    def compute(self) -> Tensor:
        """compute the metric"""
        return self.total_loss / self.total_num # type: ignore
