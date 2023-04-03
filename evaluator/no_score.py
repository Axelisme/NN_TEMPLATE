
"""return zero score for all the data"""

from .ABC import Evaluator
import torch
from torch import Tensor

class NoScore(Evaluator):
    """return zero score for all the data"""

    def eval(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return torch.zeros(1)