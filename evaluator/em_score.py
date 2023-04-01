
"""define a class to calculate em score"""

from .eval_abc import Evaluator
import torch
from torch import Tensor

class EMScore(Evaluator):
    def __init__(self):
        super(EMScore, self).__init__()

    def eval(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return NotImplemented
    