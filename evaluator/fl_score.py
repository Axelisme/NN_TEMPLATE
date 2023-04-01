
"""define a class to calculate f1 score"""

from .eval_abc import Evaluator
import torch
from torch import Tensor

class F1Score(Evaluator):
    def __init__(self,threshold=0.5):
        super(F1Score, self).__init__()
        self.threshold = threshold

    def eval(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return NotImplemented