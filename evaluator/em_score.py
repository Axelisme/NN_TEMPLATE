
"""define a class to calculate em score"""

import numpy as np
from .ABC import Evaluator
import torch
from torch import Tensor

class EMScore(Evaluator):
    def __init__(self):
        super(EMScore, self).__init__()

    def eval(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred_:np.ndarray = y_pred.detach().cpu().numpy()
        y_true_:np.ndarray = y_true.detach().cpu().numpy()
        y_pred_ = (y_pred_ > 0.5).astype(int)
        return torch.as_tensor((y_pred_ == y_true_).all())
    