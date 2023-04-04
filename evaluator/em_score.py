
"""define a class to calculate em score"""

import numpy as np
from .ABC import Evaluator
import torch
from torch import Tensor
from torch.nn.functional import one_hot

class EMScore(Evaluator):
    """define a class to calculate exact match score"""
    def __init__(self):
        super(EMScore, self).__init__()

    def eval(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """calculate em score"""
        y_pred_:np.ndarray = y_pred.detach().cpu().numpy()
        y_pred_ = (y_pred_ > 0.5).astype(int)
        y_true_:np.ndarray = one_hot(y_true.detach(), num_classes=y_pred_.shape[1]).detach().cpu().numpy()
        return torch.as_tensor((y_pred_ == y_true_).all(1).sum() / y_true_.shape[0])
    