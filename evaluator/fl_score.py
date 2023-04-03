
"""define a class to calculate f1 score"""

import numpy as np
from .ABC import Evaluator
import torch
from torch import Tensor
from sklearn.metrics import f1_score

class F1Score(Evaluator):
    def __init__(self,threshold=0.5):
        super(F1Score, self).__init__()
        self.threshold = threshold

    def eval(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred_:np.ndarray = y_pred.detach().cpu().numpy()
        y_true_:np.ndarray = y_true.detach().cpu().numpy()
        y_pred_ = (y_pred_ > self.threshold).astype(int)
        return torch.as_tensor(f1_score(y_true_, y_pred_, average='binary'))