
"""define a class to calculate f1 score"""

import numpy as np
from .ABC import Evaluator
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from sklearn.metrics import f1_score
from typing import Literal

class F1Score(Evaluator):
    def __init__(self,threshold=0.5,type:Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'micro' ):
        super(F1Score, self).__init__()
        self.threshold = threshold
        self.type:Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = type

    def eval(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """calculate f1 score"""
        y_pred_:np.ndarray = y_pred.detach().cpu().numpy()
        y_pred_ = (y_pred_ > self.threshold).astype(int)
        y_true_ = one_hot(y_true.detach().cpu(), num_classes=y_pred_.shape[1]).numpy()
        return torch.as_tensor(f1_score(y_true_, y_pred_, average=self.type))