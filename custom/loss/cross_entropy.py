
import torch
from torch.nn import CrossEntropyLoss

class MyCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, weight=None, **kwargs):
        if isinstance(weight, list):
            weight = torch.FloatTensor(weight)

        super().__init__(weight=weight, **kwargs)