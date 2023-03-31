
"""some tools for the project"""

import torch
from torch import Tensor
import torch.nn.functional as F

def loss_func(output: Tensor , label: Tensor) -> Tensor:
    """loss function"""
    return torch.zeros(label.size(0), dtype=torch.float32)

def score_func(output: Tensor , label: Tensor) -> Tensor:
    """score function"""
    return torch.zeros(label.size(0), dtype=torch.float32)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count