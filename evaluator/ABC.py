
"""define a ABC for evaluator"""

from abc import ABC, abstractmethod
from torch import Tensor

class Evaluator(ABC):
    """Abstract class for evaluator"""
    @abstractmethod
    def eval(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return NotImplemented