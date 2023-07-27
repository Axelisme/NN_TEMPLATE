"""
define a class to schedule learning rate
"""

from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduler(_LRScheduler):
    """define a class to schedule learning rate"""
    def __init__(self, optimizer, last_epoch=-1) -> None:
        """initialize a scheduler instance"""
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """get current learning rate"""
        return NotImplemented