
"""define a class to store the hyperparameters"""

from torch import device
from typing import Dict, Any

class Config(dict):
    """define a class to store the hyperparameters"""
    def __init__(self, **kwargs):
        # set default values
        self.data = kwargs

    def __getattr__(self, name):
        if name == 'data':
            return super().__getattr__(name) # type: ignore[attr-defined]
        else:
            return self.data[name]
    
    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value

    def to_wandb(self) -> None:
        """add the config to wandb"""
        import wandb
        wandb.config.update(self.data)