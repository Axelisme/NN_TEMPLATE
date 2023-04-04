
"""define a class to store the hyperparameters"""

from torch import device
from typing import Dict, Any

class Config:
    """define a class to store the hyperparameters"""
    def __init__(self, **kwargs):
        # set default values
        self.data = kwargs

    def __getattr__(self, name):
        return self.data[name]
    
    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value