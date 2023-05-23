
"""define a class to store the hyperparameters"""

from typing import Dict, Any

class Config:
    """define a class to store the hyperparameters"""
    def __init__(self, **kwargs):
        # set default values
        self.data:Dict[str,Any] = kwargs

    def __getattr__(self, name):
        try:
            if name == 'data':
                return super().__getattr__(name) # type: ignore
            else:
                return self.data[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value

    def to_wandb(self) -> None:
        """add the config to wandb"""
        import wandb
        wandb.config.update(self.data)

    def update(self,config:"Config") -> None:
        """update the config"""
        self.data.update(config.data)