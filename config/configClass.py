
"""define a class to store the hyperparameters"""

from typing import Dict, Any


class Config:
    """define a class to store the hyperparameters.
    I modified the __getattr__ and __setattr__ method to make it more convenient to use.
    But it is not a good idea to use it to store the hyperparameters.
    Maybe some time later I will change it to just a dictionary."""
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

    def __delattr__(self, name: str) -> None:
        del self.data[name]

    def update(self,config:"Config") -> None:
        """update the config"""
        self.data.update(config.data)