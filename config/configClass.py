
"""define a class to store the hyperparameters"""

import yaml
from typing import Dict, Any, Optional


class Config:
    """define a class to store the hyperparameters."""
    def __init__(self, yaml_path:Optional[str] = None, data:Optional[Dict[str,Any]] = None):
        if yaml_path is not None:
            self.load_yaml(yaml_path)
        if data is not None:
            self.load_dict(data)

    def load_dict(self, data_dict:Dict[str,Any]) -> "Config":
        """load from dict"""
        for key, value in data_dict.items():
            self.__setattr__(key, value)
        return self

    def as_dict(self) -> Dict[str,Any]:
        """return as dict"""
        return self.__dict__

    def load_yaml(self, yaml_path:str) -> "Config":
        """load from yaml file"""
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.Loader)
        return self.load_dict(yaml_data)

    def load_config(self, config:"Config") -> "Config":
        """load from another config"""
        return self.load_dict(config.as_dict())

    def __str__(self) -> str:
        return str(self.as_dict())

    def __getattr__(self, __name: str) -> Any: # make pylint happy
        return super().__getattribute__(__name)

    def __setattr__(self, name: str, value: Any) -> None: # make pylint happy
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None: # make pylint happy
        super().__delattr__(name)
