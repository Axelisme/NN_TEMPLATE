
"""define a class to store the hyperparameters"""

import yaml
from typing import Dict, Any, Optional
from argparse import Namespace

class Config(Namespace):
    """define a class to store the hyperparameters."""
    def __init__(self, yaml_path:Optional[str] = None, data:Optional[Dict[str,Any]] = None):
        super().__init__()
        if yaml_path is not None:
            self.load_yaml(yaml_path)
        if data is not None:
            self.load_dict(data)

    def load_dict(self, data_dict: Dict[str,Any]) -> "Config":
        """load from dict"""
        for key, value in data_dict.items():
            setattr(self, key, value)
        return self

    def as_dict(self) -> Dict[str,Any]:
        """return as dict"""
        return vars(self)

    def load_yaml(self, yaml_path:str) -> "Config":
        """load from yaml file"""
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.Loader)
        return self.load_dict(yaml_data)

