
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

    def load_dict(self, data_dict: Dict[str,Any]):
        """load from dict"""
        for key, value in data_dict.items():
            setattr(self, key, value)

    def as_dict(self) -> Dict[str,Any]:
        """return as dict"""
        return vars(self)

    def load_yaml(self, yaml_path:str):
        """load from yaml file"""
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.Loader)
        self.load_dict(yaml_data)

    def save_yaml(self, yaml_path:str):
        """save to yaml file"""
        with open(yaml_path, 'x') as f:
            yaml.dump(self.as_dict(), f, default_flow_style=False)
