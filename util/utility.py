
"""some tools for the project"""

import torch
from typing import List, Tuple, Dict, Any, Union, Optional

def set_seed(seed: int) -> None:
    """set seed for reproducibility"""
    from torch.backends import cudnn
    import random
    import numpy as np

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

class Result:
    """handle the training result"""
    def __init__(self, **results) -> None:
        """input: a dict of result, key is the name of the result, value is the result or a list of result"""
        self.sums: Dict[str,Any] = dict()
        self.counts: Dict[str,int] = dict()
        self.value: Dict[str,Any] = dict()
        self.log(**results)

    def log(self, **results) -> None:
        """add a result:
        input: a dict of result, key is the name of the result, value is the result or a list of result"""
        for key,value in results.items():
            if key not in self.value.keys():
                self.sums[key] = 0.
                self.counts[key] = 0
                self.value[key] = 0.
            if isinstance(value, list):
                self.sums[key] += sum(value)
                self.counts[key] += len(value)
                self.value[key] = value[-1]
            else:
                self.sums[key] += value
                self.counts[key] += 1
                self.value[key] = value

    def update(self, results: Dict[str,Any]) -> None:
        """update a result:
        input: a dict of result, key is the name of the result, value is the result or a list of result"""
        self.log(**results)

    def __getitem__(self, key) -> list:
        """get a result:
        input: the name of the result,
        output: a list of result"""
        return self.value[key]

    def average(self, key: str) -> Any:
        """get the average of a result:
        input: the name of the result,
        output: the average of the result"""
        return self.sums[key]/self.counts[key]

    def sum(self, key: str) -> Any:
        """get the sum of a result:
        input: the name of the result,
        output: the sum of the result"""
        return self.sums[key]

    def count(self, key: str) -> int:
        """get the count of a result:
        input: the name of the result,
        output: the count of the result"""
        return self.counts[key]
