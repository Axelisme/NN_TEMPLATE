
"""some tools for the project"""

import random
import numpy as np
import importlib

import torch
from torch.backends import cudnn


def set_seed(seed: int) -> int:
    """set seed for reproducibility"""
    old_seed = torch.initial_seed()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    return old_seed


def init(seed : int, start_method:str = 'forkserver') -> None:
    """Initialize the script."""
    # set float32 matmul precision
    torch.multiprocessing.set_start_method(start_method, force=True)
    torch.set_float32_matmul_precision('medium')
    # set random seed
    set_seed(seed=seed)
    cudnn.benchmark = True


def create_instance(select_conf, *args, **kwargs):
    """create instance of class"""
    module = importlib.import_module(select_conf['module'])
    return getattr(module, select_conf['name'])(*args, **kwargs, **select_conf['kwargs'])
