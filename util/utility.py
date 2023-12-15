
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


def get_cuda() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        raise RuntimeError('No cuda device available.')


def check_better(conf, current_result, best_result):
    for name in conf['check_metrics']:
        if current_metric := current_result.get(name):
            if best_metric := best_result.get(name):
                if current_metric == best_metric:
                    continue
                return conf['save_mod'] == 'max' and current_metric > best_metric or \
                        conf['save_mod'] == 'min' and current_metric < best_metric
            else:
                return True
        else:
            continue
    return False


def create_instance(select_conf, *args, **kwargs):
    """create instance of class"""
    module = importlib.import_module(select_conf['module'])
    return getattr(module, select_conf['name'])(*args, **kwargs, **select_conf['kwargs'])
