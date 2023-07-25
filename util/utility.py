
"""some tools for the project"""

import random
import numpy as np
import torch
from torch.backends import cudnn


def set_seed(seed: int, cudnn_benchmark = False) -> None:
    """set seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = cudnn_benchmark


def init(seed : int, start_method:str = 'forkserver') -> None:
    """Initialize the script."""
    # set float32 matmul precision
    torch.multiprocessing.set_start_method(start_method, force=True)
    torch.set_float32_matmul_precision('medium')
    # set random seed
    set_seed(seed=seed, cudnn_benchmark=True)


def get_cuda() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        raise RuntimeError('No cuda device available.')
