
"""some tools for the project"""

import os
import torch
from torch import nn
import wandb
import global_var.path as p
from config.configClass import Config
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

def init(config:Config):
    """Initialize the script."""
    # create directory
    os.makedirs(p.SAVED_MODELS_DIR, exist_ok=True)
    # set float32 matmul precision
    torch.set_float32_matmul_precision('high')
    # set random seed
    set_seed(seed=config.seed)
    # initialize wandb
    if config.WandB:
        wandb.init(project=config.project_name, name=config.model_name, config=config.data)

def show_result(config:Config, epoch:int, train_result:dict, valid_result:dict):
    """Print result of training and validation."""
    # print result
    #os.system('clear')
    print(f'Epoch: ({epoch} / {config.epochs})')
    print("Train result:")
    print(f'\ttrain_loss: {train_result["train_loss"]:0.4f}')
    print("Valid result:")
    for name,score in valid_result.items():
        print(f'\t{name}: {score:0.4f}')

def log_result(epoch:int, train_result:dict, valid_result:dict):
    """log the result."""
    result = train_result
    result.update(valid_result)
    wandb.log(result)

def store_model(config:Config, model:nn.Module):
    """Store the model."""
    # save model
    SAVE_MODEL_PATH = p.SAVED_MODELS_DIR + f"/{config.model_name}.pt"
    print(f'Saving model to {SAVE_MODEL_PATH}')
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    if config.WandB:
        wandb.save(SAVE_MODEL_PATH)

def conv2d_output_size(input_size: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """calculate the output size of a convolutional layer"""
    return int((input_size+2*padding-dilation*(kernel_size-1)-1)/stride+1)

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
