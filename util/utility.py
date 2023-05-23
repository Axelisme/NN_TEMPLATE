
"""some tools for the project"""

import numpy as np
import torch
import torch.nn as nn
import hyperparameter as p
from config.configClass import Config
from typing import List, Tuple, Dict, Any, Union, Optional

def set_seed(seed: int) -> None:
    """set seed for reproducibility"""
    import torch
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
    import os
    import wandb
    # create directory
    os.makedirs(p.SAVED_MODELS_DIR, exist_ok=True)
    # set float32 matmul precision
    torch.set_float32_matmul_precision('high')
    # set random seed
    set_seed(seed=config.seed)
    # initialize wandb
    if hasattr(config,"WandB") and config.WandB:
        wandb.init(project=config.project_name, name=config.model_name, config=config.data)

def show_result(config:Config, epoch:int, train_result:dict, valid_result:dict):
    """Print result of training and validation."""
    # print result
    #import os
    #os.system('clear')
    print(f'Epoch: ({epoch} / {config.epochs})')
    print("Train result:")
    print(f'\ttrain_loss: {train_result["train_loss"]:0.4f}')
    print("Valid result:")
    for name,score in valid_result.items():
        print(f'\t{name}: {score:0.4f}')

def log_result(epoch:int, train_result:dict, valid_result:dict):
    """log the result."""
    import wandb
    result = train_result.copy()
    result.update(valid_result)
    wandb.log(result)

def store_model(config:Config, model:nn.Module, save_path = None):
    """Store the model."""
    import wandb
    # save model
    if save_path is None:
        save_path = p.SAVED_MODELS_DIR + f"/{config.model_name}.pt"
    print(f'Saving model to {save_path}')
    torch.save(model.state_dict(), save_path)
    if hasattr(config,"WandB") and config.WandB:
        wandb.save(save_path)

def conv_output_size(input_size, kernel_size, stride = 1, padding = 0, dilation = 1) -> int:
    """calculate the output size of a convolutional layer"""
    return (input_size+2*padding-dilation*(kernel_size-1)-1)//stride+1

def pool_output_size(input_size, kernel_size, stride = None, padding = 0, dilation = 1) -> int:
    """calculate the output size of a pooling layer"""
    if stride is None:
        stride = kernel_size
    return conv_output_size(input_size, kernel_size, stride, padding, dilation)

def conv_transpose_output_size(input_size, kernel_size, stride = 1, padding = 0, dilation = 1, output_padding = 0) -> int:
    """calculate the output size of a convolutional transpose layer"""
    return int((input_size-1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1)

def clear_folder(path:str):
    """clear the folder"""
    import os
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def load_image(path:str, size:Tuple = None):
    """load image form given path and transform it to tensor
        input: path to the image, size of the image (H,W)
        output: a PIL image"""
    import PIL
    img = PIL.Image.open(path, mode='r').convert('RGB')
    if size is not None:
        img = img.resize(size[::-1])
    return img

def load_subfolder_as_label(root: str, loader = None, max_num = 1000):
    """load data from a folder, and use the subfolder name as label name
        input: path to the folder,
               a loader(path), default to return path,
               max number of data to load per label
        output: datas, label_names"""
    import os
    if loader is None:
        loader = lambda path: path
    datas = []
    label_names = []
    label_num = 0
    for dir, _, files in os.walk(root):
        if root == dir:
            continue
        # eg. root = 'data', dir = 'data/A/X/1', label_name = 'A_X_1'
        reldir = os.path.relpath(dir, root)
        label_names.append(reldir.replace(os.sep,'_'))
        for id, file in enumerate(files):
            if id >= max_num:
                break
            data = loader(os.path.join(dir, file))
            datas.append((data, label_num))
        label_num += 1
    return datas, label_names

class ShuffledIterable:
    """shuffle the iterable"""
    def __init__(self, iterable):
        import random
        self.iterable = iterable
        self.indices = list(range(len(iterable)))
        random.shuffle(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.iterable[original_index]

    def __len__(self):
        return len(self.iterable)


def shuffle(iterable):
    shuffled_iterable = ShuffledIterable(iterable)
    return shuffled_iterable

total_time = dict()
def measure_time(func):
    """measure the time of a function"""
    import time
    from functools import wraps
    global total_time
    total_time[func.__name__] = 0
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        total_time[func.__name__] += end-start
        return result
    return wrapper

def show_time():
    """show the time of each function"""
    global total_time
    print("Time:")
    for func_name, time in total_time.items():
        print(f'\t{func_name}: {time:0.4f}')

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
