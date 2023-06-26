
"""some tools for the project"""

import os
import sys
import time
import shutil
import random
import itertools
import numpy as np
from typing import *
from functools import wraps
from config.configClass import Config
import torch
from torch.nn import Module
from torch.optim import Optimizer

def set_seed(seed: int, cudnn_benchmark = False) -> None:
    """set seed for reproducibility"""
    from torch.backends import cudnn
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = cudnn_benchmark

def default_checkpoint(save_dir:str, model_name:str) -> str:
    """return the default path of the checkpoint"""
    return os.path.join(save_dir, model_name, f'checkpoint_{model_name}.pt')

def load_checkpoint(model:Module,
                    optim:Optimizer = None,
                    scheduler = None,
                    checkpoint:str = None,
                    device:torch.device = None) -> int:
    """Load checkpoint."""
    # load model
    if checkpoint is None:
        print('No checkpoint loaded.')
        return 0
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File {checkpoint} does not exist.")
    print(f'Loading checkpoint from {checkpoint}')
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint, map_location=device)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return epoch

def save_checkpoint(epoch:int,
                    model:Module,
                    optim:Optimizer = None,
                    scheduler = None,
                    checkpoint:str = None,
                    overwrite:bool = False) -> None:
    """Save the checkpoint."""
    if checkpoint is None:
        print('No checkpoint saved.')
        return
    if os.path.exists(checkpoint) and not overwrite:
        raise FileExistsError(f"File {checkpoint} already exists.")
    dir = os.path.dirname(checkpoint)
    os.makedirs(dir, exist_ok=True)
    print(f'Saving model to {checkpoint}')
    model_stat = model.state_dict()
    optim_stat = optim.state_dict() if optim is not None else None
    sched_stat = scheduler.state_dict() if scheduler is not None else None
    torch.save({'epoch': epoch,
                'model': model_stat,
                'optimizer': optim_stat,
                "scheduler": sched_stat},
                  checkpoint)

def get_cuda() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        raise RuntimeError('No cuda device available.')

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
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def load_subfolder_as_label(root: str, loader = None, max_num = 1000):
    """load data from a folder, and use the subfolder name as label name
        input: path to the folder,
               a loader(path), default to return path,
               max number of data to load per label
        output: datas, label_names"""
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
            file_path = os.path.join(dir, file)
            data = loader(file_path) if loader else file_path
            datas.append((data, label_num))
        label_num += 1
    return datas, label_names

def plot_confusion_matrix(cm, class_names, path = None, title='Confusion matrix', normalize=False):
    """plot the confusion matrix and save it to the given path if provided,
        input: confusion matrix, classes, path to save the figure, title of the figure
        output: None"""
    import matplotlib.pyplot as plt
    if normalize:
        cm = cm.astype('float') / np.nansum(cm, axis=1, keepdims=True)
        np.fill_diagonal(cm,np.nan)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]*100, fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()

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
    if len(total_time) == 0:
        return
    print("Time:")
    for func_name, time in total_time.items():
        print(f'\t{func_name}: {time:0.4f}s')

class LogO(object):
    """customized O for logging, it will print to the console and write to the log file"""
    def __init__(self, path: str):
        """input: path to the log file"""
        self.terminal = sys.stdout
        try:
            self.log = open(path, "a")
        except FileNotFoundError:
            raise FileNotFoundError(f'Log file {path} not found')

    def close(self):
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return sys.__stdout__.isatty()

    def fileno(self):
        return self.log.fileno()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log.close()

def write_to_log(content:str, path:str):
    """write the content to the log file"""
    with open(path, 'a') as f:
        f.write(content)

def logit(path:str = 'log.txt'):
    """log what the function print to the log file, and print it to the console"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import sys
            original_stdout = sys.stdout
            try:
                with LogO(path) as f:
                    sys.stdout = f
                    result = func(*args, **kwargs)
            finally:
                sys.stdout = original_stdout
            return result
        return wrapper
    return decorator
