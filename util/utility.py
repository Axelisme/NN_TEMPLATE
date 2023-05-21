
"""some tools for the project"""

import torch
import torch.nn as nn
import hyperparameter as p
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
    import os
    import wandb

    # create directory
    os.makedirs(p.SAVED_MODELS_DIR, exist_ok=True)

    # set float32 matmul precision
    torch.set_float32_matmul_precision('high')

    # set random seed
    if hasattr(config,"seed"):
        set_seed(seed=config.seed)
    else:
        set_seed(seed=0)

    # initialize wandb
    if hasattr(config,"WandB") and config.WandB:
        wandb.init(project=config.project_name, name=config.model_name, config=config.data)

def show_result(config:Config, epoch:int, train_result:dict, valid_result:dict):
    """Print result of training and validation."""
    # print result
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
    # set save path
    if save_path is None:
        if hasattr(config,"model_name"):
            save_path = p.SAVED_MODELS_DIR + f"/{config.model_name}.pt"
        else:
            save_path = p.SAVED_MODELS_DIR + f"/model.pt"
    # save model
    print(f'Saving model to {save_path}')
    torch.save(model.state_dict(), save_path)
    if hasattr(config,"WandB") and config.WandB:
        wandb.save(save_path)

def conv_output_size(input_size, kernel_size, stride = 1, padding = 0, dilation = 1) -> int:
    """calculate the output size of a convolutional layer"""
    return int((input_size+2*padding-dilation*(kernel_size-1)-1)/stride+1)

def load_image(path:str, size = None) -> torch.Tensor:
    """load image form given path and transform it to tensor"""
    import torchvision.transforms as transforms
    import cv2
    img = cv2.imread(path)
    if size is not None:
        img = cv2.resize(img, size[1:])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(img)
    return img

def load_subfolder_as_label(root: str, loader = None, max_num = 1000) -> Tuple[List,List[int],List[str]]:
    """load data from a folder, and use the subfolder name as label name
        input: path to the folder
        output: a tuple of (data, data_labels, label_names)"""
    import os
    datas = []
    labels = []
    label_names = []
    for label_id, dir in enumerate(os.listdir(root)):
        label_names.append(dir)
        count = 0
        for file in os.listdir(os.path.join(root,dir)):
            if count >= max_num:
                break
            file_path = os.path.join(dir,file)
            if loader is None:
                datas.append(file_path)
            else:
                datas.append(loader(os.path.join(root,file_path)))
            labels.append(label_id)
            count += 1
    return datas, labels, label_names

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
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        global total_time
        if func.__name__ not in total_time.keys():
            total_time[func.__name__] = 0
        else:
            total_time[func.__name__] += end-start
        return result
    return wrapper

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
