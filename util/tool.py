"""
Some useful tools
"""

import os
import time
import shutil
from functools import wraps
import random

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


class ShuffledIterable:
    """shuffle the iterable"""
    def __init__(self, iterable):
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
