"""
Some useful tools
"""

import os
import random


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
    return ShuffledIterable(iterable)


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
        if root == dir:  # skip the root folder
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
