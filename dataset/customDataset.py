
"""define a class for the dataset"""

import h5py
from typing import Callable, Optional
from os import path
import torch.utils.data as data
from util.tool import clear_folder
from hyperparameter import *
from config.configClass import Config

def load_hdf5(data_type: str, dataset_name: str) -> h5py.File:
    return h5py.File(path.join(PROC_DATA_DIR, data_type, dataset_name), "r")

def sample_saver(sample_dir ,input, label, label_names):
    """save transformed input and label to sample folder"""
    return None

def save_samples(reader, data_type, transform = None, freq = 100, max_num = 1000):
    TYPE_EX_DIR = path.join(TRAIN_EX_DIR, data_type)
    clear_folder(TYPE_EX_DIR)
    label_names = eval(reader.attrs["label_names"])
    for idx, (input, label) in enumerate(reader['dataset']):
        if idx % freq != 0 or idx >= max_num*freq:
            continue
        if transform is not None:
            input = transform(input)
        sample_saver(TYPE_EX_DIR, input, label, label_names)

class CustomDataSet(data.Dataset):
    """define a class for the dataset"""
    def __init__(self, conf: Config, data_type: str, dataset_name = "dataset.hdf5", transform:Optional[Callable] = None):
        """initialize the dataset
            conf: the config object
            data_type: the type of the dataset, "train", "val" or "test"
            dataset_name: the name of the dataset file
            transform: the transform function of the input
        """
        super(CustomDataSet, self).__init__()
        self.conf = conf
        self.data_type = data_type
        self.dataset_name = dataset_name
        self.transform = transform

        # Don't load file handler in initial to avoid problem of multi-process
        self.fileHandler = None
        self.dataset = None

        # load dataset attributes for __len__
        with load_hdf5(data_type, dataset_name)  as reader:
            self.length = reader.attrs["length"]
            self.label_names = eval(reader.attrs["label_names"]) #type:ignore

        # save some samples of input and label
        save_samples(self.data_type, self.dataset_name, self.transform, freq = 100, max_num = 1000)

    def __del__(self):
        # close file handler if exists
        if self.fileHandler is not None:
            self.fileHandler.close()

    def __getitem__(self, idx):
        # load file handler at first time of __getitem__
        if self.fileHandler is None:
            self.__lazy_load()
        input, label = self.dataset[idx]  # type:ignore
        if self.transform is not None:
            input = self.transform(input)
        return input, label

    def __len__(self):
        return self.length

    def __lazy_load(self):
        if self.fileHandler is None:
            self.fileHandler = load_hdf5(self.data_type, self.dataset_name)
            self.dataset = self.fileHandler["dataset"]
        else:
            raise Exception("fileHandler is not None")

