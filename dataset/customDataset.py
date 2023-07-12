
"""define a class for the dataset"""

import os
import h5py
from typing import Callable, Optional
import torch.utils.data as data
from hyperparameters import PROC_DATA_DIR
from config.configClass import Config

def load_processed_dataset(data_name: str, file_name: str) -> h5py.File:
    """load the processed dataset from the file."""
    return h5py.File(os.path.join(PROC_DATA_DIR, data_name, file_name), "r")


class CustomDataSet(data.Dataset):
    """define a class for the dataset"""
    def __init__(self, conf: Config, data_name: str, file_name = "dataset.hdf5", transform:Optional[Callable] = None):
        """initialize the dataset
            conf: the config object.
            data_name: the type name of the dataset, like "train", "valid" or "test".
            file_name: the file name of the dataset file.
            transform: the transform function of the input.
        """
        super(CustomDataSet, self).__init__()
        self.conf = conf
        self.data_name = data_name
        self.file_name = file_name
        self.transform = transform

        # load dataset meta data
        with load_processed_dataset(data_name, file_name)  as reader:
            self.length = reader.attrs["length"]

        # Don't load file handler in init() to avoid problem of multi-process in DataLoader
        # instead use __lazy_load() in __getitem__()
        self.fileHandler = None
        self.dataset = None


    def __del__(self):
        # close file handler if exists
        if self.fileHandler is not None:
            self.fileHandler.close()


    def __getitem__(self, idx):
        # load file handler at first time of __getitem__
        if self.fileHandler is None:
            self.__lazy_load()
        # get data
        input, label = self.dataset[idx]  # type:ignore
        # transform input if needed
        if self.transform is not None:
            input = self.transform(input)
        return input, label


    def __len__(self):
        return self.length


    def __lazy_load(self):
        self.fileHandler = load_processed_dataset(self.data_name, self.file_name)
        self.dataset = self.fileHandler["dataset"]

