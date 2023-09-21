
"""define a class for the dataset"""

import h5py
from typing import Callable, Optional
import torch.utils.data as data
from config.configClass import Config

def load_processed_dataset(file_path: str):
    """load the processed dataset from the file."""
    return h5py.File(file_path, "r")


class CustomDataSet(data.Dataset):
    """define a class for the dataset"""
    def __init__(self, conf: Config, dataset_path:str, transform:Optional[Callable] = None):
        """initialize the dataset
            conf: the config object.
            dataset_path: the file path of the dataset file.
            transform: the transform function before input the data to the model.
        """
        super(CustomDataSet, self).__init__()
        self.conf = conf
        self.dataset_path = dataset_path
        self.transform = transform

        # load dataset meta data
        with load_processed_dataset(self.dataset_path)  as reader:
            self.length = reader.attrs["length"]

        # Don't load file handler in init() to avoid problem of multi-process in DataLoader
        # instead use __lazy_load() in __getitem__()
        self.fileHandler = None
        self.dataset = None


    def __del__(self):
        # close file handler if exists
        if hasattr(self,'fileHandler') and self.fileHandler is not None:
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
        self.fileHandler = load_processed_dataset(self.dataset_path)
        self.dataset = self.fileHandler["dataset"]

