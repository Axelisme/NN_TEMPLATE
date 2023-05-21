
"""define a class for the dataset"""

import os
import util.utility as ul
import hyperparameter as p
from config.configClass import Config
import torch
import torch.utils.data as data

class DataSet(data.Dataset):
    """define a class for the dataset"""
    @classmethod
    def load_data(cls, config: Config|None = None) -> None:
        """load the data"""
        """load the data"""
        if hasattr(cls, "have_loaded") and cls.have_loaded:
            return
        if config is None:
            raise ValueError("config is None")
        cls.config = config
        cls.have_loaded = True

        # load data
        cls.data_root = p.PROCESSED_DATA_DIR
        cls.db = data.Dataset()                # TODO: load data
        cls.train_db, cls.test_db = data.random_split(cls.db, [1-config.test_ratio, config.test_ratio])

    def __init__(self, data_type: str):
        super(DataSet, self).__init__()

        if data_type == "train":
            self.datas = DataSet.train_db
        elif data_type == "valid":
            self.datas = DataSet.test_db
        elif data_type == "test":
            self.datas = DataSet.test_db
        else:
            raise ValueError("data_type must be 'train' or 'valid'")

        self.data_type = data_type

    def __getitem__(self,idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)