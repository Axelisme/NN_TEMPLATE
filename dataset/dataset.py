
"""define a class for the dataset"""

import global_var.path as p
import util.utility as ul
from config.config import Config
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

class DataSet(data.Dataset):
    """define a class for the dataset"""
    @classmethod
    def load_data(cls, config: Config|None = None) -> None:
        """load the data"""
        if hasattr(cls, '_rawData'):
            return
        if config is None:
            raise ValueError("config is None")
        from sklearn.datasets import load_digits
        cls.config = config
        cls._rawData = load_digits()
        data = torch.from_numpy(cls._rawData.data.reshape(-1,1,8,8)).to(torch.float32)  # type: ignore
        target = cls._rawData.target # type: ignore
        cls.train_data, cls.test_data, cls.train_label, cls.test_label = train_test_split(data, target, test_size=cls.config.split_ratio, random_state=0) # type: ignore

    def __init__(self, data_type: str):
        super(DataSet, self).__init__()
        DataSet.load_data()
        if data_type == "train":
            self.input = DataSet.train_data
            self.label = DataSet.train_label
        elif data_type == "valid":
            self.input = DataSet.test_data
            self.label = DataSet.test_label
        elif data_type == "test":
            self.input = DataSet.test_data
            self.label = DataSet.test_label
        else:
            raise ValueError("data_type must be 'train' or 'valid'")

    def __getitem__(self,idx):
        input = self.input[idx]
        label = self.label[idx]
        return input, label

    def __len__(self):
        return len(self.input)