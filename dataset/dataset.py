
"""define a class for the dataset"""

import global_var.path as p
import util.utility as ul
import torch
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class MyDataSet(data.Dataset):
    """define a class for the dataset"""
    @classmethod
    def load_data(cls) -> None:
        """load the data"""
        cls.Data = load_digits()
        cls.train_data, cls.test_data, cls.train_label, cls.test_label = train_test_split(cls.Data.data, cls.Data.target, test_size=0.2, random_state=0)
        
    def __init__(self, data_type: str):
        super(MyDataSet, self).__init__()
        if data_type == 'train':
            self.input = self.train_data
            self.label = self.train_label
        elif data_type == 'valid':
            self.input = self.test_data
            self.label = self.test_label
        elif data_type == 'test':
            self.input = self.test_data
            self.label = self.test_label
        else:
            raise ValueError('data_type must be train or test')

    def __getitem__(self,idx):
        input = torch.tensor(self.input[idx], dtype=torch.float32).reshape((1,8,8))
        label = self.label[idx]
        return input, label

    def __len__(self):
        return len(self.input)

    @classmethod
    def discription(cls) -> None:
        print(cls.Data.DESCR)