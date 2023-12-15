
from typing import List, Dict
from importlib import import_module

from torch.utils.data import ConcatDataset


class MergeDataset(ConcatDataset):
    def __init__(self, module:str, name:str, kwargs_list:List[Dict], *args, **kwargs):
        module = import_module(module)
        dataset_class = getattr(module, name)

        datasets = [dataset_class(**kws) for kws in kwargs_list]

        super(MergeDataset, self).__init__(datasets, *args, **kwargs)