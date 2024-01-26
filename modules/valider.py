
"""define a class for valid a model"""

from typing import Dict
from tqdm.auto import tqdm
from argparse import Namespace


import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from util.utility import set_seed


class Valider:
    def __init__(
            self,
            model: nn.Module,
            metrics: MetricCollection,
            device: str,
            silent: bool,
        ):
        self.model = model
        self.valid_metrics = metrics

        self.device = torch.device(device)
        self.silent = silent

        self.valid_metrics.reset()


    def close(self):
        '''close the valider'''
        self.valid_metrics.reset()


    def set_eval(self):
        '''set model and criterion to eval mode'''
        self.model.eval()
        self.model.to(self.device)
        self.valid_metrics.to(self.device)


    def pop_result(self) -> Dict[str, Tensor]:
        '''get the score of validation and reset the statistic of score'''
        scores = self.valid_metrics.compute()
        self.valid_metrics.reset()
        return scores


    def one_epoch(self, valid_loader: DataLoader, name: str):
        old_seed = set_seed(0)
        self.set_eval()
        with torch.no_grad():
            for input, *other in tqdm(valid_loader, desc=name.capitalize(), dynamic_ncols=True, disable=self.silent):
                # move input to device
                input = input.to(self.device)

                # forward
                output:Tensor = self.model(input)

                # compute and record score
                other = [item.to(self.device) for item in other if isinstance(item, Tensor)]
                self.valid_metrics.update(output, *other)
                del output, input, other

        set_seed(old_seed)

