
"""define a class for valid a model"""

from typing import Dict
from tqdm import tqdm
from argparse import Namespace


import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from util.utility import set_seed


class Valider:
    def __init__(self,
                 model: nn.Module,
                 valid_loader: DataLoader,
                 metrics: MetricCollection,
                 args: Namespace,
                 **kwargs):
        self.model = model
        self.valid_loader = valid_loader
        self.valid_metrics = metrics
        self.args = args

        self.device = torch.device(args.device)

        self.init()


    def init(self):
        '''initial the model and criterion'''
        self.model.to(self.device)
        self.valid_metrics.to(self.device)

        self.valid_metrics.reset()

    def close(self):
        '''close the valider'''
        pass


    def set_eval(self):
        '''set model and criterion to eval mode'''
        self.model.eval()


    def pop_result(self) -> Dict[str, Tensor]:
        '''get the score of validation and reset the statistic of score'''
        scores = self.valid_metrics.compute()
        self.valid_metrics.reset()
        return scores


    def one_epoch(self):
        old_seed = set_seed(0)
        self.set_eval()
        with torch.no_grad():
            for input, *other in tqdm(self.valid_loader, desc='Valid ', dynamic_ncols=True, disable=self.args.slient):
                # move input to device
                input = input.to(self.device)

                # forward
                output:Tensor = self.model(input)

                # compute and record score
                other = [item.to(self.device) for item in other if isinstance(item, Tensor)]
                self.valid_metrics.update(output, *other)
                del output, input, other

        set_seed(old_seed)

