
"""define a class to valid the model"""

from typing import Dict
from tqdm.auto import tqdm

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

class Valider:
    def __init__(self,
                 model: nn.Module,
                 loader: DataLoader,
                 metrics: MetricCollection,
                 args,
                 **kwargs) -> None:
        '''initialize a valider:
            model: nn.Module, the model to validate,
            loader: DataLoader, the data loader to load data,
            metrics: MetricCollection, the metrics to evaluate the model,
            args: Config, the config of this model'''
        self.model = model
        self.loader = loader
        self.metrics = metrics
        self.device = torch.device(args.device)


    def eval(self) -> Dict[str, Tensor]:
        '''test a model on test set:
            output: Dict, the result of this model'''
        # move module to device
        self.model.to(self.device)
        self.metrics.to(self.device)

        # initial model
        self.model.eval()

        # initial evaluators
        self.metrics.reset()

        # evaluate this model
        with torch.no_grad():
            batch_num = len(self.loader)
            pbar = tqdm(self.loader, total=batch_num, desc='Valid ', dynamic_ncols=True)
            for batch_idx, (input, *other) in enumerate(pbar, start=1):
                # move input and label to device
                input = input.to(self.device)
                other = [item.to(self.device) for item in other if isinstance(item, Tensor)]
                # forward
                output = self.model(input)
                # compute and record score
                self.metrics.update(output, *other)

        # return score
        return self.metrics.compute()

