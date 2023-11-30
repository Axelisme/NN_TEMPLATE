
"""define a class for training a model"""

from typing import Dict
from tqdm.auto import tqdm

import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 loader: DataLoader,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 args,
                 gradient_accumulation_steps:int = 1,
                 **kwargs):
        '''initialize a trainer:
            model: nn.Module, the model to train,
            loader: DataLoader, the data loader to load data,
            optimizer: Optimizer, the optimizer to update parameters,
            criterion: nn.Module, the criterion to compute loss,
            args: Config, the config of this model'''
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.result_status = MeanMetric()
        self.device = torch.device(args.device)
        self.slient = args.slient

    def fit(self, stepN = -1) -> Dict[str, Tensor]:
        '''train a model for one epoch:
            output: dict('train_loss', loss), the loss of this epoch'''
        # move module to device
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.result_status.to(self.device)

        # initial model and criterion
        self.model.train()
        self.criterion.train()

        # initial optimizer
        self.optimizer.zero_grad()

        # initial statistic
        self.result_status.reset()

        # train for one epoch
        batch_num = len(self.loader)
        pbar = tqdm(self.loader, total=batch_num, desc='Train', dynamic_ncols=True, disable=self.slient)
        for batch_idx, (input, *other) in enumerate(pbar, start=1):
            # move input and label to device
            input = input.to(self.device)
            other = [item.to(self.device) for item in other if isinstance(item, Tensor)]
            # forward
            output:Tensor = self.model(input)
            # compute loss
            loss:Tensor = self.criterion(output, *other)
            self.result_status.update(loss)
            # backward
            loss.backward()
            # update parameters
            if batch_idx % self.gradient_accumulation_steps == 0 or batch_idx == batch_num:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # return statistic result
        return {'Train_loss': self.result_status.compute()}

