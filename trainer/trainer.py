
"""define a class for training a model"""

from typing import Dict
from tqdm.auto import tqdm
import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, Metric
from config.configClass import Config

class Trainer:
    def __init__(self,
                 config: Config,
                 model: nn.Module,
                 device: torch.device,
                 loader: DataLoader,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 statistic: Metric = MeanMetric(),):
        '''initialize a trainer:
        input: config: Config, the config of this model,
                model: nn.Module, the model to train,
                device: the device to use,
                loader: the dataloader of train set,
                optimizer: the optimizer of this model,
                criterion: the criterion of this model,
                statistic: the statistic method of the loss for each batch'''
        self.config = config
        self.model = model
        self.device = device
        self.dataloader = loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.statistic = statistic

    def fit(self) -> Dict[str, Tensor]:
        '''train a model for one epoch:
        output: dict('train_loss', loss), the loss of this epoch'''
        # move module to device
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.statistic.to(self.device)

        # initial model and criterion
        self.model.train()
        self.criterion.eval()

        # initial optimizer
        self.optimizer.zero_grad()

        # initial statistic
        self.statistic.reset()

        # set gradient accumulation period
        gradient_accumulation_steps = self.config.gradient_accumulation_steps

        # train for one epoch
        batch_num = len(self.dataloader)
        pbar = tqdm(self.dataloader, total=batch_num, desc='Train', dynamic_ncols=True)
        for batch_idx, (input, label) in enumerate(pbar, start=1):
            # move input and label to device
            input = input.to(self.device)
            label = label.to(self.device)
            # forward
            output:Tensor = self.model(input)
            # compute loss
            loss:Tensor = self.criterion(output, label)
            self.statistic.update(loss)
            # backward
            loss.backward()
            # update parameters
            if batch_idx % gradient_accumulation_steps == 0 or batch_idx == batch_num:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # return statistic result
        return self.statistic.compute()

