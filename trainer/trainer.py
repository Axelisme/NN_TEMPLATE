
"""define a class for training a model"""

from typing import Dict
from tqdm.auto import tqdm
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, Metric
import util.utility as ul
from config.configClass import Config

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 config: Config,
                 loader: DataLoader,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 statistic: Metric = MeanMetric(),):
        '''initialize a trainer:
        input: model: nn.Module, the model to train,
                config: the config of this model,
                train_loader: the dataloader of train set,
                optimizer: the optimizer of this model,
                criterion: the criterion of this model,
                statistic: the statistic method of the loss for each batch'''
        self.model = model
        self.device = config.device
        self.train_loader = loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.statistic = statistic

    def train(self) -> Dict[str, Tensor]:
        '''train a model for one epoch:
        output: dict('train_loss', loss), the loss of this epoch'''
        # move model to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # set model to train mode
        self.model.train()

        # train for one epoch
        self.statistic.to(self.device).reset()
        for batch_idx,(input,label) in enumerate(tqdm(self.train_loader, desc='Train', dynamic_ncols=True)):
            # move input and label to device
            input = Tensor(input).to(self.device)
            label = Tensor(label).to(self.device)
            # forward
            self.optimizer.zero_grad()
            output:Tensor = self.model(input)
            loss:Tensor = self.criterion(output, label)
            # backward
            loss.backward()
            self.optimizer.step()
            # store loss
            self.statistic.update(loss)

        # return loss
        return self.statistic.compute()

