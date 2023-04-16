
"""define a class for training a model"""

import util.utility as ul
from config.config import Config
import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 config: Config,
                 train_loader: DataLoader,
                 optimizer: Optimizer,
                 criterion: nn.Module):
        '''initialize a trainer:
        input: model: nn.Module, the model to train,
                config: Config, the config of this model,
                train_loader: DataLoader, the dataloader of train set,
                optimizer: Optimizer, the optimizer of this model,
                criterion: nn.Module, the criterion of this model'''
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self) -> dict:
        '''train a model for one epoch:
        output: dict('train_loss', loss), the loss of this epoch'''
        # move model to device
        self.model.to(self.config.device)

        # set model to train mode
        self.model.train()

        # train for one epoch
        train_loss = ul.Result()
        for batch_idx,(input,label) in enumerate(tqdm(self.train_loader, desc='Train')):
            # move input and label to device
            input:Tensor = input.to(self.config.device)
            label:Tensor = label.to(self.config.device)
            # forward
            self.optimizer.zero_grad()
            output:Tensor = self.model(input)
            loss:Tensor = self.criterion(output, label)
            # backward
            loss.backward()
            self.optimizer.step()
            # store loss
            train_loss.log(train_loss=loss.cpu().item())

        # return loss
        return train_loss.value

