
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
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self) -> dict:
        '''train a model'''
        # move model to device
        self.model.to(self.config.device)

        # set model to train mode
        self.model.train()
        
        # train for one epoch
        train_loss = ul.Statistic()
        for idx,(input,label) in enumerate(tqdm(self.train_loader)):
            # move input and label to device
            input:Tensor = input.to(self.config.device)
            label:Tensor = label.to(self.config.device)
            # forward
            output:Tensor = self.model(input)
            loss:Tensor = self.criterion(output, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # store loss
            train_loss.update(loss.detach().cpu().item())
        
        # return loss
        return {'train_loss':train_loss.avg}

        

