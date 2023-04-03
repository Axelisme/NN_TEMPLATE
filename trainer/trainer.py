
"""define a class for training a model"""

import util.utility as ul
from evaluator.ABC import Evaluator
import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Tuple

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 epochs: int,
                 scheduler,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 score: Evaluator):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion
        self.score = score

    def train(self) -> Tuple[ul.Result, ul.Result]:
        '''train a model'''
        # move model to device
        self.model.to(self.device)
        # train model
        train_result = ul.Result()
        valid_result = ul.Result()
        for epoch in range(self.epochs):
            # print epoch
            print('-'*10)
            print(f'Epoch {epoch+1}/{self.epochs}')

            # train for one epoch
            self.model.train()
            train_loss = ul.AverageMeter()
            for input,label in tqdm(self.train_loader):
                # move input and label to device
                input = input.to(self.device)
                label = label.to(self.device)
                # forward
                output:Tensor = self.model(input)
                loss:Tensor = self.criterion(output, label)
                train_loss.update(loss.sum().item(), loss.size(0))
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # store training loss
            train_result.add(loss=train_loss.avg)
            # show training loss
            print('Training:')
            train_result.show('loss')

            # update learning rate
            self.scheduler.step()

            # evaluate
            self.model.eval()
            valid_loss = ul.AverageMeter()
            valid_score = ul.AverageMeter()
            with torch.no_grad():
                for input,label in tqdm(self.valid_loader):
                    # move input and label to device
                    input = input.to(self.device)
                    label = label.to(self.device)
                    # forward
                    output = self.model(input)
                    # calculate loss and score
                    loss = self.criterion(output, label)
                    score = self.score.eval(output, label)
                    # store loss and score
                    valid_loss.update(loss.sum().item(), loss.size(0))
                    valid_score.update(score.sum().item(), score.size(0))
            # store validation loss and score
            valid_result.add(loss=valid_loss.avg, score=valid_score.avg)
            # show validation loss and score
            print('Validation:')
            valid_result.shows(['loss','score'])


        return train_result, valid_result

