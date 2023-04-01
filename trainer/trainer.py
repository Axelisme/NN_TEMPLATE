
"""define a class for training a model"""

import util.utility as ul
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from evaluator.eval_abc import Evaluator
from tqdm.auto import tqdm
import typing as t


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
                 score: Evaluator) -> None:
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion
        self.score = score

    def train(self) -> t.Tuple[t.List[float], t.List[float], t.List[float]]:
        '''train a model'''
        # move model to device
        self.model.to(self.device)
        # train model
        train_losses = []
        valid_losses = []
        valid_scores = []
        for epoch in range(self.epochs):
            # print epoch
            print(f'Epoch {epoch+1}/{self.epochs}')
            print('-'*10)

            # train for one epoch
            self.model.train()
            train_loss = ul.AverageMeter()
            for input,label in tqdm(self.train_loader):
                # move input and label to device
                input = input.to(self.device)
                label = label.to(self.device)
                # forward
                output = self.model(input)
                loss = self.criterion(output, label)
                train_loss.update(loss.sum().item(), input.size(0))
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # print loss
            print(f'train loss: {train_loss.avg}')
            # store loss
            train_losses.append(train_loss.avg)

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
                    # update loss and score
                    valid_loss.update(loss.sum().item(), input.size(0))
                    valid_score.update(score.sum().item(), input.size(0))
            # print loss and score
            print(f'validation loss: {valid_loss.avg}')
            print(f'validation score: {valid_score.avg}')
            # store loss and score
            valid_losses.append(valid_loss.avg)
            valid_scores.append(valid_score.avg)

        return train_losses, valid_losses, valid_scores

