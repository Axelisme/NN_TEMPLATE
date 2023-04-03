
"""define a class for training a model"""

import util.utility as ul
from evaluator.ABC import Evaluator
import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

    def train(self) -> ul.Result:
        '''train a model'''
        # move model to device
        self.model.to(self.device)
        # train model
        result = ul.Result()
        for epoch in range(self.epochs):
            # print epoch
            print('-'*50)
            print(f'Epoch {epoch+1}/{self.epochs}')

            # train for one epoch
            self.model.train()
            train_loss = ul.AverageMeter()
            print('Training...')
            for input,label in tqdm(self.train_loader):
                # move input and label to device
                input = input.to(self.device)
                label = label.to(self.device)
                # forward
                output:Tensor = self.model(input)
                loss:Tensor = self.criterion(output, label)
                train_loss.update(loss.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # store training loss
            result.add(train_loss=train_loss.avg)

            # update learning rate
            self.scheduler.step()

            # evaluate
            self.model.eval()
            valid_loss = ul.AverageMeter()
            valid_score = ul.AverageMeter()
            print('Validating...')
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
                    valid_loss.update(loss.item())
                    valid_score.update(score.item())
            # store validation loss and score
            result.add(valid_loss=valid_loss.avg, valid_score=valid_score.avg)

            # show loss and score
            print('Result:')
            result.shows(['train_loss','valid_loss','valid_score'])

        return result

