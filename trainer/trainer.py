
"""define a train function for training a model"""

import util.utility as ul
import torch
from torch import Tensor
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Callable

def trainer(model: Module,
          device: device,
          train_loader: DataLoader,
          valid_loader: DataLoader,
          epochs: int,
          scheduler: _LRScheduler,
          optimizer: Optimizer,
          loss_func: Callable[[Tensor, Tensor],Tensor],
          score_func: Callable[[Tensor, Tensor],Tensor] ) -> None:
    '''train a model'''
    # move model to device
    model.to(device)
    # train model
    print('Training model...')
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-'*10)
        # train for one epoch
        model.train()
        train_loss_average: ul.AverageMeter = ul.AverageMeter()
        for input,label in train_loader:
            # move input and label to device
            input = input.to(device)
            label = label.to(device)
            # forward
            output = model(input)
            loss = loss_func(output, label)
            train_loss_average.update(loss.sum().item(), input.size(0))
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update learning rate
            scheduler.step()
        # print loss
        print(f'train loss: {train_loss_average.avg}')

        # evaluate
        model.eval()
        valid_loss_average: ul.AverageMeter = ul.AverageMeter()
        valid_score_average: ul.AverageMeter = ul.AverageMeter()
        with torch.no_grad():
            for input,label in valid_loader:
                # move input and label to device
                input = input.to(device)
                label = label.to(device)
                # forward
                output = model(input)
                # calculate loss and score
                loss = loss_func(output, label)
                score = score_func(output, label)
                # update loss and score
                valid_loss_average.update(loss.sum().item(), input.size(0))
                valid_score_average.update(score.sum().item(), input.size(0))
        # print loss and score
        print(f'validation loss: {valid_loss_average.avg}')
        print(f'validation score: {valid_score_average.avg}')
