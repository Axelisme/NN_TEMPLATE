
"""define a class for training a model"""

from typing import Dict
from tqdm import tqdm
from argparse import Namespace


import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, MetricCollection
from torch.optim.lr_scheduler import _LRScheduler

from util.datatools import cycle_iter


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: _LRScheduler,
                 criterion: nn.Module,
                 args: Namespace,
                 grad_acc_steps:int = 1,
                 **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.args = args

        self.grad_acc_steps = grad_acc_steps

        self.train_metrics = MetricCollection({
            'train_loss': MeanMetric(),
        })
        self.device = torch.device(args.device)

        self.train_bar = tqdm(total=len(train_loader), desc='Train', dynamic_ncols=True, disable=args.slient)

        self.cycle_loader = cycle_iter(train_loader, callback=self.train_bar.reset)

        self.init()


    def init(self):
        '''initial the model and criterion'''
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.train_metrics.to(self.device)

        self.train_metrics.reset()

    def close(self):
        '''close the trainer'''
        self.train_bar.close()


    @property
    def lr(self) -> float:
        '''get the learning rate of optimizer'''
        return self.optimizer.param_groups[0]['lr']


    def set_train(self):
        '''set model and criterion to train mode'''
        # initial model and criterion
        self.model.train()
        self.criterion.train()


    def pop_result(self) -> Dict[str, Tensor]:
        '''get the average loss of training and reset the statistic of loss'''
        loss = self.train_metrics.compute()
        self.train_metrics.reset()
        return loss


    def one_step(self):
        self.set_train()
        for steps, (input, *other) in enumerate(self.cycle_loader, start=1):
            # move input and label to device
            input = input.to(self.device)

            # forward
            output:Tensor = self.model(input)

            # compute loss and record
            other = [item.to(self.device) for item in other if isinstance(item, Tensor)]
            loss:Tensor = self.criterion(output, *other)
            self.train_metrics.update(loss)

            # backward
            (loss / self.grad_acc_steps).backward()

            # update parameters
            if steps >= self.grad_acc_steps:
                self.train_bar.update(steps)
                self.train_bar.refresh()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                break


    def one_epoch(self):
        self.set_train()
        for steps, (input, *other) in enumerate(self.train_loader, start=1):
            # move input to device
            input = input.to(self.device)

            # forward
            output:Tensor = self.model(input)

            # compute loss and record
            other = [item.to(self.device) for item in other if isinstance(item, Tensor)]
            loss:Tensor = self.criterion(output, *other)
            self.train_metrics.update(loss)
            del output, input, other

            # backward
            (loss / self.grad_acc_steps).backward()
            del loss

            # update parameters
            if steps % self.grad_acc_steps == 0 or steps == len(self.train_loader):
                self.train_bar.update(steps)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

