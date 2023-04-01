
"""define a class to test the model"""

import config.path as p
import util.utility as ul
from evaluator.eval_abc import Evaluator
import torch
from torch import nn
from torch import device
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class Inferencer:
    def __init__(self,
                 model: nn.Module,
                 device: device,
                 test_loader: DataLoader,
                 score: Evaluator ) -> None:
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.score = score
    
    def evaluate(self) -> float:
        '''test a model on test set'''
        # move model to device
        self.model.to(self.device)
        
        # evaluate this model
        self.model.eval()
        test_score_average: ul.AverageMeter = ul.AverageMeter()
        with torch.no_grad():
            for input,label in tqdm(self.test_loader):
                # move input and label to device
                input = input.to(self.device)
                label = label.to(self.device)
                # forward
                output = self.model(input)
                # calculate score
                score = self.score.eval(output, label)
                # update loss and score
                test_score_average.update(score.sum().item(), input.size(0))
        # print loss and score
        print(f'test score: {test_score_average.avg}')

        # return score
        return test_score_average.avg

