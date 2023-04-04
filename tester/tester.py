
"""define a class to test the model"""

import util.utility as ul
from config.config import Config
from evaluator.ABC import Evaluator
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict

class Tester:
    def __init__(self,
                 model: nn.Module,
                 config: Config,
                 test_loader: DataLoader,
                 evaluators: Dict[str,Evaluator]) -> None:
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.evaluators = evaluators
    
    def eval(self) -> dict:
        '''test a model on test set'''
        # move model to device
        self.model.to(self.config.device)
        
        # set model to eval mode
        self.model.eval()

        # evaluate this model
        test_scores = {name:ul.Statistic() for name in self.evaluators.keys()}
        with torch.no_grad():
            for idx,(input,label) in enumerate(tqdm(self.test_loader)):
                # move input and label to device
                input = input.to(self.config.device)
                label = label.to(self.config.device)
                # forward
                output = self.model(input)
                # calculate score
                for name,evaluator in self.evaluators.items():
                    test_scores[name].update(evaluator.eval(output, label).detach().cpu().item())

        # return score
        return {name:score.avg for name,score in test_scores.items()}

