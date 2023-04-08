
"""define a class to test the model"""

import util.utility as ul
from config.config import Config
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm.auto import tqdm
from typing import Dict

class Tester:
    def __init__(self,
                 model: nn.Module,
                 config: Config,
                 test_loader: DataLoader) -> None:
        '''initialize a tester:
        input: model: nn.Module, the model to test,
               config: Config, the config of this model,
               test_loader: DataLoader, the dataloader of test set'''
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.evaluators = dict()

    def add_evaluator(self, evaluator: Metric, name: str|None = None) -> None:
        '''add an evaluator to this tester:
        input: evaluator: Evaluator, the evaluator to add,
               name: str|None = None, the name of this evaluator'''
        if name is None:
            name = type(evaluator).__name__
        self.evaluators[name] = evaluator

    def eval(self) -> dict:
        '''test a model on test set:
        output: dict(evaluator name, score), the result of this model'''
        # move model to device
        self.model.to(self.config.device)

        # set model to eval mode
        self.model.eval()

        # evaluate this model
        test_scores = ul.Result()
        with torch.no_grad():
            for batch_idx,(input,label) in enumerate(tqdm(self.test_loader)):
                # move input and label to device
                input = input.to(self.config.device)
                label = label.to(self.config.device)
                # forward
                output = self.model(input)
                # calculate score
                result = dict()
                for name,evaluator in self.evaluators.items():
                    evaluator = evaluator.to(self.config.device)
                    score = evaluator(output, label).cpu().item()
                    result[name] = score
                test_scores.update(result)

        # return score
        return test_scores.value

