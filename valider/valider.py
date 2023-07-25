
"""define a class to valid the model"""

from typing import Dict, Optional
from tqdm.auto import tqdm
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from config.configClass import Config

class Valider:
    def __init__(self,
                 config: Config,
                 model: nn.Module,
                 device: torch.device,
                 loader: DataLoader,
                 evaluators: MetricCollection = MetricCollection([])) -> None:
        '''initialize a valider:
        input: config: Config, the config of this model,
               model: the model to test,
               config: the config of this model,
               loader: the dataloader of test set,
               evaluators: the evaluator collection to use'''
        self.config = config
        self.model = model
        self.device = device
        self.dataloader = loader
        self.evaluators = evaluators


    def add_evaluator(self, evaluator: Metric, name: Optional[str] = None) -> None:
        '''add an evaluator to this tester:
        input: evaluator: Evaluator, the evaluator to add,
               name: str|None = None, the name of this evaluator'''
        if name is None:
            self.evaluators.add_metrics(evaluator)
        else:
            self.evaluators.add_metrics({name:evaluator})

    def eval(self) -> Dict[str, Tensor]:
        '''test a model on test set:
        output: Dict, the result of this model'''
        # move module to device
        self.model.to(self.device)
        self.evaluators.to(self.device)

        # initial model
        self.model.eval()

        # initial evaluators
        self.evaluators.reset()

        # evaluate this model
        with torch.no_grad():
            batch_num = len(self.dataloader)
            pbar = tqdm(self.dataloader, total=batch_num, desc='Test ', dynamic_ncols=True)
            for batch_idx, (input, label) in enumerate(pbar, start=1):
                # move input and label to device
                input = input.to(self.device)
                label = label.to(self.device)
                # forward
                output = self.model(input)
                # compute and record score
                self.evaluators.update(output, label)

        # return score
        return self.evaluators.compute()

