
"""define a class to test the model"""

from typing import Dict, Optional
from tqdm.auto import tqdm
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection

class Tester:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 loader: DataLoader,
                 evaluators: MetricCollection = MetricCollection([])) -> None:
        '''initialize a tester:
        input: model: the model to test,
               config: the config of this model,
               loader: the dataloader of test set,
               evaluators: the evaluator collection to use'''
        self.model = model
        self.device = device
        self.test_loader = loader
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
        # initial model
        self.model.to(self.device)
        self.model.eval()
        self.evaluators.to(self.device).reset()

        # evaluate this model
        with torch.no_grad():
            for batch_idx,(input,label) in enumerate(tqdm(self.test_loader, desc='Test ', dynamic_ncols=True)):
                # move input and label to device
                input = Tensor(input).to(self.device)
                label = Tensor(label).to(self.device)
                # forward
                output = self.model(input)
                # calculate score
                self.evaluators.update(output, label)

        # return score
        return self.evaluators.compute()

