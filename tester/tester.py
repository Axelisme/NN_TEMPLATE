
"""define a class to test the model"""

from typing import Dict, Optional
from tqdm.auto import tqdm
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection, MeanMetric
from config.configClass import Config

class Tester:
    def __init__(self,
                 model: nn.Module,
                 config: Config,
                 loader: DataLoader,
                 criterion: Optional[nn.Module] = None,
                 evaluators: Optional[MetricCollection] = None) -> None:
        '''initialize a tester:
        input: model: the model to test,
               config: the config of this model,
               loader: the dataloader of test set,
               evaluators: the evaluator collection to use'''
        self.model = model
        self.device = config.device
        self.test_loader = loader
        self.criterion = criterion
        self.loss_statistic = MeanMetric()
        self.evaluators = MetricCollection([]) if evaluators is None else evaluators


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
        if self.criterion is not None:
            self.criterion.to(self.device)
            self.criterion.eval()
        self.loss_statistic.to(self.device).reset()
        self.evaluators.to(self.device).reset()

        # evaluate this model
        with torch.no_grad():
            for batch_idx,(input,label) in enumerate(tqdm(self.test_loader, desc='Test ', dynamic_ncols=True)):
                # move input and label to device
                input = Tensor(input).to(self.device)
                label = Tensor(label).to(self.device)
                # forward
                output = self.model(input)
                # calculate loss
                if self.criterion is not None:
                    loss = self.criterion(output, label)
                    self.loss_statistic.update(loss)
                # calculate score
                self.evaluators.update(output, label)

        # return score
        return {**self.evaluators.compute(), 'valid_loss':self.loss_statistic.compute()}

