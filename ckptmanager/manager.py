"""
define a class to manage the checkpoint
"""
import os
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Tuple
from config.configClass import Config
from hyperparameters import SAVED_MODELS_DIR

class CheckPointManager:
    def __init__(self,
                 conf:Config,
                 model:Module,
                 ckpt_dir:Optional[str] = None,
                 optim:Optional[Optimizer] = None,
                 scheduler:Optional[LRScheduler] = None,):
        self.conf       = conf
        self.model      = model
        self.ckpt_dir   = ckpt_dir if ckpt_dir else os.path.join(SAVED_MODELS_DIR, conf.model_name)
        self.optim      = optim
        self.scheduler  = scheduler

        self.best_score: Optional[float] = None
        self.epoch: Optional[int]        = None

    def default_ckpt_path(self) -> str:
        model_name = self.conf.model_name
        return os.path.join(self.ckpt_dir, f'ckpt_{model_name}.pt')

    def load(self,
             ckpt_path: Optional[str] = None,
             device = torch.device('cpu')) -> Tuple[float, int]:
        """Load checkpoint."""
        # create the default path if not specified
        if ckpt_path is None:
            ckpt_path = self.default_ckpt_path()


        # check if the file exists
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"File {ckpt_path} does not exist.")


        # load checkpoint
        print(f'Loading checkpoint from {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location=device)

        self.model.load_state_dict(checkpoint['model'])
        if device is not None:
            self.model.to(device)

        if self.optim is not None:
            self.optim.load_state_dict(checkpoint['optimizer'])

        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.best_score = checkpoint['best_score']
        self.epoch = checkpoint['epoch']

        return self.best_score, self.epoch

    def save(self,
             score: float,
             epoch: int,
             ckpt_path: Optional[str] = None,
             overwrite: bool = False) -> None:
        """Save the checkpoint."""
        # create the default path if not specified
        if ckpt_path is None:
            ckpt_path = self.default_ckpt_path()

        # check if the file exists and create the directory if not
        if os.path.exists(ckpt_path) and not overwrite:
            raise FileExistsError(f"File {ckpt_path} already exists.")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # save the checkpoint
        print(f'Saving model to {ckpt_path}')
        save_dict = {}

        save_dict['model'] = self.model.state_dict()

        if self.optim is not None:
            save_dict['optimizer'] = self.optim.state_dict()

        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()

        save_dict['best_score'] = score
        save_dict['epoch'] = epoch

        torch.save(save_dict, ckpt_path)

    def update(self, score: float, epoch: int, overwrite=True, lower_better=False, **kwargs) -> bool:
        """save the checkpoint if the score is better."""
        if  self.best_score is None or \
                not lower_better and score > self.best_score or \
                lower_better and score < self.best_score:
            self.save(score=score, epoch=epoch, overwrite=overwrite, **kwargs)
            self.best_score = score
            self.epoch = epoch
            return True
        return False