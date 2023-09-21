"""
define a class to manage the checkpoint
"""
import os
from pathlib import Path
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Tuple, List
from config.configClass import Config
from hyperparameters import SAVED_MODELS_DIR

class CheckPointManager:
    def __init__(self,
                 conf:Config,
                 model:Module,
                 ckpt_dir:Optional[str] = None,
                 optim:Optional[Optimizer] = None,
                 scheduler:Optional[LRScheduler] = None,
                 lower_better:bool = True,
                 keep_num:int = 3):
        self.conf       = conf
        self.model      = model
        self.ckpt_dir   = ckpt_dir if ckpt_dir else os.path.join(SAVED_MODELS_DIR, conf.model_name)
        self.optim      = optim
        self.scheduler  = scheduler
        self.lower_better = lower_better
        self.keep_num   = keep_num

        self.best_score: Optional[float] = None
        self.epoch: Optional[int]        = None

    def get_ckpts(self) -> List[Path]:
        model_name = self.conf.model_name
        ckpts = list(Path(self.ckpt_dir).glob(f"{model_name}_E_*_S_*.pth"))
        return sorted(ckpts, key=lambda x: self.parse_ckpt_path(x)[0], reverse=self.lower_better)

    def default_load_path(self) -> Path:
        """Get the default path to load the checkpoint."""
        ckpts = self.get_ckpts()
        if len(ckpts) == 0:
            raise FileNotFoundError(f"No checkpoint found in {self.ckpt_dir}")
        return ckpts[-1]

    def default_save_path(self, score: float, epoch: int) -> Path:
        """Get the default path to save the checkpoint."""
        model_name = self.conf.model_name
        ckpt = Path(self.ckpt_dir) / f"{model_name}_E_{epoch}_S_{score:.4f}.pth"
        return ckpt

    def parse_ckpt_path(self, ckpt: str) -> Tuple[float, int]:
        """Parse the score and epoch from the checkpoint path."""
        ckpt:Path = Path(ckpt)
        try:
            epoch = int(ckpt.stem.split('_')[-3])
            score = float(ckpt.stem.split('_')[-1])
        except:
            print(f"Cannot parse the epoch and score from {ckpt.stem}")
            print("Set epoch and score to None.")
            epoch = None
            score = None
        return score, epoch

    def clean_old_ckpts(self):
        ckpts = self.get_ckpts()
        if len(ckpts) > self.keep_num:
            for ckpt in ckpts[self.keep_num:]:
                os.remove(ckpt)

    def load(self,
             ckpt_path: Optional[str] = None,
             device = torch.device('cpu')):
        """Load checkpoint."""
        # create the default path if not specified
        if ckpt_path is None:
            ckpt_path:Path = self.default_load_path()
        else:
            ckpt_path:Path = Path(ckpt_path)

        # check if the file exists
        if not ckpt_path.exists():
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

        # load the best score and epoch

        self.best_score = checkpoint.get('best_score', None)
        if self.best_score is None:
            print(f"Cannot find best_score in {ckpt_path.stem}")
            print("Set best_score to None.")

        self.epoch = checkpoint.get('epoch', None)
        if self.epoch is None:
            print(f"Cannot find epoch in {ckpt_path.stem}")
            print("Set epoch to None.")

    def save(self,
             score: float,
             epoch: int,
             ckpt_path: Optional[str] = None,
             overwrite: bool = False):
        """Save the checkpoint."""
        # create the default path if not specified
        if ckpt_path is None:
            ckpt_path = self.default_save_path(score=score, epoch=epoch)
        else:
            ckpt_path = Path(ckpt_path)

        # check if the file exists and create the directory if not
        if ckpt_path.exists() and not overwrite:
            raise FileExistsError(f"File {ckpt_path} already exists.")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # save the checkpoint
        print(f'Saving model to {ckpt_path}')
        save_dict = {}

        save_dict['model'] = self.model.state_dict()

        if self.optim is not None:
            save_dict['optimizer'] = self.optim.state_dict()

        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()

        torch.save(save_dict, ckpt_path)

        # clean old checkpoints
        self.clean_old_ckpts()

    def update(self, score: float, epoch: int, overwrite=True, **kwargs) -> bool:
        """save the checkpoint if the score is better."""
        if  self.best_score is None or (self.lower_better ^ (score >= self.best_score)):
            self.save(score=score, epoch=epoch, overwrite=overwrite, **kwargs)
            self.best_score = score
            self.epoch = epoch
            return True
        return False