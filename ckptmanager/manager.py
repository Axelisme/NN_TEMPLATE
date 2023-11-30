"""
define a class to manage the checkpoint
"""
import os
import yaml
from pathlib import Path
from typing import Dict

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Tuple, List
from global_vars import SAVED_MODELS_DIR

class CheckPointManager:
    def __init__(self,
                 save_conf:dict,
                 name:str,
                 model:Module,
                 save_mod:str = 'max',
                 check_metric:list = [],
                 ckpt_dir:Optional[str] = None,
                 keep_num:int = 3,
                 **kwargs):
        self.conf       = save_conf
        self.name       = name
        self.model      = model

        assert save_mod in ['max', 'min'], f"save_mod should be 'max' or 'min', but got {save_mod}"
        self.save_mod   = save_mod

        assert len(check_metric) > 0, "check_metric should not be empty"
        self.check_metric = check_metric

        self.ckpt_dir   = ckpt_dir if ckpt_dir else os.path.join(SAVED_MODELS_DIR, self.name)

        assert keep_num > 0, "keep_num should be greater than 0"
        self.keep_num   = keep_num

        self.best_result:dict = {}
        self.epoch: int       = 0

    def save_config(self, config_name:str = 'config.yaml'):
        """Save the config file."""
        dir_path = Path(self.ckpt_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        config_path = Path(self.ckpt_dir) / config_name
        with open(config_path, 'x') as f:
            yaml.dump(self.conf, f, default_flow_style=False)

    def get_ckpts(self) -> List[Path]:
        ckpts = list(Path(self.ckpt_dir).glob(f"{self.name}_E*.pth"))
        def order(ckpt):
            return int(Path(ckpt).stem.split('E')[-1])
        return sorted(ckpts, key=order)

    def default_save_path(self, epoch: int) -> Path:
        """Get the default path to save the checkpoint."""
        return Path(self.ckpt_dir) / f"{self.name}_E{epoch}.pth"

    def clean_old_ckpts(self):
        ckpts = self.get_ckpts()
        if len(ckpts) > self.keep_num:
            for ckpt in ckpts[:-self.keep_num]:
                os.remove(ckpt)

    def load(self,
             ckpt_path: str):
        """Load checkpoint and return save config if exists.
           Notice: only overwrite the model, optimizer and scheduler, not the config."""
        # create the default path if not specified
        ckpt_path:Path = Path(ckpt_path)

        # check if the file exists
        if not ckpt_path.exists():
            raise FileNotFoundError(f"File {ckpt_path} does not exist.")

        # load checkpoint
        print(f'Loading checkpoint from {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model'])

        # load the best score and epoch
        self.best_result = checkpoint.get('result')
        if self.best_result is None:
            print(f"Cannot find result in {ckpt_path.stem}")
            print("Set best_result to {}.")
            self.best_result = {}

        self.epoch = checkpoint.get('epoch')
        if self.epoch is None:
            print(f"Cannot find epoch in {ckpt_path.stem}")
            print("Set epoch to 0.")
            self.epoch = 0

        # return the save config if exists
        return checkpoint.get('conf')

    def save(self,
             ckpt_path: Optional[str] = None,
             overwrite: bool = False):
        """Save the checkpoint."""
        # create the default path if not specified
        if ckpt_path is None:
            ckpt_path = self.default_save_path(epoch=self.epoch)
        else:
            ckpt_path = Path(ckpt_path)

        # check if the file exists and create the directory if not
        if ckpt_path.exists() and not overwrite:
            raise FileExistsError(f"File {ckpt_path} already exists.")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # save the checkpoint
        print(f'Saving model to {ckpt_path}')
        save_dict = {
            'model': self.model.state_dict(),
            'result': self.best_result,
            'epoch': self.epoch,
            'conf': self.conf
        }
        torch.save(save_dict, ckpt_path)

        # clean old checkpoints
        self.clean_old_ckpts()

    def update(self, result:dict, epoch: int, overwrite=False, **kwargs) -> bool:
        """save the checkpoint if the result is better."""
        for metric in self.check_metric:
            cur_metric = result.get(metric)
            assert cur_metric is not None, f"Cannot find {metric} in result"
            best_metric = self.best_result.get(metric)
            if best_metric is None or \
               self.save_mod == 'max' and cur_metric > best_metric or \
               self.save_mod == 'min' and cur_metric < best_metric:
                 self.best_result = result
                 self.epoch = epoch
                 self.save(overwrite=overwrite)
                 return True
        return False
