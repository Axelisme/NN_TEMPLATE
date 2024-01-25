"""
define a class to manage the checkpoint
"""
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch import Tensor

from util.io import show


class CheckPointManager:
    def __init__(self, ckpt_dir: str):
        show(f"[CheckpointManager] Using {ckpt_dir} as checkpoint directory")
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # default setting
        self.save_mod = 'max'
        self.check_metrics = []
        self.keep_num = -1


    def set(self, ckpt_conf: Dict[str, Dict|Any]):
        """set the action of checkpoint manager"""
        self.check_metrics = ckpt_conf['check_metrics']
        assert isinstance(self.check_metrics, list), f"check_metrics should be list, but got {self.check_metrics}"
        names = [check_metric['name'] for check_metric in self.check_metrics]
        show(f"[CheckpointManager] Using {names} as check_metrics")

        self.keep_num = ckpt_conf['keep_num']
        assert isinstance(self.keep_num, int), f"keep_num should be int, but got {self.keep_num}"
        assert self.keep_num >= -1, f"keep_num should be greater than -1, but got {self.keep_num}"
        show(f"[CheckpointManager] Keep {self.keep_num} checkpoints in ckpt_dir")


    def check_better(self, current_result, best_result) -> bool:
        for check_metric in self.check_metrics:
            name = check_metric['name']
            mod = check_metric['mod']
            if current_metric := current_result.get(name):
                if best_metric := best_result.get(name):
                    if current_metric == best_metric:
                        continue
                    return mod == 'max' and current_metric > best_metric or \
                            mod == 'min' and current_metric < best_metric
                else:
                    return True
            else:
                continue
        return False


    def dump_config(self, save_conf: Dict, file_name: str):
        """Save the config file."""
        config_path = self.ckpt_dir / file_name
        show(f'[CheckpointManager] Dumping config to {config_path}')
        with open(config_path, 'x') as f:
            yaml.dump(save_conf, f, default_flow_style=False)


    def load_config(self, ckpt_path: str, conf_name: str):
        # use mmap to save memory
        ckpt = torch.load(ckpt_path, map_location='cpu', mmap=True)
        show(f'[CheckpointManager] Loading config {conf_name} from {ckpt_path}')
        return ckpt['config'][conf_name]


    def load_param(self, ckpt_path: str, param_name: str):
        # use mmap to save memory
        ckpt = torch.load(ckpt_path, map_location='cpu', mmap=True)
        return ckpt['params'][param_name]


    def save_model(
        self,
        ckpt_name: str,
        params: Dict[str, Tensor],
        confs: Dict[str, Any],
        overwrite: bool = False):
        """Save the checkpoint."""
        # create the default path if not specified
        ckpt_path = self.ckpt_dir / ckpt_name

        # check if the file exists
        if ckpt_path.exists() and not overwrite:
            raise FileExistsError(f"File {ckpt_path} already exists.")

        # save the checkpoint
        show(f'[CheckpointManager] Saving model to {ckpt_path}')
        save_dict = {'params': params, 'config': confs}
        torch.save(save_dict, ckpt_path)


    def clean_old_ckpt(self, pattern: str):
        if self.keep_num == -1: # keep all ckpts
            return
        ckpts = list(self.ckpt_dir.glob(pattern))
        ckpts = sorted(ckpts, key=lambda ckpt: int(Path(ckpt).stem.split('-')[-1]))
        if self.keep_num != -1 and len(ckpts) > self.keep_num:
            for ckpt in ckpts[:-self.keep_num]:
                os.remove(ckpt)