"""
define a class to manage the checkpoint
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
from torch import Tensor

from util.io import show


class CheckPointManager:
    def __init__(
            self,
            ckpt_dir: str,
            check_metrics: list,
            check_datasets: list,
            keep_num: int = -1,
            metric_compare_first: bool = True
        ):
        show(f"[CheckpointManager] Using {ckpt_dir} as checkpoint directory")
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # check
        assert len(check_metrics) > 0, "check_metrics must be non-empty"
        assert len(check_datasets) > 0, "check_datasets must be non-empty"

        # pair the check metrics and datasets
        self.pair_metrics = []
        if metric_compare_first:
            for check_metric in check_metrics:
                for check_dataset in check_datasets:
                    self.pair_metrics.append((check_metric, check_dataset))
        else:
            for check_dataset in check_datasets:
                for check_metric in check_metrics:
                    self.pair_metrics.append((check_metric, check_dataset))

        # show the check pairs
        show(f"[CheckpointManager] Checkpoint save policy:")
        for i, (metric, dataset) in enumerate(self.pair_metrics, start=1):
            show(f"\t{i}. {dataset}-{metric['name']}")

        if keep_num == -1:
            show(f"[CheckpointManager] Keep all checkpoints in ckpt_dir")
        elif keep_num >= 0:
            show(f"[CheckpointManager] Keep at most {keep_num} checkpoints in ckpt_dir")
        else:
            raise ValueError(f"keep_num must be non-negative, but got {keep_num}")
        self.keep_num = keep_num


    def check_better(self, current_results: Dict[str, Dict[str, float]], best_results: Dict[str, Dict[str, float]]):
        for metric, dataset in self.pair_metrics:
            name = metric['name']
            mod = metric['mod']
            if current_metric := current_results.get(dataset, {}).get(name):
                if best_metric := best_results.get(dataset, {}).get(name):
                    if current_metric == best_metric:
                        continue
                    elif mod == 'max':
                        return current_metric > best_metric
                    elif mod == 'min':
                        return current_metric < best_metric
                    else:
                        raise ValueError(f"Unknown mod {mod}")
                else:
                    return True
            else:
                raise ValueError(f"Metric {name} not found in current_results")

    @classmethod
    def dump_config(cls, save_conf: Dict, path: str, overwrite: bool = False):
        """Save the config file."""
        show(f'[CheckpointManager] Dumping config to {path}')
        with open(path, 'w' if overwrite else 'x') as f:
            yaml.dump(save_conf, f, default_flow_style=False)

    @classmethod
    def load_config(cls, ckpt_path: str, conf_name: str):
        # use mmap to save memory
        ckpt = torch.load(ckpt_path, map_location='cpu', mmap=True)
        show(f'[CheckpointManager] Loading config {conf_name} from {ckpt_path}')
        return ckpt['config'][conf_name]

    @classmethod
    def load_param(cls, ckpt_path: str, param_name: str):
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