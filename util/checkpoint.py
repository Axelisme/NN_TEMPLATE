"""
Save and load checkpoints.
"""
import os
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional


def default_checkpoint(save_dir:str, model_name:str) -> str:
    """return the default path of the checkpoint"""
    return os.path.join(save_dir, model_name, f'checkpoint_{model_name}.pt')


def load_checkpoint(model:Module,
                    ckpt_path:str,
                    optim:Optional[Optimizer] = None,
                    scheduler:Optional[LRScheduler] = None,
                    device:Optional[torch.device] = None) -> None:
    """Load checkpoint."""
    # load model
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"File {ckpt_path} does not exist.")

    print(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if device is not None:
        model.to(device)
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])


def save_checkpoint(model:Module,
                    ckpt_path:str,
                    optim:Optional[Optimizer] = None,
                    scheduler:Optional[LRScheduler] = None,
                    overwrite:bool = False) -> None:
    """Save the checkpoint."""
    if os.path.exists(ckpt_path) and not overwrite:
        raise FileExistsError(f"File {ckpt_path} already exists.")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    print(f'Saving model to {ckpt_path}')
    save_dict = {}
    save_dict['model'] = model.state_dict()
    if optim is not None:
        save_dict['optimizer'] = optim.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    torch.save(save_dict, ckpt_path)
