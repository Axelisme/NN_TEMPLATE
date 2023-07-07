"""
Save and load checkpoints.
"""
import os
import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing import Optional


def default_checkpoint(save_dir:str, model_name:str) -> str:
    """return the default path of the checkpoint"""
    return os.path.join(save_dir, model_name, f'checkpoint_{model_name}.pt')


def load_checkpoint(model:Module,
                    optim:Optional[Optimizer] = None,
                    scheduler = None,
                    checkpoint_path:Optional[str] = None,
                    device:torch.device = torch.device("cuda")) -> int:
    """Load checkpoint."""
    # load model
    if checkpoint_path is None:
        print('No checkpoint loaded.')
        return 0
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"File {checkpoint_path} does not exist.")
    print(f'Loading checkpoint from {checkpoint_path}')
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return epoch


def save_checkpoint(epoch:int,
                    model:Module,
                    optim:Optional[Optimizer] = None,
                    scheduler = None,
                    checkpoint_path:Optional[str] = None,
                    overwrite:bool = False) -> None:
    """Save the checkpoint."""
    if checkpoint_path is None:
        print('No checkpoint saved.')
        return
    if os.path.exists(checkpoint_path) and not overwrite:
        raise FileExistsError(f"File {checkpoint_path} already exists.")
    dir = os.path.dirname(checkpoint_path)
    os.makedirs(dir, exist_ok=True)
    print(f'Saving model to {checkpoint_path}')
    model_stat = model.state_dict()
    optim_stat = optim.state_dict() if optim is not None else None
    sched_stat = scheduler.state_dict() if scheduler is not None else None
    torch.save({'epoch': epoch,
                'model': model_stat,
                'optimizer': optim_stat,
                "scheduler": sched_stat},
                  checkpoint_path)
