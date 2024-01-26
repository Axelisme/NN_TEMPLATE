"""
optimize hyperparameters for the model using Weights and Biases
"""
import os
import sys
sys.path.append(os.getcwd())

import yaml
import wandb
import argparse
from functools import partial
from argparse import Namespace

from modules.runner import Runner
from util.utility import init


def train_for_sweep(args: Namespace):
    wandb.init()

    # fork is needed for sweep
    overwrite_conf = wandb.config

    if 'taskrc' not in overwrite_conf:
        overwrite_conf['taskrc'] = {}
    if 'WandB' not in overwrite_conf['taskrc']:
        overwrite_conf['taskrc']['WandB'] = {}
    overwrite_conf['taskrc']['WandB']['track_params'] = False # must be False
    overwrite_conf['taskrc']['WandB']['init_kwargs'] = {}

    runner = Runner(args, overwrite_conf, start_method='fork')
    runner.run()


def get_default_args(modelrc: str, taskrc: str) -> Namespace:
    # default args
    default_args = {
        'mode' : 'train',
        'ckpt' : None,
        'name' : 'sweep_run',
        'modelrc' : modelrc,
        'taskrc' : taskrc,
        'resume' : False,
        'ckpt_dir' : None,
        'train_loader' : 'train_loader',
        'valid_loader' : 'devel_loader',
        'device' : 'cuda:0',
        'slient' : False,
        'WandB' : True,
        'disable_save' : True,
        'overwrite' : False
    }

    default_args = Namespace(**default_args)

    return default_args


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('-M', '--modelrc', type=str, help='modelrc file, use to train a new model, default to configs/modelrc.yaml')
    parser.add_argument('-T', '--taskrc', type=str, help='taskrc file, default to configs/taskrc.yaml')
    parser.add_argument('-s', '--sweeprc', type=str, default='configs/sweeprc.yaml', help='path to sweep config file')
    args = parser.parse_args()

    with open(args.sweeprc, 'r') as f:
        sweeprc = yaml.load(f, Loader=yaml.FullLoader)

    train_fn = partial(train_for_sweep, args = get_default_args(args.modelrc, args.taskrc))

    sweep_id = wandb.sweep(sweeprc['config'], project=sweeprc['project'])
    wandb.agent(sweep_id, function=train_fn, count=sweeprc['num_trials'])


if __name__ == '__main__':
    main()