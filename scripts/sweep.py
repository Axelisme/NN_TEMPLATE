"""
optimize hyperparameters for the model using Weights and Biases
"""
import yaml
import wandb
import argparse
from copy import deepcopy
from functools import partial
from typing import Dict

from scripts.run import start_train
from util.utility import init
from util.io import show


def override_conf(conf, new_conf):
    for key, value in new_conf.items():
        if isinstance(value, dict):
            if key not in conf:
                print(f"something wrong with {key}")
                raise ValueError
            override_conf(conf[key], value)
        else:
            print(f'override {key} from {conf[key]} to {value}')
            conf[key] = value


def train_for_sweep(default_setup:Dict):
    wandb.init()

    setup = deepcopy(default_setup)
    override_conf(setup, wandb.config)

    args = argparse.Namespace(**setup['args'])
    conf = setup['conf']

    show.set_slient(args.slient)

    init(seed=args.seed, start_method='fork')
    start_train(args, conf)


def get_default_setup(conf_path:str):
    # load config
    with open(conf_path, 'r') as f:
        default_conf = yaml.load(f, Loader=yaml.Loader)

    # default args
    default_args = {
        'name': 'sweep',
        'mode': 'train',
        'seed': 0,
        'train_loader': 'train_loader',
        'valid_loader': 'valid_loader',
        'load': None,
        'device': 'cuda:0',
        'slient': False,
        'WandB': True,
        'not_track_params': True,
        'disable_save': True,
        'overwrite': False
    }

    return {'args': default_args, 'conf': default_conf}


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('-c', '--default_config', type=str, default='configs/template.yaml', help='path to train config file')
    parser.add_argument('-s', '--sweep_config', type=str, default='configs/sweep.yaml', help='path to sweep config file')
    args = parser.parse_args()

    with open(args.sweep_config, 'r') as f:
        sweep_conf = yaml.load(f, Loader=yaml.Loader)

    default_setup = get_default_setup(args.default_config)

    train_fn = partial(train_for_sweep, default_setup=default_setup)

    sweep_id = wandb.sweep(sweep_conf['config'], project=sweep_conf['project'])
    wandb.agent(sweep_id, function=train_fn, count=sweep_conf['num_trials'])


if __name__ == '__main__':
    main()