"""
optimize hyperparameters for the model using Weights and Biases
"""
import yaml
import wandb
import argparse
from copy import deepcopy

from scripts.run import start_train
from util.utility import init
from util.io import show


default_args = argparse.Namespace()
default_args.name = 'sweep'
default_args.mode = 'train'
default_args.seed = 0
default_args.train_loader = 'train_loader'
default_args.valid_loader = 'valid_loader'
default_args.load = None
default_args.device = 'cuda:0'
default_args.slient = False

default_args.WandB = True
default_args.not_track_params = True
default_args.disable_save = True
default_args.overwrite = False

default_conf = None


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

def train_for_sweep():
    wandb.init()

    global default_args, default_conf
    args = deepcopy(default_args)
    conf = deepcopy(default_conf)
    override_conf(conf, wandb.config)

    show.set_slient(args.slient)

    init(seed=args.seed, start_method='fork')
    start_train(args, conf)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('-c', '--config', type=str, default='configs/template.yaml', help='path to config file')
    args = parser.parse_args()

    # load config
    global default_conf
    with open(args.config, 'r') as f:
        default_conf = yaml.load(f, Loader=yaml.Loader)

    sweep_conf = default_conf['WandB']['sweep']
    sweep_id = wandb.sweep(sweep_conf['config'], project=default_conf['WandB']['project'])
    wandb.agent(sweep_id, function=train_for_sweep, count=sweep_conf['num_trials'])


if __name__ == '__main__':
    main()