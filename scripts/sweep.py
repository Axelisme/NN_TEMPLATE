"""
optimize hyperparameters for the model using Weights and Biases
"""
import yaml
import wandb
import argparse

from scripts.training import start_train
from util.utility import init
from util.io import show

conf = None

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

    global conf
    override_conf(conf, wandb.config)

    args = argparse.Namespace()
    args.load = None
    args.disable_save = True
    args.WandB = True
    args.not_track_params = True
    args.slient = True
    args.device = 'cuda:0'

    show.set_slient(args.slient)

    init(seed=0, start_method='fork')
    start_train(args, conf)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('-c', '--config', type=str, default='config/template.yaml', help='path to config file')
    args = parser.parse_args()

    # load config
    global conf
    with open(args.config, 'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    sweep_conf = conf['WandB']['sweep']
    sweep_id = wandb.sweep(sweep_conf['config'], project=conf['WandB']['project'])
    wandb.agent(sweep_id, function=train_for_sweep, count=sweep_conf['num_trials'])


if __name__ == '__main__':
    main()