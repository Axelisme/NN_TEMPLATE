"""
optimize hyperparameters for the model using Weights and Biases
"""

import copy
import wandb
import training as tr
from hyperparameters import train_conf
from util.utility import init

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {'value': 10},
        'seed': {'values': [0, 1]},
        'batch_size': {'min': 40, 'max': 60, 'distribution': 'int_uniform'},
        'init_lr': {'min': 9e-5, 'max': 1.3e-4, 'distribution': 'uniform'},
    }
}

def train_for_sweep():
    wandb.init()

    sweep_conf = copy.deepcopy(train_conf)
    sweep_conf.Save = False
    sweep_conf.Load = False
    sweep_conf.WandB = True
    sweep_conf.Sweep = True

    for key, value in wandb.config.items():
        setattr(sweep_conf, key, value)

    init(seed=sweep_conf.seed, start_method='fork')
    tr.start_train(sweep_conf)

sweep_id = wandb.sweep(sweep_config, project=train_conf.project_name)
wandb.agent(sweep_id, function=train_for_sweep, count=10)