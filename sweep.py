"""
optimize hyperparameters for the model using Weights and Biases
"""
import wandb
from copy import deepcopy
from training import start_train
from hyperparameters import sweep_conf, train_conf
from util.utility import init

project_name = sweep_conf.project_name
sweep_name = sweep_conf.name
sweep_parameters = sweep_conf.config
num_trials = sweep_conf.num_trials

def train_for_sweep():
    wandb.init(name=sweep_name)

    conf = deepcopy(train_conf)
    conf.Save = False
    conf.Load = False
    conf.WandB = True
    conf.Sweep = True

    for key, value in wandb.config.items():
        setattr(conf, key, value)

    init(seed=conf.seed, start_method='fork')
    start_train(conf)

sweep_id = wandb.sweep(sweep_parameters, project=project_name)
wandb.agent(sweep_id, function=train_for_sweep, count=num_trials)