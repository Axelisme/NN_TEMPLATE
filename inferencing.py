
"""A script to run inference on a trained model."""

import os
from sys import argv
from typing import Any
import global_var.path as p
import util.utility as ul
from model.model import Model
from dataset.dataset import DataSet
from config.configClass import Config
from config.hyperparameter import config
from tester.tester import Tester
import torch
from torch import nn
from torch.utils.data import DataLoader

def init(config:Config):
    """Initialize the script."""
    # create directory
    os.makedirs(p.SAVED_MODELS_DIR, exist_ok=True)
    # set float32 matmul precision
    torch.set_float32_matmul_precision('high')
    # set random seed
    ul.set_seed(seed=config.seed)

def main(config:Config, model_path:str) -> None:
    """Main function of the script."""

    # create model and load
    model = Model(config)                                                      # create model
    model.load_state_dict(torch.load(model_path, map_location=config.device))  # load model
    model.eval()                                                               # set model to eval mode

    # prepare dataset and dataloader
    DataSet.load_data(config)           # initialize dataset
    test_set:DataSet = DataSet("test")  # create test dataset
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True) # create test dataloader

    pass


if __name__ == '__main__':
    # print version information
    print(f'Torch version: {torch.__version__}')

    # find the save model path
    save_model_path = ''
    if len(argv) == 2:
        save_model_path = argv[1]
    else:
        import os
        all_model = os.listdir(p.SAVED_MODELS_DIR)
        if '.gitkeep' in all_model:
            all_model.remove('.gitkeep')
        if len(all_model) == 0:
            raise Exception("No model found in saved_models directory.")
        save_model_path = os.path.join(p.SAVED_MODELS_DIR,all_model[0]) 

    # initialize
    init(config)

    # run main function
    main(config, save_model_path)