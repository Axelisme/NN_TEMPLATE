
"""A script to run inference on a trained model."""

import os
from sys import argv
from typing import Any
import util.utility as ul
from model.custom_model import Model
from dataset.dataset import DataSet
from config.configClass import Config
import hyperparameter as p
from tester.tester import Tester
import torch
from torch import nn
from torch.utils.data import DataLoader

load_path = os.path.join(p.SAVED_MODELS_DIR, f"{p.infer_config.model_name}.pt")

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

    # TODO: inference
    pass


if __name__ == '__main__':
    # print version information
    print(f'Torch version: {torch.__version__}')

    # initialize
    ul.init(p.infer_config)

    # run main function
    main(p.infer_config, load_path)