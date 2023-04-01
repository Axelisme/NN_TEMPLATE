
"""A script to run inference on a trained model."""

from sys import argv
import config.path as p
import util.utility as ul
from model.model import Model
from dataset.dataset import Dataset
from inferencer.inferencer import Inferencer
from evaluator.fl_score import F1Score
import torch
from torch import nn
from torch.utils.data import DataLoader

def main(model_path: str) -> None:
    """main function of inference.py"""
    # load the model
    model:Model = Model()
    model.load_state_dict(torch.load(model_path))

    # create variables for the inferencer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    score = F1Score()

    # create data loader
    test_set = Dataset("test")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    # create inferencer
    inferencer = Inferencer(model, device, test_loader, score)

    # evaluate the model
    inferencer.evaluate()

if __name__ == '__main__':
    main(argv[1])