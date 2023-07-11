import os
import numpy as np
from tqdm.auto import tqdm
import torch
from torchvision import transforms
from dataset.customDataset import CustomDataSet
from util.utility import init
from util.io import logit
from util.checkpoint import load_checkpoint
from hyperparameters import *
from model.customModel import CustomModel
from config.configClass import Config


@logit(LOG_FILE)
def main(conf:Config) -> None:
    """Main function of the script."""

    # create model and load
    model = CustomModel(conf)                                                # create model
    load_checkpoint(model, checkpoint_path=conf.load_path)                   # load model
    model.eval().to(torch.device(conf.device))                               # set model to eval mode

    # prepare test dataset and dataloader
    test_dataset = CustomDataSet(conf, "test", dataset_name=conf.dataset_name)    # create test dataset

    # print label names
    for id,name in enumerate(test_dataset.label_names):
        print(f"{id}:{name}", end=' ')
    print()

    # start inference
    with torch.no_grad():
        err_count = 0
        total_count = len(test_dataset) #type:ignore
        for input, label  in tqdm(test_dataset, desc="Inferencing"): #type:ignore
            output = model(input.to(conf.device).unsqueeze(0)).squeeze(0).cpu()
            pred = output.argmax(dim=0).item()
            if label != pred:
                err_count += 1
        print(f"Model: {conf.model_name}, dataset: {conf.dataset_name}")
        print(f"Error: {err_count}/{total_count} = {err_count/total_count*100:0.4f}%")


if __name__ == '__main__':
    # print version information
    print(f'Torch version: {torch.__version__}')
    init(infer_conf.seed)

    # run main function
    main(infer_conf)
