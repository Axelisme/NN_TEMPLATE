import time
import torch
from torch import Tensor
from dataset.customDataset import CustomDataSet
from util.utility import init
from hyperparameters import infer_conf
from model.customModel import CustomModel
from config.configClass import Config
from ckptmanager.manager import CheckPointManager
from torch.utils.data import Dataset

def infer(conf:Config, model:CustomModel, dataset:Dataset, device:torch.device):
    """Inferencing model on given dataset."""
    input, label = dataset[0]
    input = torch.tensor(input).to(device).unsqueeze(0)
    label = torch.tensor(label).to(device).unsqueeze(0)
    output:Tensor = model(input)
    print(f"input shape: {input.shape}")
    print(f"label shape: {label.shape}")
    print(f"output shape: {output.shape}")
    print("output 0:")
    print(output[0])


def main(conf:Config):
    """Inferencing model base on given config."""

    # device setting
    device = torch.device(conf.device)

    # setup model and other components
    model = CustomModel(conf)                                                # create model
    model.eval()

    # load model from checkpoint if needed
    ckpt_manager = CheckPointManager(conf, model)
    ckpt_manager.save_config(f"infer_{time.strftime('%Y%m%d_%H%M%S')}.yaml")
    if conf.Load:
        ckpt_manager.load(ckpt_path=conf.load_path, device=device)

    # prepare test dataset and dataloader
    test_transform = None
    test_dataset = CustomDataSet(conf, conf.test_dataset, transform=test_transform)  # create test dataset

    # start inference
    infer(conf, model, test_dataset, device)


if __name__ == '__main__':
    #%% print version information
    print(f'Torch version: {torch.__version__}')
    # initialize
    init(infer_conf.seed)

    #%% run main function
    main(infer_conf)
