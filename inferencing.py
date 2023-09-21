import torch
from dataset.customDataset import CustomDataSet
from util.utility import init
from hyperparameters import infer_conf
from model.customModel import CustomModel
from config.configClass import Config
from ckptmanager.manager import CheckPointManager
from torch.utils.data import DataLoader

def infer(conf:Config, model:CustomModel, loader:DataLoader, device:torch.device) -> None:
    """Inferencing model on given dataset."""
    pass


def main(conf:Config) -> None:
    """Inferencing model base on given config."""

    # device setting
    device = torch.device(conf.device)

    # setup model and other components
    model = CustomModel(conf)                                                # create model
    model.eval()

    # load model from checkpoint if needed
    ckpt_manager = CheckPointManager(conf, model)
    if conf.Load:
        ckpt_manager.load(ckpt_path=conf.load_path, device=device)

    # prepare test dataset and dataloader
    test_transform = None
    test_dataset = CustomDataSet(conf, conf.test_dataset, transform=test_transform)  # create test dataset
    batch_size = conf.batch_size
    num_workers = conf.num_workers
    loader = DataLoader(dataset     = test_dataset,
                        batch_size  = batch_size,
                        shuffle     = True,
                        pin_memory  = True,
                        num_workers = num_workers)  # create train dataloader

    # start inference
    infer(conf, model, loader, device)


if __name__ == '__main__':
    #%% print version information
    print(f'Torch version: {torch.__version__}')
    # initialize
    init(infer_conf.seed)

    #%% run main function
    main(infer_conf)
