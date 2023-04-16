
"""A script to run inference on a trained model."""

from sys import argv
import global_var.path as p
import util.utility as ul
from model.model import Model
from dataset.dataset import DataSet
from config.config import Config
from tester.tester import Tester
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics.classification as cf

# create config
config = Config(
    project_name = 'NN_Template',
    model_name = 'NN_test',
    seed = 0,
    input_size = 8,
    output_size = 10,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    batch_size = 16,
    split_ratio = 0.2
)

def init(config:Config):
    """Initialize the script."""
    # create directory
    os.makedirs(p.SAVED_MODELS_DIR, exist_ok=True)
    # set float32 matmul precision
    torch.set_float32_matmul_precision('high')
    # set random seed
    ul.set_seed(seed=config.seed)

def show_result(config:Config, valid_result:dict):
    """Print result of training and validation."""
    # print result
    print("Valid result:")
    for name,score in valid_result.items():
        print(f'\t{name}: {score}')

def main(config:Config, model_path:str) -> None:
    """Main function of the script."""

    # create model and load
    model = Model(config)                                                      # create model
    model.load_state_dict(torch.load(model_path, map_location=config.device))  # load model
    evaluator = cf.MulticlassAccuracy(num_classes=config.output_size,average='micro', validate_args=True)  # create evaluator

    # prepare dataset and dataloader
    DataSet.load_data(config)           # initialize dataset
    test_set:DataSet = DataSet("test")  # create test dataset
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, pin_memory=True)  # create test dataloader

    # create trainer
    tester = Tester(model=model, config=config, test_loader=test_loader)
    tester.add_evaluator(evaluator)

    # start testing
    result:dict = tester.eval()

    # show result
    show_result(config, result)

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