
"""define hyperparameters for training and testing """

from config.configClass import Config
import torch
from os import path

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))           # Path to the workspace of the project
DATA_DIR = path.join(ROOT_DIR, 'data')                                  # Path to the data directory
RAW_DATA_DIR = path.join(DATA_DIR, 'rawdata')                           # Path to the raw data directory
PROCESSED_DATA_DIR = path.join(DATA_DIR, 'processed')                   # Path to the processed data directory
TRAIN_EX_DIR = path.join(DATA_DIR, 'training_example')                  # Path to the training examples directory
INFER_EX_DIR = path.join(DATA_DIR, 'inference_examples')                # Path to the inference examples directory
SAVED_MODELS_DIR = path.join(DATA_DIR, 'saved_models')                  # Path to the saved models directory
MISPRED_DIR = path.join(DATA_DIR, 'mispredict')                         # Path to the mispredictions directory

# create config
base_config = Config(
    project_name = 'NN_TEMPLATE',
    model_name = 'model_1',
    seed = 0,
    input_size = (3, 224, 224),
    output_size = 1,
)

train_config = Config(
    batch_size = 8,
    epochs = 20,
    lr = 0.00001,
    weight_decay = 0.7,
    split_ratio = 0.2,
    device = torch.device('cuda' if torch.cuda.is_available() else 'none'),
    SAVE = False,
    WandB = False,
)

infer_config = Config(
    device = torch.device('cpu'),
    split_ratio = 0.2,
    batch_size = 1,
)

train_config.update(base_config)
infer_config.update(base_config)