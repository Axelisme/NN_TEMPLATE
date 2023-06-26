"""define hyperparameters for training and testing """

#%%
from os import path
import util.utility as ul
from config.configClass import Config

ROOT_DIR           = path.dirname(path.abspath(__file__))                 # Path to the workspace of the project
DATA_DIR           = path.join(ROOT_DIR, 'data')                          # Path to the data directory
RAW_DATA_DIR       = path.join(DATA_DIR, 'rawdata')                       # Path to the raw data directory
PROC_DATA_DIR      = path.join(DATA_DIR, 'processed')                     # Path to the processed data directory
TRAIN_EX_DIR       = path.join(DATA_DIR, 'training_data')                 # Path to the training examples directory
INFER_EX_DIR       = path.join(DATA_DIR, 'inference_data')                # Path to the inference examples directory
SAVED_MODELS_DIR   = path.join(DATA_DIR, 'saved_models')                  # Path to the saved models directory
LOG_CONSOLE        = path.join(DATA_DIR, 'log.txt')                       # Path to the log file

# create config
base_config = Config(
    project_name = 'Template',
    model_name = 'Version_1',
    seed = 0,
    data_ratio = (0.8, 0.1, 0.1),  #train, val, test
    input_size = (3, 224, 224),    #channel, height, width
    output_size = 8,
)
base_config.load_path = ul.default_checkpoint(SAVED_MODELS_DIR, base_config.model_name)
base_config.save_path = ul.default_checkpoint(SAVED_MODELS_DIR, base_config.model_name)