"""define hyperparameters for training and testing """

#%%
from os import path
from config.configClass import Config


ROOT_DIR           = path.dirname(path.abspath(__file__))                 # Path to the workspace of the project
DATA_DIR           = path.join(ROOT_DIR, 'data')                          # Path to the data directory
RAW_DATA_DIR       = path.join(DATA_DIR, 'rawdata')                       # Path to the raw data directory
PROC_DATA_DIR      = path.join(DATA_DIR, 'processed')                     # Path to the processed data directory
TRAIN_EX_DIR       = path.join(DATA_DIR, 'training_data')                 # Path to the training examples directory
INFER_EX_DIR       = path.join(DATA_DIR, 'inference_data')                # Path to the inference examples directory
SAVED_MODELS_DIR   = path.join(DATA_DIR, 'saved_models')                  # Path to the saved models directory
LOG_FILE           = path.join(DATA_DIR, 'log.txt')                       # Path to the log file


# create config
all_conf = Config(yaml_path='hyperparameters.yaml')
base_conf  = Config(data=all_conf.base)
train_conf = Config(data=all_conf.train)
infer_conf = Config(data=all_conf.infer)

train_conf.load_config(base_conf)
infer_conf.load_config(base_conf)