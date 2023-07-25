"""define hyperparameters for training and testing """

from config.configClass import Config

# load hyperparameters.yaml as Config object
all_conf = Config(yaml_path='hyperparameters.yaml')

# Path to the data directory
RAW_DATA_DIR       = all_conf.RAW_DATA_DIR
PROC_DATA_DIR      = all_conf.PROC_DATA_DIR
TRAIN_EX_DIR       = all_conf.TRAIN_EX_DIR
INFER_EX_DIR       = all_conf.INFER_EX_DIR
SAVED_MODELS_DIR   = all_conf.SAVED_MODELS_DIR
LOG_FILE           = all_conf.LOG_FILE

# create config for training and inference
base_conf  = Config(data=all_conf.base)
train_conf = Config(data=all_conf.train)
infer_conf = Config(data=all_conf.infer)
train_conf.load_config(base_conf)
infer_conf.load_config(base_conf)