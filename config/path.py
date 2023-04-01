
"""path to different directories"""

from os import path

# Path to the root of the project
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
# Path to the data directory
DATA_DIR = path.join(ROOT_DIR, '/data')
# Path to the saved models directory
SAVED_MODELS_DIR = path.join(DATA_DIR, '/saved_models')
# Path to the inference results directory
INFERENCE_RESULTS_DIR = path.join(DATA_DIR, '/inference_results')
# Path to the raw data directory
RAW_DATA_DIR = path.join(DATA_DIR, '/raw_data')
# Path to the processed data directorydsaf
PROCESSED_DATA_DIR = path.join(DATA_DIR, '/processed_data')
