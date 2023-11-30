
"""A script to generate data for the project."""
import os
import numpy as np

from util.io import clear_folder
from util.datatools import generate_hdf5_data, sampling_hdf5_data
from global_vars import RAW_DATA_DIR, PROC_DATA_DIR

dataset_name = "template"
SAVE_DIR = os.path.join(PROC_DATA_DIR, dataset_name)
clear_folder(SAVE_DIR) # clear the folder before generating data


# some parameters
split_ratio = {
                "train": 0.8,
                "valid": 0.1,
                "test": 0.1
            }
data_dtype = np.dtype([("input", np.float32, (128,)), ("label", np.uint8)])
data_length = 10000


# define the data loader
def data_loader(_:int) -> np.ndarray:
    input = np.random.randn(128)
    label = input.sum().astype(np.uint8) % 10 # just sum over the input and mod 10
    return np.array((input, label), dtype=data_dtype)

# generate the dataset and save to hdf5 file
for mode, ratio in split_ratio.items():
    DATASET_PATH = os.path.join(SAVE_DIR, f"{mode}.h5")
    mode_length = int(data_length * ratio)
    meta_dict = {
        "meta1": "123",
        "meta2": "456"
    }

    print(f"Writting {mode} dataset with length {mode_length} to {DATASET_PATH}")
    generate_hdf5_data(DATASET_PATH, data_dtype, data_loader, mode_length, meta_dict=meta_dict)

# define the sampler
def sample_saver(idx, sample, meta_dict):
    print(f"sampling {idx}")
    input, label = sample
    print(f"label: {label}")
    if idx % 5 == 0:
        print(f"meta1: {meta_dict['meta1']}")
        print(f"meta2: {meta_dict['meta2']}")

# sampling some samples from the hdf5 dataset
for mode in split_ratio.keys():
    dataset_path = os.path.join(SAVE_DIR, f"{mode}.h5")
    sampling_hdf5_data(dataset_path, sample_saver)
