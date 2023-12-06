
"""A script to generate data for the project."""
import os
import numpy as np

from global_vars import RAW_DATA_DIR, PROC_DATA_DIR
from util.io import clear_folder
from util.datatools import generate_hdf5_data, sampling_hdf5_data


# define the data parameters
dataset_name = "template"
split_ratio = {
    "train": 0.8,
    "valid": 0.1,
    "test" : 0.1
}
data_dtype = np.dtype([("input", np.float32, (128,)), ("label", np.uint8)])
meta_dict = {
    "meta1": "123",
    "meta2": "456"
}
data_length = 10000


# define the data loader, should have your own custom data loader
def data_loader(_:int) -> np.ndarray:
    input = np.random.randn(128)
    x1 = input[4] > 0
    x2 = input[78] < 0
    x3 = input[115] > 0
    label = np.uint8(4 * x1 + 2 * x2 + x3)
    return np.array((input, label), dtype=data_dtype)

# define the sample saver, should have your own custom sample saver
def sample_saver(idx, sample, meta_dict):
    print(f"sampling {idx}")
    input, label = sample
    print(f"label: {label}")
    if idx % 5 == 0:
        print(f"meta1: {meta_dict['meta1']}")
        print(f"meta2: {meta_dict['meta2']}")


def main():
    # create the folder to save the generated data
    SAVE_DIR = os.path.join(PROC_DATA_DIR, dataset_name)
    clear_folder(SAVE_DIR) # clear the folder before generating data

    # generate the dataset and save to hdf5 file
    for mode, ratio in split_ratio.items():
        DATASET_PATH = os.path.join(SAVE_DIR, f"{mode}.h5")
        mode_length = int(data_length * ratio)

        print(f"Writting {mode} dataset with length {mode_length} to {DATASET_PATH}")
        generate_hdf5_data(DATASET_PATH, data_dtype, data_loader, mode_length, meta_dict=meta_dict)

    # sampling some samples from the hdf5 dataset
    for mode in split_ratio.keys():
        dataset_path = os.path.join(SAVE_DIR, f"{mode}.h5")
        sampling_hdf5_data(dataset_path, sample_saver)

if __name__ == "__main__":
    main()
