
"""A script to generate data for the project."""
#%%
import os
import h5py
import numpy as np
import random
from typing import Callable
from util.io import clear_folder
from hyperparameters import base_conf, RAW_DATA_DIR, PROC_DATA_DIR
import multiprocessing as mp

# some parameters
dataset_name = "template"
split_ratio = base_conf.split_ratio
SAVE_DIR = os.path.join(PROC_DATA_DIR, dataset_name)
clear_folder(SAVE_DIR) # clear the folder before generating data


#%%
def generate_process_data(dataset_path:str,
                          data_dtype:np.dtype,
                          data_loader:Callable,
                          mode_length:int = 1000,
                          max_batch_num:int = 100) -> None:
    """
    Generate processed data for the project.
    """
    # save data to h5 file
    with mp.Pool(mp.cpu_count()) as pool:
        with h5py.File(dataset_path, mode='x') as writer:
            # write meta data
            writer.attrs["length"] = mode_length
            writer.attrs["data_dtype"] = str(data_dtype)
            # create dataset
            dataset = writer.create_dataset("dataset", (mode_length,), dtype=data_dtype)
            # write dataset
            saved_num = 0
            while saved_num < mode_length:
                batch_num = min(max_batch_num, mode_length - saved_num)
                save_ids = list(range(saved_num, saved_num + batch_num))
                batch = pool.imap_unordered(data_loader, save_ids)
                for idx, data in zip(save_ids, batch):
                    dataset[idx] = data
                saved_num += batch_num

def data_loader(idx:int) -> np.ndarray:
    return np.random.randn(1,80,80), np.random.randint(0,10)

data_dtype = np.dtype([("input", np.float32, (1,80,80)), ("label", np.uint8)])  # TODO: the data format in the dataset
data_length = 1000
for mode, ratio in split_ratio.items():
    DATASET_PATH = os.path.join(SAVE_DIR, f"{mode}.h5")
    mode_length = int(data_length * ratio)

    print(f"Writting {mode} dataset with length {mode_length} to {DATASET_PATH}")
    generate_process_data(DATASET_PATH, data_dtype, data_loader, mode_length)


#%%
def sampling_process_samples(dataset_path:str,
                             sample_saver:Callable,
                             max_num:int = 20,
                             *args, **kwargs) -> None:
    """
    Sampling some samples from the processed data.
    """
    with h5py.File(dataset_path, mode='r') as reader:
        # load data
        dataset = reader["dataset"]
        data_length = reader.attrs["length"]
        # sampling
        sample_ids = random.sample(range(data_length), max_num)
        # save samples
        for idx, sample_id in enumerate(sample_ids):
            print(f"Saving sample {idx}")
            sample = dataset[sample_id]
            sample_saver(sample, *args, **kwargs)

def sample_saver(sample):
    pass

for mode in split_ratio.keys():
    dataset_path = os.path.join(SAVE_DIR, f"{mode}.h5")
    sampling_process_samples(dataset_path, sample_saver)
