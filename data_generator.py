
"""A script to generate data for the project."""
#%%
import os
import h5py
import numpy as np
from typing import Dict, List
from torch.utils.data import random_split, RandomSampler, BatchSampler
from util.io import clear_folder
from hyperparameters import base_conf, PROC_DATA_DIR

# some parameters
dataset_name = base_conf.dataset_name
split_ratio = base_conf.split_ratio
SAVE_DIR = os.path.join(PROC_DATA_DIR, os.path.splitext(dataset_name)[0])
clear_folder(SAVE_DIR) # clear the folder before generating data


#%%
def generate_process_data(save_dir:str, dataset_name:str) -> None:
    """
    Generate whole processed data for the project.
    """
    # create folder for processed data
    ALL_DATASET_PATH = os.path.join(save_dir, "all", dataset_name)
    os.makedirs(os.path.dirname(ALL_DATASET_PATH), exist_ok=True)

    # load data
    data_dtype = np.dtype([("input", np.uint32, (2,)), ("label", np.uint8)])  # TODO: the data format in the dataset
    datas = [([i,10*i],100*i) for i in range(10)] #TODO load data
    data_length = len(datas)

    # save data to h5 file
    print(f"Writting total dataset to {ALL_DATASET_PATH}")
    with h5py.File(ALL_DATASET_PATH, mode='w') as writer:
        # write meta data
        writer.attrs["name"] = "all"
        writer.attrs["length"] = data_length
        writer.attrs["ratio"] = 1.0
        writer.attrs["data_dtype"] = str(data_dtype)
        # write dataset
        dataset = writer.create_dataset("dataset", (data_length,), dtype=data_dtype)
        for idx, data_i in enumerate(datas):
            dataset[idx] = data_i

generate_process_data(SAVE_DIR, dataset_name)


#%%
def split_process_data(save_dir:str, dataset_name:str, split_ratio:Dict[str,float]) -> None:
    """
    Split the whole processed data into some dataset.
    """
    ALL_DATASET_PATH = os.path.join(save_dir, "all", dataset_name)

    with h5py.File(ALL_DATASET_PATH, mode='r') as reader:
        # load data
        dataset = reader["dataset"]
        # split dataset
        splited_datasets = random_split(dataset, list(split_ratio.values())) # type: ignore

        # save splited dataset
        for (name, ratio), named_dataset in zip(split_ratio.items(), splited_datasets):
            NAMED_DATASET_PATH = os.path.join(save_dir, name, dataset_name)
            os.makedirs(os.path.dirname(NAMED_DATASET_PATH), exist_ok=True)
            print(f"Writting {name} dataset to {NAMED_DATASET_PATH}")

            # save data to h5 file
            with h5py.File(NAMED_DATASET_PATH, mode='w') as writer:
                # write meta data
                writer.attrs["name"] = name
                writer.attrs["length"] = len(named_dataset)
                writer.attrs["ratio"] = ratio
                writer.attrs["data_dtype"] = reader.attrs["data_dtype"]
                print(reader.attrs["data_dtype"])
                # write dataset
                dataset = writer.create_dataset("dataset", data=named_dataset)

split_process_data(SAVE_DIR, dataset_name, split_ratio)


#%%
def sampling_process_samples(save_dir, dataset_name:str, data_name:List[str], num = 100):
    """
    Sampling some samples from the processed data.
    """
    for name in data_name:
        DATASER_PATH = os.path.join(save_dir, name, dataset_name)
        SAMPLES_DIR = os.path.join(save_dir, name, "samples")
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        with h5py.File(DATASER_PATH, mode='r') as reader:
            # load data
            dataset = reader["dataset"]
            # sampling
            sampler = RandomSampler(dataset, replacement=False) # type: ignore
            batch_sampler = BatchSampler(sampler, batch_size=num, drop_last=False)
            batch = next(iter(batch_sampler))
            # save samples
            for idx, sample in enumerate(batch):
                print(f"Saving sample {idx}")
                #TODO save sample

sampling_process_samples(SAVE_DIR, dataset_name, list(split_ratio.keys()), num=100)
