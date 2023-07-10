
"""A script to generate data for the project."""
#%%
import os
import h5py
import numpy as np
import torch.utils.data as data
import multiprocessing as mp
from util.io import logit
from util.tool import measure_time, show_time, clear_folder, load_subfolder_as_label
from hyperparameter import *


raw_data_name = "raw_data_name"
dataset_name = "dataset.hdf5"
data_types = ["train", "valid", "test"]
data_ratios = base_conf.data_ratio

mydtype = np.dtype([("input", np.uint8, (1,)), ("label", np.uint8)])


def data_loader(path, label) -> np.ndarray:
    return np.zeros((1,1), dtype=mydtype)


#%%
@logit(LOG_FILE)
@measure_time
def generate_process_data():
    RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, raw_data_name)
    all_paths_labels, label_names = load_subfolder_as_label(RAW_DATA_PATH, max_num=100000)
    types_paths_labels = data.random_split(all_paths_labels, data_ratios) #type:ignore
    for type, type_paths_labels in zip(data_types, types_paths_labels):
        print(f"Loading {type} data from raw datas")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            inputs_labels = pool.starmap(data_loader, type_paths_labels) #type:ignore
        TYPE_DIR = os.path.join(PROC_DATA_DIR, type)
        DATASET_PATH = os.path.join(TYPE_DIR, dataset_name)
        print(f"Saving {type} data to {TYPE_DIR}......", end="  ")
        clear_folder(TYPE_DIR)
        with h5py.File(DATASET_PATH, mode='w') as writer:
            writer.attrs["label_names"] = str(label_names)
            writer.attrs["length"] = len(type_paths_labels)
            writer.attrs["data_type"] = type
            dataset = writer.create_dataset("dataset", (len(type_paths_labels),), dtype=mydtype)
            for idx, input_label in enumerate(inputs_labels):
                dataset[idx] = input_label
        print(f"Saving successfully!")
generate_process_data()
show_time()


#%%
def data_saver(input, label, label_names) -> None:
    return None


@logit(LOG_FILE)
def save_process_samples():
    freq = 500
    max_num = 100
    for data_type in data_types:
        TYPE_DIR = os.path.join(PROC_DATA_DIR, data_type)
        TYPE_EX_DIR = os.path.join(TYPE_DIR, "example")
        DATASET_PATH = os.path.join(TYPE_DIR, dataset_name)
        clear_folder(TYPE_EX_DIR)
        with h5py.File(DATASET_PATH, mode='r') as reader:
            label_names = eval(reader.attrs["label_names"]) #type:ignore
            for id, name in enumerate(label_names):
                print(f"Label {id}: {name}")
            length = reader.attrs['length']
            dataset = reader["dataset"]
            print(f"Data type: '{data_type}', total number: {length}")
            for idx, (input, label) in enumerate(dataset): #type:ignore
                if idx % freq != 0 or idx >= max_num*freq:
                    continue
                data_saver(input, label, label_names)
save_process_samples()
