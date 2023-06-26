
"""A script to generate data for the project."""
#%%
import os
import h5py
import numpy as np
import torch.utils.data as data
import multiprocessing as mp
import util.utility as ul
from hyperparameter import *

raw_data_name = "raw_data_name"
dataset_name = "dataset.hdf5"
data_types = ["train", "valid", "test"]
data_ratios = base_config.data_ratio

mydtype = np.dtype([("input", np.uint8, (1,)), ("label", np.uint8)])

def data_loader(path, label) -> np.ndarray:
    return 0, label

#%%
@ul.logit(LOG_CONSOLE)
@ul.measure_time
def save_data():
    RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, raw_data_name)
    all_paths_labels, label_names = ul.load_subfolder_as_label(RAW_DATA_PATH, max_num=100000)
    types_paths_labels = data.random_split(all_paths_labels, data_ratios)
    for type, type_paths_labels in zip(data_types, types_paths_labels):
        print(f"Loading {type} data from raw datas")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            inputs_labels = pool.starmap(data_loader, type_paths_labels)
        TYPE_DIR = os.path.join(PROCESSED_DATA_DIR, type)
        DATASET_PATH = os.path.join(TYPE_DIR, dataset_name)
        print(f"Saving {type} data to {TYPE_DIR}......", end="  ")
        os.remove(DATASET_PATH) if os.path.exists(DATASET_PATH) else None
        with h5py.File(DATASET_PATH, mode='w') as writer:
            writer.attrs["label_names"] = str(label_names)
            writer.attrs["length"] = len(type_paths_labels)
            writer.attrs["data_type"] = type
            dataset = writer.create_dataset("dataset", (len(type_paths_labels),), dtype=mydtype)
            for idx, input_label in enumerate(inputs_labels):
                dataset[idx] = input_label
        print(f"Saving successfully!")
save_data()
ul.show_time()

#%%
def data_saver(input, label, label_names) -> None:
    return None

@ul.logit(LOG_CONSOLE)
def sample_data():
    show_freq = 500
    for data_type in data_types:
        EXAMPLE_FOLDER = os.path.join(PROCESSED_DATA_DIR, data_type, "example")
        DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, data_type, dataset_name)
        ul.clear_folder(EXAMPLE_FOLDER)
        with h5py.File(DATASET_PATH, mode='r') as reader:
            label_names = eval(reader.attrs["label_names"])
            for id, name in enumerate(label_names):
                print(f"Label {id}: {name}")
            length = reader.attrs['length']
            dataset = reader["dataset"]
            print(f"Data type: '{data_type}', total number: {length}")
            for idx in range(0, length, show_freq):
                input, label = dataset[idx]
                data_saver(input, label, label_names)
sample_data()

# %%
