"""
Some useful data tools
"""

import os
import random
import h5py
import numpy as np
import multiprocessing as mp
from typing import Callable, Dict
from tqdm.auto import tqdm

def generate_hdf5_data(dataset_path:str,
                          data_dtype:np.dtype,
                          data_loader:Callable,
                          mode_length:int = 1000,
                          max_batch_num:int = 100,
                          meta_dict:Dict[str,str] = None) -> None:
    """
    Generate processed data for the project.
    data_loader: a function that takes an index and returns a data sample.
    """
    # save data to h5 file
    with mp.Pool(mp.cpu_count()) as pool:
        with h5py.File(dataset_path, mode='x') as writer:
            # write meta data
            writer.attrs["length"] = mode_length
            writer.attrs["data_dtype"] = str(data_dtype)
            if meta_dict:
                for key, value in meta_dict.items():
                    writer.attrs[key] = value
            # create dataset
            dataset = writer.create_dataset("dataset", (mode_length,), dtype=data_dtype)
            # write dataset
            saved_num = 0
            bar = tqdm(total=mode_length)
            while saved_num < mode_length:
                batch_num = min(max_batch_num, mode_length - saved_num)
                save_ids = list(range(saved_num, saved_num + batch_num))
                batch = pool.imap_unordered(data_loader, save_ids)
                for idx, data in zip(save_ids, batch):
                    dataset[idx] = data
                    bar.update()
                saved_num += batch_num
            bar.close()


def sampling_hdf5_data(dataset_path:str,
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
        meta_dict = {key: value for key, value in reader.attrs.items()}
        meta_dict.pop("length")
        # sampling
        sample_ids = random.sample(range(data_length), max_num)
        # save samples
        for idx, sample_id in enumerate(sample_ids):
            print(f"Saving sample {idx}")
            sample = dataset[sample_id]
            sample_saver(idx, sample, meta_dict, *args, **kwargs)


def load_subfolder_as_label(root: str, loader = None, max_num = 1000):
    """load data from a folder, and use the subfolder name as label name
        input: path to the folder,
               a loader(path), default to return path,
               max number of data to load per label
        output: datas, label_names"""
    datas = []
    label_names = []
    label_num = 0
    for dir, _, files in os.walk(root):
        if root == dir:  # skip the root folder
            continue
        # eg. root = 'data', dir = 'data/A/X/1', label_name = 'A_X_1'
        reldir = os.path.relpath(dir, root)
        label_names.append(reldir.replace(os.sep,'_'))
        for id, file in enumerate(files):
            if id >= max_num:
                break
            file_path = os.path.join(dir, file)
            data = loader(file_path) if loader else file_path
            datas.append((data, label_num))
        label_num += 1
    return datas, label_names
