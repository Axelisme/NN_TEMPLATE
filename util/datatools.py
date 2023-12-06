"""
Some useful data tools
"""

import os
import random
from typing import Callable, Dict, List

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def generate_hdf5_data(dataset_path:str,
                          data_dtype:np.dtype,
                          data_loader:Callable,
                          length:int = 1000,
                          n_jobs:int = -1,
                          meta_dict:Dict[str,str] = None) -> None:
    """
    Generate processed data for the project.
    data_loader: a function that takes an index and returns a data sample.
    """
    # save data to h5 file
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    with h5py.File(dataset_path, mode='x') as writer:
        # write meta data
        writer.attrs["length"] = length
        writer.attrs["data_dtype"] = str(data_dtype)
        if meta_dict:
            for key, value in meta_dict.items():
                writer.attrs[key] = value
        # create dataset
        dataset = writer.create_dataset("dataset", (length,), dtype=data_dtype)
        # write dataset
        ids = list(range(length))
        batch = parallel(delayed(data_loader)(i) for i in ids)
        for idx, data in zip(ids, batch):
            dataset[idx] = data


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


def load_dir_as_label(root: str, postfix: str, dict: Dict[str, int] = None):
    """load data from a root folder, and use the parent folder name of file as label name
        input: path to the folder,
               postfix of the files to load
        output: datas, label_names"""
    datas = []
    label_dict = dict if dict else {}
    for dir, _, files in os.walk(root):
        files = [f for f in files if f.endswith(postfix)]
        label_name = os.path.basename(dir)
        if len(files) != 0 and label_name not in label_dict:
            if dict:
                raise ValueError(f"Label name {label_name} not in label dict")
            label_dict[label_name] = len(label_dict)
        for file in files:
            file_path = os.path.join(dir, file)
            datas.append((file_path, label_dict[label_name], label_name))
    return datas, label_dict


def apply_to_all_files(root: str, target_dir: str, process_func: Callable, postfix: str):
    """apply a process function to all files in root folder,
        and save it to target folder with the same structure"""
    os.makedirs(target_dir, exist_ok=True)
    parallel = Parallel(n_jobs=-1, return_as="generator")
    delayed_funcs = []
    for dir, _, files in os.walk(root):
        files = [f for f in files if f.endswith(postfix)]
        if len(files) == 0:
            continue
        target_subdir = os.path.join(target_dir, os.path.relpath(dir, root))
        os.makedirs(target_subdir, exist_ok=True)
        for file in files:
            file_path = os.path.join(dir, file)
            target_file_path = os.path.join(target_subdir, file)
            delayed_funcs.append(delayed(process_func)(file_path, target_file_path))
    pbar = tqdm(total=len(delayed_funcs), desc="Processing")
    for _ in parallel(delayed_funcs):
        pbar.update()
    pbar.close()


def split_list(data_list: List, split_ratios: List) -> List[List]:
    """split a list into several parts"""
    assert sum(split_ratios) == 1
    split_lens = [int(len(data_list) * ratio) for ratio in split_ratios]
    split_lens[-1] = len(data_list) - sum(split_lens[:-1])
    split_list = []
    start_idx = 0
    for split_len in split_lens:
        split_list.append(data_list[start_idx:start_idx + split_len])
        start_idx += split_len
    return split_list


def cycle_iter(iterable, callback=None):
    """cycle iterator"""
    while True:
        for item in iterable:
            yield item
        else:
            if callback:
                callback()