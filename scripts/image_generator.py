
"""A script to generate data for the project."""
import os
import csv
import argparse

import PIL
import random

from global_vars import RAW_DATA_DIR, PROC_DATA_DIR
from util.io import clear_folder
from util.datatools import apply_to_all_files, load_dir_as_label, split_list


# define the data parameters
split_ratio = {
    "train": 0.8,
    "valid": 0.1,
    "test" : 0.1
}
postfix = ".jpg"
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
label_dict = {label: idx for idx, label in enumerate(labels)}


# define the data loader, should have your own custom data loader
def transform_image(src_path, dst_path):
    """transform the image from src_path to dst_path"""
    img = PIL.Image.open(src_path)
    img = img.resize((1920, 960))
    img = img.convert("RGB")
    img.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    raw_data_root = os.path.join(RAW_DATA_DIR, dataset_name)

    # clear the target folder
    target_dir = os.path.join(PROC_DATA_DIR, dataset_name)
    clear_folder(target_dir)

    # generate data
    apply_to_all_files(raw_data_root, target_dir, transform_image, postfix=postfix)

    # split data by different csv files
    path_labels, _ = load_dir_as_label(target_dir, postfix, label_dict)
    random.shuffle(path_labels)
    split_datas = split_list(path_labels, list(split_ratio.values()))
    for data, split in zip(split_datas, split_ratio.keys()):
        split_file = os.path.join(target_dir, f"split-{split}.csv")
        with open(split_file, "x") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label_id", "label_name"])
            writer.writerows(data)


if __name__ == "__main__":
    main()
