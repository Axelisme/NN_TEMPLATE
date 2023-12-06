
"""define a class for the dataset"""

import cv2
import csv

import numpy as np
import torch.utils.data as data


def image_loader(path:str, *others):
    img = cv2.imread(path)
    assert img is not None, f"cannot read image from {path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    return img, *others, path


class ImageCsvDataset(data.Dataset):
    """A dataset that read image data with csv meta data."""
    def __init__(self, dataset_path:str, *args, **kwargs):
        """initialize the dataset
            dataset_path: the file path of the dataset file.
        """
        super(ImageCsvDataset, self).__init__()
        self.dataset_path = dataset_path

        # load the meta data
        with open(dataset_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            self.filedata = list(reader)[1:] # skip the header


    def __getitem__(self, idx):
        filepath, *others = self.filedata[idx]
        image, *others = image_loader(filepath, *others)
        return image, *others


    def __len__(self):
        return len(self.filedata)
