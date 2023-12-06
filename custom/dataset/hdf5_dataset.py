
"""define a class for the dataset"""

import h5py
import weakref

import torch.utils.data as data


def data_transform(input, *other):
    return input, *other


class HDF5DataSet(data.Dataset):
    """A dataset that read data from hdf5 file."""
    def __init__(self, dataset_path:str, *args, **kwargs):
        """initialize the dataset
            dataset_path: the file path of the dataset file.
            transform: the transform function before input the data to the model.
        """
        super(HDF5DataSet, self).__init__()
        self.dataset_path = dataset_path

        # load dataset meta data
        self.meta_dict = {}
        with h5py.File(self.dataset_path, "r") as reader:
            self.meta_dict.update(reader.attrs)

        # Don't load file handler in init() to avoid problem of multi-process in DataLoader
        # instead use __lazy_load() in __getitem__()
        self.fileHandler = None
        self.filedata = None
        self.__finalizer = None


    def __getitem__(self, idx):
        # load file handler at first time of __getitem__
        if self.fileHandler is None:
            self.__lazy_load()
        # get data
        input, *other = self.filedata[idx]  # type:ignore
        # transform input if needed
        input, *other = data_transform(input, *other)
        return input, *other


    def __len__(self):
        return self.meta_dict["length"]


    def close(self):
        """close the file handler."""
        if self.fileHandler is not None:
            self.filedata = None
            self.fileHandler.close()
            self.fileHandler = None


    def __lazy_load(self):
        self.fileHandler = h5py.File(self.dataset_path, "r")
        self.filedata = self.fileHandler["dataset"]
        # register the close() method to the weakref.finalize() to close the file handler when the object is deleted
        self.__finalizer = weakref.finalize(self, self.close)
