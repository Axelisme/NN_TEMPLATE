
"""define a class for the dataset"""

import csv

import cv2
import torch.utils.data as data
import torchvision.transforms as transforms

non_aug_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])

aug_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.GaussianBlur((5, 9), sigma=(0.1, 5)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Grayscale(3),
    ]),
    transforms.RandomPerspective(),
    transforms.RandomAutocontrast(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])

def image_loader(transform_fn, path:str, label_id:str, label_name:str):
    img = cv2.imread(path)
    assert img is not None, f"cannot read image from {path}"
    assert img.shape == (224, 224, 3), f"image shape is not (224, 224, 3) but {img.shape}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform_fn(img)
    return img, int(label_id), label_name, path


class ImageCsvDataset(data.Dataset):
    """A dataset that read image data with csv meta data."""
    def __init__(self, dataset_path:str, apply_augment = True, **kwargs):
        """initialize the dataset
            dataset_path: the file path of the dataset file.
        """
        super(ImageCsvDataset, self).__init__()
        self.dataset_path = dataset_path

        if apply_augment:
            self.transform_fn = aug_transform
        else:
            self.transform_fn = non_aug_transform

        # load the meta data
        with open(dataset_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            self.filedata = list(reader)[1:] # skip the header


    def __getitem__(self, idx):
        filepath, *others = self.filedata[idx]
        image, *others = image_loader(self.transform_fn, filepath, *others)
        return image, *others


    def __len__(self):
        return len(self.filedata)

