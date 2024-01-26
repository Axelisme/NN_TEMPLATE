"""
This file contains functions for input and output
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm


class PrintO(object):
    def __init__(self):
        self.slient = False

    def __call__(self, *args, **kwargs):
        """print the statement to the console"""
        if not self.slient:
            tqdm.write(*args, **kwargs)


    def set_silent(self, slient:bool):
        """set slient or not"""
        self.slient = slient

show = PrintO()


def clear_folder(path:str):
    """clear the folder"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def plot_confusion_matrix(cm, class_names, path = None, title='Confusion matrix', normalize=False):
    """plot the confusion matrix and save it to the given path if provided,
        input: confusion matrix, classes, path to save the figure, title of the figure
        output: None"""
    if normalize:
        cm = cm.astype('float') / np.nansum(cm, axis=1, keepdims=True)
        np.fill_diagonal(cm,np.nan)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]*100, fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()