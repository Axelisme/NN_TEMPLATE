"""
This file contains functions for input and output
"""
import os
import sys
import shutil
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Dict, Any


class PrintO(object):
    def __init__(self):
        self.slient = False

    def __call__(self, *args, **kwargs):
        """print the statement to the console"""
        if not self.slient:
            print(*args, **kwargs)

    def set_slient(self, slient:bool):
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


def show_train_result(
        conf         : Dict[str,Dict|Any],
        step         : int,
        lr           : float,
        train_result : Dict[str,float]
    ):
    """Print result of training."""
    # print result
    show(f'Step: ({step} / {conf["total_steps"]})')
    show(f'lr: {lr:0.3e}')
    show("Train result:")
    for name, evaluator in train_result.items():
        show(f'\t{name}: {evaluator:0.4f}')


def show_valid_result(
        conf         : Dict[str,Dict|Any],
        step         : int,
        valid_result : Dict[str,float]
    ):
    """Print result of validation."""
    # print result
    show(f'Step: ({step} / {conf["total_steps"]})')
    show("Valid result:")
    for name, evaluator in valid_result.items():
        show(f'\t{name}: {evaluator:0.4f}')