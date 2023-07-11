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


def clear_folder(path:str):
    """clear the folder"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


class LogO(object):
    """customized O for logging, it will print to the console and write to the log file"""
    def __init__(self, path: str):
        """input: path to the log file"""
        self.terminal = sys.stdout
        try:
            self.log = open(path, "a")
        except FileNotFoundError:
            raise FileNotFoundError(f'Log file {path} not found')

    def close(self):
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return sys.__stdout__.isatty()

    def fileno(self):
        return self.log.fileno()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log.close()


def write_to_log(content:str, path:str):
    """write the content to the log file"""
    with open(path, 'a') as f:
        f.write(content)


def logit(path:str = 'log.txt'):
    """log what the function print to the log file, and print it to the console"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            try:
                with LogO(path) as f:
                    sys.stdout = f
                    result = func(*args, **kwargs)
            finally:
                sys.stdout = original_stdout
            return result
        return wrapper
    return decorator


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
