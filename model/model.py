
"""A complex CNN model for the QINN dataset."""

from torch import nn

class WaveFuncTrans(nn.Module):
    def __init__(self):
        super(WaveFuncTrans, self).__init__()

    def forward(self, x):
        return x
    
