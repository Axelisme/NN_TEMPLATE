
"""A neural network model."""

from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.flaten = nn.Flatten()
        self.fc1 = nn.Linear(4*4*32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.flaten(x)
        x = self.fc1(x)
        return x
    
