
"""A neural network model."""

import util.utility as ul
from config.configClass import Config
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class ClassicalConv(nn.Module):
    def __init__(self, config:Config):
        """Initialize a neural network model."""
        super(ClassicalConv, self).__init__()
        # input: 3*H*W
        self.input_channel, self.input_height, self.input_width = config.input_size

        # conv1: 64*H1*W1
        self.Conv1_channel = 64
        self.Conv1_height = ul.conv_output_size(ul.conv_output_size(self.input_height, 5), 2, 2)
        self.Conv1_width  = ul.conv_output_size(ul.conv_output_size(self.input_width , 5), 2, 2)
        self.Conv1 = nn.Sequential(
            nn.Conv2d(self.input_channel,
                        self.Conv1_channel,
                        kernel_size = 5,
                        stride = 1,
                        padding = 0,
                        dilation = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(self.Conv1_channel),
            nn.ReLU(inplace=True)
        )

        # conv2: 64*H2*W2
        self.Conv2_channel = 64
        self.Conv2_height = ul.conv_output_size(ul.conv_output_size(self.Conv1_height, 7), 2, 2)
        self.Conv2_width  = ul.conv_output_size(ul.conv_output_size(self.Conv1_width , 7), 2, 2)
        self.Conv2 = nn.Sequential(
            nn.Conv2d(self.Conv1_channel,
                        self.Conv2_channel,
                        kernel_size = 7,
                        stride = 1,
                        padding = 0,
                        dilation = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(self.Conv2_channel),
            nn.ReLU(inplace=True)
        )

        # conv3: 128*H3*W3
        self.Conv3_channel = 128
        self.Conv3_height = ul.conv_output_size(ul.conv_output_size(self.Conv2_height, 7), 2, 2)
        self.Conv3_width  = ul.conv_output_size(ul.conv_output_size(self.Conv2_width , 7), 2, 2)
        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.Conv2_channel,
                        self.Conv3_channel,
                        kernel_size = 9,
                        stride = 1,
                        padding = 0,
                        dilation = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(self.Conv3_channel),
            nn.ReLU(inplace=True)
        )

        # flatten
        self.flat = nn.Flatten()
        self.flatten_out = self.Conv3_channel * self.Conv3_height * self.Conv3_width

        # linear
        self.output_size = config.output_size
        self.fc = nn.Linear(self.flatten_out, self.output_size)

    #@torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.flat(x)
        x = self.fc(x)
        return x.softmax(dim = 1)

