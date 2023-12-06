
"""A neural network model."""

import torch
from torch import nn
from torch import Tensor
from util.CNN import conv_output_size
from config.configClass import Config


class SimpleConv(nn.Module):
    def __init__(self, config:Config):
        """Initialize a neural network model."""
        super(SimpleConv, self).__init__()
        # input: 3*H*W
        self.input_channel, self.input_height, self.input_width = config.input_size

        # conv1: 64*H1*W1
        self.Conv1_channel = 64
        self.Conv1_height = conv_output_size(conv_output_size(self.input_height, 5), 2, 2)
        self.Conv1_width  = conv_output_size(conv_output_size(self.input_width , 5), 2, 2)
        self.Conv1 = nn.Sequential(
            nn.Conv2d(self.input_channel,
                        self.Conv1_channel,
                        kernel_size = 5,
                        stride = 1,
                        padding = 0,
                        dilation = 1,
                        bias = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(self.Conv1_channel),
            nn.ReLU()
        )

        # conv2: 64*H2*W2
        self.Conv2_channel = 64
        self.Conv2_height = conv_output_size(conv_output_size(self.Conv1_height, 7), 2, 2)
        self.Conv2_width  = conv_output_size(conv_output_size(self.Conv1_width , 7), 2, 2)
        self.Conv2 = nn.Sequential(
            nn.Conv2d(self.Conv1_channel,
                        self.Conv2_channel,
                        kernel_size = 7,
                        stride = 1,
                        padding = 0,
                        dilation = 1,
                        bias = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(self.Conv2_channel),
            nn.ReLU()
        )

        # conv3: 128*H3*W3
        self.Conv3_channel = 128
        self.Conv3_height = conv_output_size(conv_output_size(self.Conv2_height, 9), 2, 2)
        self.Conv3_width  = conv_output_size(conv_output_size(self.Conv2_width , 9), 2, 2)
        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.Conv2_channel,
                        self.Conv3_channel,
                        kernel_size = 9,
                        stride = 1,
                        padding = 0,
                        dilation = 1,
                        bias = False),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(self.Conv3_channel),
            nn.ReLU()
        )

        # flatten
        self.flatten = nn.Flatten()
        self.flatten_out = self.Conv3_channel * self.Conv3_height * self.Conv3_width

        # linear
        self.output_size = config.output_size
        self.Linear = nn.Linear(self.flatten_out, self.output_size)

    @torch.compile
    def forward(self, x:Tensor) -> Tensor:
        """Forward a batch of data through the model."""
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.flatten(x)
        x = self.Linear(x)
        return x.softmax(dim = 1)
