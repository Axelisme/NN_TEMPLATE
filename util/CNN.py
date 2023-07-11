"""
Some useful tools for CNNs
"""


def conv_output_size(input_size, kernel_size, stride = 1, padding = 0, dilation = 1) -> int:
    """calculate the output size of a convolutional layer"""
    return (input_size+2*padding-dilation*(kernel_size-1)-1)//stride+1


def pool_output_size(input_size, kernel_size, stride = None, padding = 0, dilation = 1) -> int:
    """calculate the output size of a pooling layer"""
    if stride is None:
        stride = kernel_size
    return conv_output_size(input_size, kernel_size, stride, padding, dilation)


def conv_transpose_output_size(input_size, kernel_size, stride = 1, padding = 0, dilation = 1, output_padding = 0) -> int:
    """calculate the output size of a convolutional transpose layer"""
    return int((input_size-1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1)
