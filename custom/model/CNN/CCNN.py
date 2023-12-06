import torch
import torch.nn as nn


class CLinear(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CLinear, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.re_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)
    self.im_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)

    nn.init.xavier_uniform_(self.re_linear.weight)
    nn.init.xavier_uniform_(self.im_linear.weight)



  def forward(self, x):
    x_re = x[..., 0]
    x_im = x[..., 1]

    out_re = self.re_linear(x_re) - self.im_linear(x_im)
    out_im = self.re_linear(x_im) + self.im_linear(x_re)

    out = torch.stack([out_re, out_im], -1)

    return out


class CConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


        self.re_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
        self.im_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)

        nn.init.xavier_uniform_(self.re_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.re_conv(x_re) - self.im_conv(x_im)
        out_im = self.re_conv(x_im) + self.im_conv(x_re)

        out = torch.stack([out_re, out_im], -1)

        return out


class CConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.re_conv = nn.Conv3d(self.in_channels, self.out_channels, **kwargs)
        self.im_conv = nn.Conv3d(self.in_channels, self.out_channels, **kwargs)

        nn.init.xavier_uniform_(self.re_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.re_conv(x_re) - self.im_conv(x_im)
        out_im = self.re_conv(x_im) + self.im_conv(x_re)

        out = torch.stack([out_re, out_im], -1)

        return out


class CConvTrans2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CConvTrans2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.re_Tconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, **kwargs)
        self.im_Tconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, **kwargs)

        nn.init.xavier_uniform_(self.re_Tconv.weight)
        nn.init.xavier_uniform_(self.im_Tconv.weight)

    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.re_Tconv(x_re) - self.im_Tconv(x_im)
        out_im = self.re_Tconv(x_im) + self.im_Tconv(x_re)

        out = torch.stack([out_re, out_im], -1)

        return out


class CConvTrans3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CConvTrans3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.re_Tconv = nn.ConvTranspose3d(self.in_channels, self.out_channels, **kwargs)
        self.im_Tconv = nn.ConvTranspose3d(self.in_channels, self.out_channels, **kwargs)

        nn.init.xavier_uniform_(self.re_Tconv.weight)
        nn.init.xavier_uniform_(self.im_Tconv.weight)

    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.re_Tconv(x_re) - self.im_Tconv(x_im)
        out_im = self.re_Tconv(x_im) + self.im_Tconv(x_re)

        out = torch.stack([out_re, out_im], -1)

        return out

class CConv1x1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, padding=0, bias=False, **kwargs):
        super(CConv1x1, self).__init__()
        self.conv = CConv2d(in_channel,
                            out_channel,
                            kernel_size=1,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                            **kwargs)

    def forward(self, x):
        return self.conv(x)


class CConv1x1x1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, padding=0, bias=False, **kwargs):
        super(CConv1x1x1, self).__init__()
        self.conv = CConv3d(in_channel,
                            out_channel,
                            kernel_size=1,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                            **kwargs)

    def forward(self, x):
        return self.conv(x)


class CConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, padding=1, bias=False, **kwargs):
        super(CConv3x3, self).__init__()
        self.conv = CConv2d(in_channel,
                            out_channel,
                            kernel_size=3,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                            **kwargs)

    def forward(self, x):
        return self.conv(x)


class CConv3x3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, padding=1, bias=False, **kwargs):
        super(CConv3x3x3, self).__init__()
        self.conv = CConv3d(in_channel,
                            out_channel,
                            kernel_size=3,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                            **kwargs)

    def forward(self, x):
        return self.conv(x)


class CConvTrans3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, output_padding=1, bias=False, **kwargs):
        super(CConvTrans3x3, self).__init__()
        self.Tconv = CConvTrans2d(in_channel,
                                    out_channel,
                                    kernel_size=3,
                                    stride=stride,
                                    output_padding=output_padding,
                                    bias=bias,
                                    **kwargs)

    def forward(self, x):
        return self.Tconv(x)


class CConvTrans3x3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, output_padding=1, bias=False, **kwargs):
        super(CConvTrans3x3x3, self).__init__()
        self.Tconv = CConvTrans3d(in_channel,
                                    out_channel,
                                    kernel_size=3,
                                    stride=stride,
                                    output_padding=output_padding,
                                    bias=bias,
                                    **kwargs)

    def forward(self, x):
        return self.Tconv(x)


class CBatchnorm2d(nn.Module):
    def __init__(self, in_channels):
        super(CBatchnorm2d, self).__init__()
        self.in_channels = in_channels

        self.re_batch = nn.BatchNorm2d(in_channels)
        self.im_batch = nn.BatchNorm2d(in_channels)


    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re =  self.re_batch(x_re)
        out_im =  self.im_batch(x_im)


        out = torch.stack([out_re, out_im], -1)

        return out


class CBatchnorm3d(nn.Module):
    def __init__(self, in_channels):
        super(CBatchnorm3d, self).__init__()
        self.in_channels = in_channels

        self.re_batch = nn.BatchNorm3d(in_channels)
        self.im_batch = nn.BatchNorm3d(in_channels)


    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re =  self.re_batch(x_re)
        out_im =  self.im_batch(x_im)


        out = torch.stack([out_re, out_im], -1)

        return out


class CLayerNorm(nn.Module):
    def __init__(self, in_channels):
        super(CLayerNorm, self).__init__()
        self.in_channels = in_channels

        self.re_layer = nn.LayerNorm(in_channels)
        self.im_layer = nn.LayerNorm(in_channels)


    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re =  self.re_layer(x_re)
        out_im =  self.im_layer(x_im)


        out = torch.stack([out_re, out_im], -1)

        return out


class CMaxPool2d(nn.Module):
    def __init__(self, kernel_size, **kwargs):
        super(CMaxPool2d, self).__init__()
        self.kernel_size = kernel_size


        self.CMax_re = nn.MaxPool2d(self.kernel_size, **kwargs)
        self.CMax_im = nn.MaxPool2d(self.kernel_size, **kwargs)

    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.CMax_re(x_re)
        out_im = self.CMax_im(x_im)


        out = torch.stack([out_re, out_im], -1)

        return out


class CMaxPool3d(nn.Module):
    def __init__(self, kernel_size, **kwargs):
        super(CMaxPool3d, self).__init__()
        self.kernel_size = kernel_size


        self.CMax_re = nn.MaxPool3d(self.kernel_size, **kwargs)
        self.CMax_im = nn.MaxPool3d(self.kernel_size, **kwargs)

    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.CMax_re(x_re)
        out_im = self.CMax_im(x_im)


        out = torch.stack([out_re, out_im], -1)

        return out


class CAvgPool2d(nn.Module):
    def __init__(self, kernel_size, **kwargs):
        super(CAvgPool2d, self).__init__()
        self.kernel_size = kernel_size


        self.CMax_re = nn.AvgPool2d(self.kernel_size, **kwargs)
        self.CMax_im = nn.AvgPool2d(self.kernel_size, **kwargs)

    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.CMax_re(x_re)
        out_im = self.CMax_im(x_im)


        out = torch.stack([out_re, out_im], -1)

        return out


class CAvgPool3d(nn.Module):
    def __init__(self, kernel_size, **kwargs):
        super(CAvgPool3d, self).__init__()
        self.kernel_size = kernel_size


        self.CMax_re = nn.AvgPool3d(self.kernel_size, **kwargs)
        self.CMax_im = nn.AvgPool3d(self.kernel_size, **kwargs)

    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.CMax_re(x_re)
        out_im = self.CMax_im(x_im)


        out = torch.stack([out_re, out_im], -1)

        return out


class CReLU(nn.Module):
    def __init__(self, inplace=False):
        super(CReLU, self).__init__()
        self.inplace = inplace

        self.CReLU = nn.ReLU(inplace=self.inplace)

    def forward(self, x):
        return self.CReLU(x)


class CFlatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(CFlatten, self).__init__()
        self.Cflat_re = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        self.Cflat_im = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.Cflat_re(x_re)
        out_im = self.Cflat_im(x_im)

        out = torch.stack([out_re, out_im], -1)

        return out


class CResBlock2d(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(CResBlock2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = CConv3x3(self.in_channel, self.out_channel, stride=stride)
        self.bn1 = CBatchnorm2d(self.out_channel)
        self.relu = CReLU(inplace=True)
        self.conv2 = CConv3x3(self.out_channel, self.out_channel)
        self.bn2 = CBatchnorm2d(self.out_channel)
        self.downsample = nn.Sequential(
            CConv1x1(self.in_channel, self.out_channel, stride=stride),
            CBatchnorm2d(self.out_channel),
        ) if stride != 1 or self.in_channel != self.out_channel else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class CResBlock3d(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(CResBlock3d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = CConv3x3x3(self.in_channel, self.out_channel, stride=stride)
        self.bn1 = CBatchnorm3d(self.out_channel)
        self.relu = CReLU(inplace=True)
        self.conv2 = CConv3x3x3(self.out_channel, self.out_channel)
        self.bn2 = CBatchnorm3d(self.out_channel)
        self.downsample = nn.Sequential(
            CConv1x1x1(self.in_channel, self.out_channel, stride=stride),
            CBatchnorm3d(self.out_channel),
        ) if stride != 1 or self.in_channel != self.out_channel else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out
