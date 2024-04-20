import cv2
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class FeatureExtract1(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm='batch'):
        super(FeatureExtract1, self).__init__()
        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, activation, norm=None)
        self.conv2 = ConvBlock(output_size, 64, kernel_size, stride, padding, activation, norm=None)
        self.conv3 = ConvBlock(64, 64, 1, 1, 0, activation, norm=None)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        return d3


class FeatureExtract2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(FeatureExtract2, self).__init__()
        self.conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.conv3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation, norm=None)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        return d3


class FeatureExtract3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation='prelu', norm='batch'):
        super(FeatureExtract3, self).__init__()
        self.conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.conv3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation, norm=None)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        return d3


class GmOut(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(GmOut, self).__init__()
        self.conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.conv3 = ConvBlock(num_filter, 1, 1, 1, 0, activation=None, bias=False, norm=None)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        return d3
    
    
class DilaConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, dilation=2, bias=True,
                 activation='prelu', norm=None):
        super(DilaConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, dilation, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out


class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True,
                 activation='prelu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = Upsampler(scale, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = Upsampler(scale, num_filter)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu',
                 norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True,
                 activation='prelu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class PSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True,
                 activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor ** 2, kernel_size, stride, padding,
                                    bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(2))
            if bn: modules.append(torch.nn.BatchNorm2d(n_feat))
            # modules.append(torch.nn.PReLU())
        self.up = torch.nn.Sequential(*modules)

        self.activation = act
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out


class UpBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock_x8, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv4 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.up_conv5 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv6 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv7 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        h1 = self.up_conv2(h0)
        h2 = self.up_conv3(h1)
        l0 = self.up_conv4(h2)
        h3 = self.up_conv5(l0 - x)
        h4 = self.up_conv6(h3)
        h5 = self.up_conv7(h4)
        return h2 + h5


class DownBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock_x8, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv4 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv5 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h1 = self.down_conv2(l0)
        h2 = self.down_conv3(h1)
        h3 = self.down_conv4(h2)
        l1 = self.down_conv5(h3 - x)
        return l0 + l1


class D_DownBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_DownBlock_x8, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv4 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv5 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)

    def forward(self, x):
        x1 = self.conv(x)
        l0 = self.down_conv1(x1)
        h0 = self.down_conv2(l0)
        h1 = self.down_conv3(h0)
        h2 = self.down_conv4(h1)
        l1 = self.down_conv5(h2 - x1)
        return l1 + l0


class D_UpBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_UpBlock_x8, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.up_conv4 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.up_conv5 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv6 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv7 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        h1 = self.up_conv2(h0)
        h2 = self.up_conv3(h1)
        l0 = self.up_conv4(h2)
        h3 = self.up_conv5(l0 - x)
        h4 = self.up_conv6(h3)
        h5 = self.up_conv7(h4)
        return h2 + h5


class MultiBlock1(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiBlock1, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(2 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(3 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(4 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(5 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(5 * 64, 64, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        # print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class MultiBlock2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiBlock2, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(6 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(6 * 64, 64, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        # print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class MultiBlock3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiBlock3, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(4 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(5 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(6 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(7 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(7 * 64, 64, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        # print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class MultiBlock4(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiBlock4, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(5 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(6 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(7 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(8 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(8 * 64, 64, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        # print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class MultiBlock5(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiBlock5, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(6 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(7 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(8 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(9 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(9 * 64, 64, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        # print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class UpBlockP(torch.nn.Module):
    def __init__(self, num, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlockP, self).__init__()
        self.conv1 = ConvBlock(num * num_filter, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv1(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


#####2019.07.22
class MultiViewBlock1(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock1, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(2 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(3 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(4 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(5 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(5 * 64, 64, 1, 1, 0, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        # print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class MultiViewBlock2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock2, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(6 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(6 * 64, 64, 1, 1, 0, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class MultiViewBlock3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock3, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(6 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(6 * 64, 64, 1, 1, 0, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class MultiViewBlock4(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock4, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(6 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(6 * 64, 64, 1, 1, 0, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class MultiViewBlock5(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock5, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(6 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        self.output_conv1 = ConvBlock(6 * 64, 64, 1, 1, 0, activation=None, norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.output_conv1(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return x_prior1_2


class LowFusion(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LowFusion, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)

    def forward(self, x):
        x0 = self.conv1(x)
        out = self.conv2(x0)
        return out


class LowFusion2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LowFusion2, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.res = ResnetBlock(num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)

    def forward(self, x):
        x0 = self.conv1(x)
        res = self.res(x0)
        out = self.conv2(res)
        return out


class LF_Mask(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LF_Mask, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)
        self.act = torch.nn.Sigmoid()

    def forward(self, rgb, depth):
        c0 = self.conv1(rgb)
        c1 = self.conv2(c0)
        d0 = self.conv11(depth)
        d1 = self.conv22(d0)
        res = c1 - d1
        att = self.conv3(res)
        mask = self.act(att)
        mask2 = 1 - mask
        out = mask2 * rgb + depth
        return out


class LF_Mask4(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LF_Mask4, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)
        self.att = torch.nn.PReLU()
        self.act = torch.nn.Sigmoid()

    def forward(self, rgb, depth):
        c0 = self.conv1(rgb)
        c1 = self.conv2(c0)
        d0 = self.conv1(depth)
        d1 = self.conv2(d0)
        res = c1 - d1
        att = self.att(res)
        mask = self.act(att)
        mask2 = 1 - mask
        out = mask2 * rgb + depth
        return out


def draw_features(width,height,x,savename):
    fig = plt.figure()
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # for i in range(width*height):
    #     plt.subplot(height,width, i + 1)
    plt.axis('off')
    # print(x.size())
    img = x[0, 0, :, :]
    # img = x.squeeze()
    img = img.numpy()
    #print(img)
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
    # img = img * 255

    # img = np.mean(img, axis=0)
    # img = img * 255
    # print(img)
    img = img.astype(np.uint8)  #转成unit8
    # img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
    # img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
    # plt.imshow(img, cmap='gray')
    plt.imshow(img)
    fig.savefig(savename, dpi=200)
    fig.clf()
    plt.close()

    
class LF_Mask3(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LF_Mask3, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)
        self.act = torch.nn.Sigmoid()

    def forward(self, rgb, depth):
        c0 = self.conv1(rgb)
        c1 = self.conv2(c0)
        d0 = self.conv1(depth)
        d1 = self.conv2(d0)
        res = c1 - d1
        att = self.conv3(res)
        mask = torch.clamp(self.act(att), 1e-7, 1-1e-7)
        # mask = self.act(att)
        mask2 = 1 - mask

        # vis = mask.cpu().clone().detach()
        # vis2 = mask2.cpu().clone().detach()
        # draw_features(8, 8, vis2, './Results/vis/mask_sub34_test.png')
        # draw_features(8, 8, vis, './Results/vis/mask_sub.png')

        # vis = mask2.cpu().clone().detach()
        # save_img = vis.clamp(0, 1).numpy()
        # save_dir = './Results/vis'
        # cv2.imwrite(save_dir + '/' + 'mask.png', np.uint8(save_img * 255))

        out = mask2 * rgb + depth
        return out


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class LDE(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LDE, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter , 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv_ch1 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch2 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch3 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch4 = ConvBlock(num_filter // 2, num_filter, 1, 1, 0, activation=None, bias=False, norm=None)

        self.softmax = torch.nn.Softmax(-1)

    def forward(self, rgb, depth):
        b, c, h, w = depth.shape
        c0 = self.conv2(self.conv1(rgb))
        c1 = self.conv_ch1(c0).view(b, c//2, -1)  # B * C/2 * (H * W)
        c1 = c1.permute(0, 2, 1)  # B * (H * W) * C/2

        d0 = self.conv2(self.conv1(depth))
        d1 = self.conv_ch2(d0).view(b, c//2, -1)  # B * C/2 * (H * W)
        d1 = d1.permute(0, 2, 1)  # B * (H * W) * C/2
        d2 = self.conv_ch3(d0).view(b, c//2, -1)  # B * C/2 * (H * W)

        self_map = self.softmax(torch.matmul(d1, d2))  # B * (H * W) * (H * W)
        guided = torch.matmul(self_map, c1)  # B * (H * W) * C/2
        guided = guided.permute(0, 2, 1).contiguous().view(b, c//2, h, w)
        guided = self.conv_ch4(guided)

        out = guided + d0
        return out


class LDE_gm_cos(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LDE_gm_cos, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv_ch1 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch2 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch3 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch4 = ConvBlock(num_filter // 2, num_filter, 1, 1, 0, activation=None, bias=False, norm=None)

        self.dyconv1 = Dynamic_conv2d(32, 32,  kernel_size=3, stride=1, padding=1)
        self.dyconv2 = Dynamic_conv2d(64, 64,  kernel_size=3, stride=1, padding=1)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, rgb, depth):
        # non_local
        b, c, h, w = depth.shape
        c0 = self.conv2(self.conv1(rgb))
        c1 = self.conv_ch1(c0).view(b, c//2, -1)  # B * C/2 * (H * W)
        c1 = c1.permute(0, 2, 1)  # B * (H * W) * C/2

        d0 = self.conv2(self.conv1(depth))
        d1 = self.conv_ch2(d0).view(b, c//2, -1)  # B * C/2 * (H * W)
        d1 = d1.permute(0, 2, 1)  # B * (H * W) * C/2
        d2 = self.conv_ch3(d0).view(b, c//2, -1)  # B * C/2 * (H * W)

        self_map = self.softmax(torch.matmul(d1, d2))  # B * (H * W) * (H * W)
        guided = torch.matmul(self_map, c1)  # B * (H * W) * C/2
        guided = guided.permute(0, 2, 1).contiguous().view(b, c//2, h, w)
        dy_guided = self.dyconv1(guided)
        dy_guided = self.conv_ch4(dy_guided)

        out1 = dy_guided + d0

        # cos
        # cosine_eps = 1e-7
        # rgb_q = rgb.contiguous().view(b, c, -1)  # B * C * (H * W)
        # rgb_q_norm = torch.norm(rgb_q, 2, 1, True)
        # depth_k = depth.contiguous().view(b, c, -1).permute(0, 2, 1)  # B * (H * W) * C
        # depth_k_norm = torch.norm(depth_k, 2, 2, True)
        # cos_sim = torch.matmul(depth_k, rgb_q) / (torch.matmul(depth_k_norm, rgb_q_norm) + cosine_eps)
        #
        # rgb_sim = torch.matmul(rgb_q, cos_sim).contiguous().view(b, c, h, w)
        # rgb_sim = self.dyconv2(rgb_sim)
        # out = out1 * rgb_sim

        cos_sim = F.cosine_similarity(rgb, depth, dim=1).unsqueeze(1).repeat(1, c, 1, 1)  # B * C * H * W
        rgb_sim = rgb * cos_sim
        rgb_sim = self.dyconv2(rgb_sim)
        out = out1 * rgb_sim

        return out


class LDE_gm_cos2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LDE_gm_cos2, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv_ch1 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch2 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch3 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch4 = ConvBlock(num_filter // 2, num_filter, 1, 1, 0, activation=None, bias=False, norm=None)

        self.dyconv1 = Dynamic_conv2d(32, 32,  kernel_size=3, stride=1, padding=1)
        self.dyconv2 = Dynamic_conv2d(64, 64,  kernel_size=3, stride=1, padding=1)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, rgb, depth):
        # non_local
        b, c, h, w = depth.shape
        c0 = self.conv2(self.conv1(rgb))
        c1 = self.conv_ch1(c0).view(b, c//2, -1)  # B * C/2 * (H * W)
        c1 = c1.permute(0, 2, 1)  # B * (H * W) * C/2

        d0 = self.conv2(self.conv1(depth))
        d1 = self.conv_ch2(d0).view(b, c//2, -1)  # B * C/2 * (H * W)
        d1 = d1.permute(0, 2, 1)  # B * (H * W) * C/2
        d2 = self.conv_ch3(d0).view(b, c//2, -1)  # B * C/2 * (H * W)

        self_map = self.softmax(torch.matmul(d1, d2))  # B * (H * W) * (H * W)
        guided = torch.matmul(self_map, c1)  # B * (H * W) * C/2
        guided = guided.permute(0, 2, 1).contiguous().view(b, c//2, h, w)
        dy_guided = self.dyconv1(guided)
        dy_guided = self.conv_ch4(dy_guided)

        out1 = dy_guided + d0

        # cos
        # cosine_eps = 1e-7
        # rgb_q = rgb.contiguous().view(b, c, -1)  # B * C * (H * W)
        # rgb_q_norm = torch.norm(rgb_q, 2, 1, True)
        # depth_k = depth.contiguous().view(b, c, -1).permute(0, 2, 1)  # B * (H * W) * C
        # depth_k_norm = torch.norm(depth_k, 2, 2, True)
        # cos_sim = torch.matmul(depth_k, rgb_q) / (torch.matmul(depth_k_norm, rgb_q_norm) + cosine_eps)
        #
        # rgb_sim = torch.matmul(rgb_q, cos_sim).contiguous().view(b, c, h, w)
        # rgb_sim = self.dyconv2(rgb_sim)
        # out = out1 * rgb_sim

        cos_sim = F.cosine_similarity(rgb, depth, dim=1).unsqueeze(1).repeat(1, c, 1, 1)  # B * C * H * W
        rgb_sim = rgb * cos_sim
        rgb_sim = self.dyconv2(rgb_sim)
        out = out1 + rgb_sim

        return out


class LDE_gm_cos3(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LDE_gm_cos3, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv_ch1 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch2 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch3 = ConvBlock(num_filter, num_filter // 2, 1, 1, 0, activation=None, bias=False, norm=None)
        self.conv_ch4 = ConvBlock(num_filter // 2, num_filter, 1, 1, 0, activation=None, bias=False, norm=None)

        self.dyconv1 = Dynamic_conv2d(32, 32,  kernel_size=3, stride=1, padding=1)
        self.dyconv2 = Dynamic_conv2d(64, 64,  kernel_size=3, stride=1, padding=1)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, rgb, depth):
        # non_local
        b, c, h, w = depth.shape
        c0 = self.conv2(self.conv1(rgb))
        c1 = self.conv_ch1(c0).view(b, c//2, -1)  # B * C/2 * (H * W)
        c1 = c1.permute(0, 2, 1)  # B * (H * W) * C/2

        d0 = self.conv2(self.conv1(depth))
        d1 = self.conv_ch2(d0).view(b, c//2, -1)  # B * C/2 * (H * W)
        d1 = d1.permute(0, 2, 1)  # B * (H * W) * C/2
        d2 = self.conv_ch3(d0).view(b, c//2, -1)  # B * C/2 * (H * W)

        self_map = self.softmax(torch.matmul(d1, d2))  # B * (H * W) * (H * W)
        guided = torch.matmul(self_map, c1)  # B * (H * W) * C/2
        guided = guided.permute(0, 2, 1).contiguous().view(b, c//2, h, w)
        dy_guided = self.dyconv1(guided)
        dy_guided = self.conv_ch4(dy_guided)

        out = dy_guided + d0

        # cos
        # cosine_eps = 1e-7
        # rgb_q = rgb.contiguous().view(b, c, -1)  # B * C * (H * W)
        # rgb_q_norm = torch.norm(rgb_q, 2, 1, True)
        # depth_k = depth.contiguous().view(b, c, -1).permute(0, 2, 1)  # B * (H * W) * C
        # depth_k_norm = torch.norm(depth_k, 2, 2, True)
        # cos_sim = torch.matmul(depth_k, rgb_q) / (torch.matmul(depth_k_norm, rgb_q_norm) + cosine_eps)
        #
        # rgb_sim = torch.matmul(rgb_q, cos_sim).contiguous().view(b, c, h, w)
        # rgb_sim = self.dyconv2(rgb_sim)
        # out = out1 * rgb_sim

        # cos_sim = F.cosine_similarity(rgb, depth, dim=1).unsqueeze(1).repeat(1, c, 1, 1)  # B * C * H * W
        # rgb_sim = rgb * cos_sim
        # rgb_sim = self.dyconv2(rgb_sim)
        # out = out1 + rgb_sim

        return out


class LDE_cosPrior(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LDE_cosPrior, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)
        self.act = torch.nn.Sigmoid()

    def forward(self, rgb, depth, cos_sim):
        c0 = self.conv1(rgb)
        c1 = self.conv2(c0) * cos_sim
        d0 = self.conv1(depth)
        d1 = self.conv2(d0) * cos_sim
        res = c1 - d1
        att = self.conv3(res)
        mask = torch.clamp(self.act(att), 1e-7, 1-1e-7)
        # mask = self.act(att)
        mask2 = 1 - mask

        # vis = mask.cpu().clone().detach()
        # vis2 = mask2.cpu().clone().detach()
        # draw_features(8, 8, vis2, './Results/vis/mask_sub34_test.png')
        # draw_features(8, 8, vis, './Results/vis/mask_sub.png')

        # vis = mask2.cpu().clone().detach()
        # save_img = vis.clamp(0, 1).numpy()
        # save_dir = './Results/vis'
        # cv2.imwrite(save_dir + '/' + 'mask.png', np.uint8(save_img * 255))

        out = mask2 * rgb + depth
        return out


class LDE_GMprior(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LDE_GMprior, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.res = nn.Sequential(
            ResnetBlock(num_filter),
            nn.PReLU()
        )
        self.rdb = nn.Sequential(
            RDB(G0=num_filter*2, C=4, G=32),
            channel_attentionBlock(num_filter*2),
            nn.Conv2d(num_filter*2, num_filter, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, rgb, depth, c_mask=None, d_mask=None):
        c0 = self.conv1(rgb)
        c1 = self.conv2(c0)
        d0 = self.conv1(depth)
        d1 = self.conv2(d0)

        d_enh = c1 * d_mask + d1
        d_enh1 = self.res(d_enh)
        
        if c_mask is not None:
            d_enh2 = torch.cat((c1 * c_mask, d_enh1), dim=1)
            out = self.rdb(d_enh2)
        else:
            out = d_enh1

        return out


class LDE_GMprior2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LDE_GMprior2, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.res = nn.Sequential(
        #     ResnetBlock(num_filter),
        #     nn.PReLU()
        # )
        self.field = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        )
        self.rdb = RDB(G0=num_filter, C=4, G=32)

    def forward(self, rgb, depth, d_mask):
        c0 = self.conv1(rgb)
        c1 = self.conv2(c0)
        d0 = self.conv1(depth)
        d1 = self.conv2(d0)
        res = torch.abs(c1 - d1)

        d_enh = c1 * d_mask + d1
        d_enh1 = self.rdb(d_enh)
        
        d_max, _ = torch.max(d1, dim=1, keepdim=True)
        d_max = self.field(d_max)
        mask = torch.sigmoid(d_max)

        out = mask * res * 0.1 + d_enh1

        return out


class LDE_GMprior3(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LDE_GMprior3, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        # self.res = nn.Sequential(
        #     ResnetBlock(num_filter),
        #     nn.PReLU()
        # )
        self.field = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        )
        self.rdb = RDB(G0=num_filter, C=4, G=32)

    def forward(self, rgb, depth, d_mask):
        c0 = self.conv1(rgb)
        c1 = self.conv2(c0)
        d0 = self.conv1(depth)
        d1 = self.conv2(d0)
        res = torch.abs(c1 - d1)

        d_enh = c1 * d_mask * 0.5 + d1
        d_enh1 = self.rdb(d_enh)

        d_max, _ = torch.max(d1, dim=1, keepdim=True)
        d_max = self.field(d_max)
        mask = torch.sigmoid(d_max)

        out = mask * res * 0.1 + d_enh1

        return out


class LDE_GMprior4(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LDE_GMprior4, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)

        self.field = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        )
        self.rdb = nn.Sequential(
            RDB(G0=num_filter*2, C=4, G=32),
            CALayer(num_filter*2),
            nn.Conv2d(num_filter*2, num_filter, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, rgb, depth, d_mask):
        c0 = self.conv1(rgb)
        c1 = self.conv2(c0)
        d0 = self.conv1(depth)
        d1 = self.conv2(d0)
        res = torch.abs(c1 - d1)

        cat = torch.cat((c1*d_mask, d1), dim=1)
        d_enh1 = self.rdb(cat)

        d_max, _ = torch.max(d1, dim=1, keepdim=True)
        d_max = self.field(d_max)
        mask = torch.sigmoid(d_max)

        out = mask * res * 0.1 + d_enh1

        return out


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Pyramid(nn.Module):
    def __init__(self, num_filter=64, factor=8, py_num=3, kernel_size=6, stride=2, padding=2):
        super(Pyramid, self).__init__()

        self.factor = factor
        self.py_num = py_num

        self.strideConv_list = []
        for idx in range(py_num):
            if idx == 0:
                self.strideConv_list.append(
                    ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
                )
            else:
                self.strideConv_list.append(
                    ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
                )
        self.strideConv_list = nn.ModuleList(self.strideConv_list)

        self.DownConv_list = []
        for idx in range(py_num):
            if idx == 0:
                self.DownConv_list.append(
                    ConvBlock(1, 1, 1, 1, 0, activation='prelu', norm=None)
                )
            else:
                self.DownConv_list.append(
                    ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
                )
        self.DownConv_list = nn.ModuleList(self.DownConv_list)

        self.DeConv_list = []
        for idx in range(py_num):
            if idx == 0:
                self.DeConv_list.append(
                    DeconvBlock(1, 1, 1, 1, 0, activation='prelu', norm=None)
                )
            elif idx == 1:
                self.DeConv_list.append(
                    DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
                )
            else:
                self.DeConv_list.append(
                    DeconvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm=None)
                )
        self.DeConv_list = nn.ModuleList(self.DeConv_list)

        self.init_merge = []
        self.beta_conv = []
        for idx in range(py_num):
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(num_filter*2, num_filter, kernel_size=1, padding=0, bias=False),
                nn.PReLU(),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1, bias=False),
                nn.PReLU(),
                nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1, bias=False),
                nn.PReLU()
            ))

        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)

        self.res1 = nn.Sequential(
            nn.Conv2d(num_filter*py_num, num_filter, kernel_size=1, padding=0, bias=False),
            nn.PReLU(),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1, bias=False),
        )
        self.PReLu = nn.PReLU()

        self.alpha_conv = []
        for idx in range(py_num-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(num_filter*2, num_filter, kernel_size=1, stride=1, padding=0, bias=False),
                nn.PReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

    def forward(self, depth, rgb):
        b, c, h, w = depth.shape

        pyramid_feat_list = []
        depth_bin_list = []
        out_list = []

        for idx in range(self.py_num):
            if idx == 0:
                depth_bin = self.strideConv_list[idx](depth)
            else:
                depth_bin = self.strideConv_list[idx](depth_bin_list[idx-1])
            depth_bin_list.append(depth_bin)
            rgb_bin = rgb.expand((-1, -1, int(h/(2**idx)), int(w/(2**idx))))
            # rgb_bin = rgb.repeat(1, 1, h/2**idx, w/2**idx)
            merge_feat_bin = torch.cat([depth_bin, rgb_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                # pre_feat_bin = F.interpolate(pre_feat_bin, scale_factor=1/2, mode='bicubic', align_corners=False, recompute_scale_factor=True)
                pre_feat_bin = self.DownConv_list[idx](pre_feat_bin)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            pyramid_feat_list.append(merge_feat_bin)
            merge_feat_bin_out = merge_feat_bin
            if idx >= 1:
                # merge_feat_bin_out = F.interpolate(merge_feat_bin_out, scale_factor=2**idx, mode='bicubic', align_corners=False, recompute_scale_factor=True)
                merge_feat_bin_out = self.DeConv_list[idx](merge_feat_bin_out)
            out_list.append(merge_feat_bin_out)

        out = torch.cat(out_list, 1)
        out = self.res1(out)
        out = self.PReLu(self.res2(out) + out)

        return out


class Norm(nn.Module):
    def __init__(self, num_filter):
        super(Norm, self).__init__()
        self.norm = nn.Sequential(
            nn.BatchNorm2d(num_filter),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):

        return self.norm(x)


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)

        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)

        return x


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        # x0 = x[:, 0]
        # x1 = x[:, 1]
        # x2 = x[:, 2]
        # x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        # x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)
        #
        # x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        # x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)
        #
        # x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        # x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)
        # 
        # x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        # x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        # x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)
        #
        # x = torch.cat([x0, x1, x2], dim=1)
        # return x

        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=2)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=2)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x

    
class LF_Mask2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(LF_Mask2, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)
        self.conv11 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv22 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)
        self.conv3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)
        self.act = torch.nn.Sigmoid()

    def forward(self, rgb, depth):
        c0 = self.conv1(rgb)
        c1 = self.conv2(c0)
        d0 = self.conv11(depth)
        d1 = self.conv22(d0)
        res = c1 - d1
        att = self.conv3(res)
        mask = self.act(att)
        mask2 = 1 - mask
        out = mask2 * rgb + depth
        return out


class HighFusion(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(HighFusion, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)

    def forward(self, x):
        x0 = self.conv1(x)
        out = self.conv2(x0)
        return out


class HSG(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(HSG, self).__init__()
        self.conv = ConvBlock(num_filter+1, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)

        self.res = ResnetBlock(num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.relu = nn.PReLU()

    def forward(self, rep_c, depth):
        b, c, h, w = depth.shape
        cosine_eps = 1e-7
        # print(rep_c.size(),depth.size())
        sim = F.cosine_similarity(depth, rep_c[..., None, None], dim=1).contiguous().view(b, -1)  # B * (H * W)
        # print(sim.size())
        sim_mean = (sim - sim.min(1)[0].unsqueeze(1))/(sim.max(1)[0].unsqueeze(1) - sim.min(1)[0].unsqueeze(1) + cosine_eps)

        # print(sim_mean.size())
        prior = sim_mean.contiguous().view(b, 1, h, w)

        vis = prior.cpu().clone().detach()

        fus = self.conv(torch.cat((prior, depth), 1))
        out = self.relu(self.res(fus))

        return out, vis


class HSG2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(HSG2, self).__init__()

        self.rdb = RDB(G0=num_filter, C=4, G=32)
        self.relu = nn.PReLU()

    def forward(self, rep_c, depth):
        b, c, h, w = depth.shape
        cosine_eps = 1e-7
        # print(rep_c.size(),depth.size())
        sim = F.cosine_similarity(depth, rep_c[..., None, None], dim=1).contiguous().view(b, -1)  # B * (H * W)
        # print(sim.size())
        sim_mean = (sim - sim.min(1)[0].unsqueeze(1))/(sim.max(1)[0].unsqueeze(1) - sim.min(1)[0].unsqueeze(1) + cosine_eps)
        # print(sim_mean.size())
        sim_mean = torch.clamp(sim_mean, 1e-7, 1-1e-7)
        prior = sim_mean.contiguous().view(b, 1, h, w)
        fus = prior * depth + depth
        out = self.relu(self.rdb(fus))

        return out


class HSG3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(HSG3, self).__init__()
        self.conv = nn.Conv2d(num_filter+1, num_filter, kernel_size=1, stride=1, padding=0, bias=False)

        self.res = ResnetBlock(num_filter+1, kernel_size, stride, padding, activation='prelu', norm=None)
        self.ca = CA_layer(num_filter, num_filter+1, 16)



    def forward(self, rep_c, depth):
        b, c, h, w = depth.shape
        cosine_eps = 1e-7
        # print(rep_c.size(),depth.size())
        sim = F.cosine_similarity(depth, rep_c[..., None, None], dim=1).contiguous().view(b, -1)  # B * (H * W)
        # print(sim.size())
        sim_mean = (sim - sim.min(1)[0].unsqueeze(1))/(sim.max(1)[0].unsqueeze(1) - sim.min(1)[0].unsqueeze(1) + cosine_eps)

        # print(sim_mean.size())
        prior = sim_mean.contiguous().view(b, 1, h, w)

        fus = self.res(torch.cat((prior, depth), 1))
        att = self.ca(rep_c)

        out = self.conv(att * fus) + depth

        return out


class HSG4(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(HSG4, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(num_filter + 1, num_filter, kernel_size=1, stride=1, padding=0, bias=True),
            ResnetBlock(num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        )
        self.ca = CA_layer(num_filter, num_filter, 16)

    def forward(self, rep_c, depth):
        b, c, h, w = depth.shape
        cosine_eps = 1e-7
        # print(rep_c.size(),depth.size())
        sim = F.cosine_similarity(depth, rep_c[..., None, None], dim=1).contiguous().view(b, -1)  # B * (H * W)
        # print(sim.size())
        sim_mean = (sim - sim.min(1)[0].unsqueeze(1))/(sim.max(1)[0].unsqueeze(1) - sim.min(1)[0].unsqueeze(1) + cosine_eps)

        # print(sim_mean.size())
        prior = sim_mean.contiguous().view(b, 1, h, w)

        fus = self.res(torch.cat((prior, depth), 1))
        att = self.ca(rep_c)

        out = att * fus + depth

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x: degradation representation: B * C
        '''
        att = self.conv_du(x[:, :, None, None])

        return att


class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.PReLU()

    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class HighFusion2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(HighFusion2, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.res = ResnetBlock(num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation=None, norm=None)

    def forward(self, x):
        x0 = self.conv1(x)
        res = self.res(x0)
        out = self.conv2(res)
        return out


class HighSemantic(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(HighSemantic, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.s1 = SpatialAttention()

    def forward(self, x):
        x0 = self.conv1(x)
        s = self.s1(x0)
        out = x0 + x0 * s
        return out


class HighConv(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(HighConv, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.res = ResnetBlock(num_filter, kernel_size, stride, padding, activation='prelu', norm=None)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.res(x0)
        return x1


class NewFeedbackBlock1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(NewFeedbackBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.up_1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.up_1(x0)
        res1 = x - x1
        res2 = self.conv2(res1)
        out_la = x0 + res2

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        res3 = out_la - l0
        h1 = self.up_conv3(res3)
        out = h0 + h1
        return out


class NewFeedbackBlock3(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(NewFeedbackBlock3, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm=None)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        res1 = x - l00

        h0 = self.up_conv1(res1)
        l0 = self.up_conv2(h0)
        res2 = res1 - l0
        out1 = self.up_conv3(res2)
        out = out1 + h0
        return out


class FeedbackBlock1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(FeedbackBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.act_2 = torch.nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(l00 - x)
        out_la = x + 0.1 * (act1 * x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        act2 = self.act_2(h1)
        return h0 + 0.1 * (h0 * act2)


class FeedbackBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(FeedbackBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.act_2 = torch.nn.ReLU(True)

    def forward(self, x):
        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(l00 - x)
        out_la = x + 0.1 * (act1 * x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        act2 = self.act_2(h1)
        return h0 + 0.1 * (h0 * act2)


class AAP_alter1(torch.nn.Module):
    def __init__(self, scale, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(AAP_alter1, self).__init__()
        self.up_conv1 = DeconvBlock(in_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        # self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.down1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.up_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.act = torch.nn.ReLU(True)

    def forward(self, x):
        x_hr = self.up_conv1(x)
        # x_hr1 = self.conv1(x_hr)
        x_down = self.down1(x_hr)
        x_up = self.up_conv2(x_down)

        sub = x_up - x_hr
        # sub = x_hr - x_up  # x4 x8 x16 case RGBDD

        act = self.act(sub) + 1e-7

        return x_hr + act * x_hr


class AAP_alter2(torch.nn.Module):
    def __init__(self, scale, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(AAP_alter2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.act = torch.nn.ReLU(True)
        self.scale = scale
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x_down = self.down1(x)
        # x_down = self.conv1(x_down)
        x_up = self.up_conv1(x_down)
        sub = x_up - x

        act = self.act(sub) + 1e-7

        return x + act * x


class AttentionProjection1(torch.nn.Module):
    def __init__(self, scale, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(AttentionProjection1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)
        self.scale = scale

        self.up_conv1 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_conv2 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_conv4 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_x1 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_x2 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_x3 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_x4 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        sub1 = l00 - x
        act1 = self.act_1(sub1)

        if self.scale == 4:
            h0 = self.up_conv1(act1)
            h = self.up_conv2(h0)
            hx1 = self.up_x1(x)
            hx = self.up_x2(hx1)
        elif self.scale == 8:
            h0 = self.up_conv1(act1)
            h1 = self.up_conv2(h0)
            h = self.up_conv3(h1)
            hx1 = self.up_x1(x)
            hx2 = self.up_x2(hx1)
            hx = self.up_x3(hx2)
        elif self.scale == 16:
            h0 = self.up_conv1(act1)
            h1 = self.up_conv2(h0)
            h2 = self.up_conv3(h1)
            h = self.up_conv4(h2)
            hx1 = self.up_x1(x)
            hx2 = self.up_x2(hx1)
            hx3 = self.up_x3(hx2)
            hx = self.up_x4(hx3)
            
        return hx + h * hx


class AttentionProjection2(torch.nn.Module):
    def __init__(self, scale, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(AttentionProjection2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)
        self.scale = scale

        self.up_conv1 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_conv2 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)
        self.up_conv4 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation, norm=None)

    def forward(self, x):
        x1 = self.down1(x)
        x1 = self.conv1(x1)
        p1 = self.avgpool_1(x1)
        l00 = self.up_1(p1)
        sub1 = l00 - x1
        act1 = self.act_1(sub1)
        # out_la = x + 0.1 * (act1 * sub1 + sub1)

        if self.scale == 4:
            h0 = self.up_conv1(act1)
            h = self.up_conv2(h0)
        elif self.scale == 8:
            h0 = self.up_conv1(act1)
            h1 = self.up_conv2(h0)
            h = self.up_conv3(h1)
        elif self.scale == 16:
            h0 = self.up_conv1(act1)
            h1 = self.up_conv2(h0)
            h2 = self.up_conv3(h1)
            h = self.up_conv4(h2)

        return x + h * x


class channel_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(channel_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)

    def forward(self, x):
        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x + x * act2

        return y


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, stride=1, padding=padding, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return torch.clamp(self.sigmoid(x), 1e-7, 1-1e-7)
        # return self.sigmoid(x)


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_channels, sqz_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pooling = torch.nn.AdaptiveMaxPool2d(1)
        self.fc_1 = FC(in_channels, in_channels // sqz_ratio, False, True)
        self.fc_2 = FC(in_channels // sqz_ratio, in_channels, False, False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # ftr: [B, C, H, W]
        avg_out = self.avg_pooling(x).squeeze(-1).squeeze(-1)  # [B, C]
        max_out = self.max_pooling(x).squeeze(-1).squeeze(-1)  # [B, C]
        avg_weights = self.fc_2(self.fc_1(avg_out))  # [B, C]
        max_weights = self.fc_2(self.fc_1(max_out))  # [B, C]
        weights = self.sigmoid(avg_weights + max_weights)  # [B, C]
        return x * weights.unsqueeze(-1).unsqueeze(-1) + x
