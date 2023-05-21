# coding=utf-8
"""
@Author : wangkangli
@Email: 455389059@qq.com
@createtime : 2023-05-21 16:06
"""


import numpy as np
import pandas as pd
import math
import torch
from torch import nn



class Spatblock_spilt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Spatblock_spilt, self).__init__()
        self.dilaconv1 = Sparable(in_channels, in_channels)
        self.dilaconv2 = Sparable(in_channels, in_channels)
        self.dilaconv3 = Sparable(in_channels, in_channels // 2)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels // 2, out_channels=out_channels, kernel_size=1, stride=1)

        self.att = eca_layer(in_channels // 2, k_size=3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 1))
        self.shortcut = nn.quantized.FloatFunctional()

    def forward(self, x):
        a1, a2 = x.chunk(2, dim=3)
        x1 = self.dilaconv1(a1)
        x2 = self.dilaconv2(x1)
        x3 = self.dilaconv3(x2)
        o1 = self.conv1(a1)
        o2 = self.conv2(a2)
        o3 = self.shortcut.add(o1, o2)
        o3 = self.shortcut.add(o3, x3)
        o3 = self.att(o3)
        o3 = self.conv3(o3)
        o3 = self.norm(o3)
        o3 = self.relu(o3)
        o3 = self.pool(o3)

        return o3


class Spatblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Spatblock, self).__init__()
        self.dilaconv1 = Sparable(in_channels, in_channels)
        self.dilaconv2 = Sparable(in_channels, in_channels)
        self.dilaconv3 = Sparable(in_channels, in_channels // 2)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=3 * in_channels // 2, out_channels=out_channels, kernel_size=1, stride=1)

        self.att = eca_layer(3 * in_channels // 2, k_size=3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.dilaconv1(x)
        x2 = self.dilaconv2(x1)
        x3 = self.dilaconv3(x2)
        o1 = self.conv1(x)
        o2 = self.conv2(x1)
        o3 = torch.cat((o1, o2, x3), dim=1)
        o3 = self.att(o3)
        o3 = self.conv3(o3)
        o3 = self.norm(o3)
        o3 = self.relu(o3)
        o3 = self.pool(o3)

        return o3


class Sparable(nn.Module):
    def __init__(self, cin, cout, p=0.25, min_mid_channels=4):
        """

        :param cin:
        :param cout:
        """
        super(Sparable, self).__init__()

        assert 0.0 <= p <= 1.0
        mid_channels = int(min(max(cin, 4), max(min_mid_channels, math.ceil(p * cin))))
        # print(mid_channels)
        # --- 1 x 1 conv
        self.ch_conv1 = nn.Conv2d(
            cin,
            cout,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=cout,
            bias=False,
            padding_mode="zeros",
        )

        # --- Spatial convolution channel-wise convolution
        # 1 x 3 with dilation 2
        self.sp_conv1 = nn.Conv2d(
            cout,
            cout,
            kernel_size=(3, 1),
            stride=1,
            padding=(2, 0),
            dilation=(2, 1),
            groups=cout,
            bias=False,
            padding_mode="zeros",
        )
        # 3 x 1 with dilation 2
        self.sp_conv2 = nn.Conv2d(
            cout,
            cout,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 2),
            dilation=(1, 2),
            groups=cout,
            bias=False,
            padding_mode="zeros",
        )

        self.norm = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.quantized.FloatFunctional()
        # self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # x = self.pw1(x)
        # x = self.pw2(x)
        x = self.ch_conv1(x)
        x = self.shortcut.add(self.sp_conv1(x), self.sp_conv2(x))
        # x = self.sp_conv1(x) + self.sp_conv2(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class depthwise_conv2d(nn.Module):
    def __init__(self, n_in, n_out, stride):
        super(depthwise_conv2d, self).__init__()
        self.depth_wise = nn.Conv2d(n_in, n_in, kernel_size=3, stride=stride, padding=1, groups=n_in)
        self.point_wise = nn.Conv2d(n_in, n_out, kernel_size=1)

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, in_channels, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.shortcut = nn.quantized.FloatFunctional()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return self.shortcut.mul(x, y.expand_as(x))


class Cnn(nn.Module):
    def __init__(
            self,
            channels=[24, 32, 64, 64, 64, 64],
    ):
        """

        :param mel_bins:
        :param channels:
        """
        super(Cnn, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.shortcut = nn.quantized.FloatFunctional()
        self.dequant = torch.quantization.DeQuantStub()



        self.conv = nn.Conv2d(2, 16, kernel_size=1)
        # self.conv = Sparable(1,16)
        self.conv1 = Spatblock(16, channels[0])
        self.conv2 = Spatblock(channels[0], channels[1])
        self.conv3 = Spatblock(channels[1], channels[2])
        self.conv4 = Spatblock(channels[2], channels[3])
        self.conv5 = Spatblock(channels[3], channels[4])
        self.conv6 = Spatblock(channels[4], channels[5])

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier1

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(224, 100)
        self.l2 = nn.Linear(100, 13)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        """Input size - (batch_size, 1, time_steps, mel_bins)  """


        x = self.quant(x)
        x = x.transpose(2, 3)
        x = self.conv(x)

        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(feat4)
        # feat6 = self.conv6(feat5)

        m1 = self.pool(feat2)
        o1 = self.pool(feat3)
        o2 = self.pool(feat4)
        o3 = self.pool(feat5)



        x_1 = torch.cat((m1, o1, o2, o3), dim=1)

        x_1 = self.flatten(x_1)
        x_1 = self.relu(self.l1(x_1))
        x_1 = self.dropout(x_1)
        x_1 = self.l2(x_1)
        x_1 = self.dequant(x_1)

        return x_1




model = Cnn()
model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
