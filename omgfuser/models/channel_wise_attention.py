"""Original Operation-Wise Attention Fusion implementation.

Presented in paper:
    Charitidis, P., Kordopatis-Zilos, G., Papadopoulos, S., & Kompatsiaris, I.
    (2021, June). Operation-wise attention network for tampering localization
    fusion. In 2021 International Conference on Content-Based Multimedia
    Indexing (CBMI) (pp. 1-6). IEEE.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Fusion(nn.Module):
    def __init__(
        self, input_maps=10, steps=2, num_layers=2, channel_dropout=0.05, attention="v2"
    ):
        super(Fusion, self).__init__()
        self.input_maps = input_maps
        self.input_channels = self.input_maps * 3  # Each map is an RGB one.
        self.groups = self.input_maps
        self.steps = steps
        self.num_layers = num_layers
        self.num_ops = len(Operations)
        self.channel_dropout = nn.Dropout2d(p=channel_dropout)
        self.kernel_size = 3
        self.FEB = nn.Sequential(
            # Is this convolution needed at all? It seems to only learn to convert rgb to
            # grayscale. Why not to provide the RGB input immediately?
            nn.Conv2d(
                self.input_channels,
                self.input_maps,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=self.groups,
            ),
            ResBlock(self.input_maps, self.input_maps, self.kernel_size, 1, 1, False),
            ResBlock(self.input_maps, self.input_maps, self.kernel_size, 1, 1, False),
            ResBlock(self.input_maps, self.input_maps, self.kernel_size, 1, 1, False),
            ResBlock(self.input_maps, self.input_maps, self.kernel_size, 1, 1, False),
        )
        if attention == "v2":
            self.CWALayer = CWALayerv2(self.input_maps, 128)
        elif attention == "v3":
            self.CWALayer = CWALayerv3(self.input_maps)
        elif attention == "v1":
            self.CWALayer = CWALayer(self.input_maps)
        else:
            raise NotImplementedError

        # a stack of operation-wise attention layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            attention = OALayer(self.input_maps, self.steps, self.num_ops)
            self.layers += [attention]
            layer = GroupOLs(self.steps, self.input_maps)
            self.layers += [layer]

        # Output layer
        self.conv2 = nn.Conv2d(
            self.input_maps, 1, self.kernel_size, padding=1, bias=False
        )
        self.out_sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> dict[str, Optional[torch.Tensor]]:
        x = self.FEB(x)
        x = self.channel_dropout(x)
        att = self.CWALayer(x)
        att = nn.Softmax(-1)(att)
        x = torch.einsum("bchw,bc->bchw", x, att)
        for _, layer in enumerate(self.layers):
            if isinstance(layer, OALayer):
                weights = layer(x)
                weights = F.softmax(weights, dim=-1)
            else:
                x = layer(x, weights)
                # print(x.shape)

        x = self.conv2(x)

        x = self.out_sigmoid(x)

        out: dict[str, Optional[torch.Tensor]] = {
            "localization": x,
            "detection": torch.amax(x, dim=(2, 3)),
            "attention": att if return_attention else None
        }
        return out


class CWALayerv3(nn.Module):
    def __init__(self, num_channels):
        super(CWALayerv3, self).__init__()
        self.num_channels = num_channels
        self.block = ResBlock(num_channels, num_channels, 3, 1, 1, affine=True)
        self.pool = nn.AvgPool2d(8)
        self.norm = nn.InstanceNorm2d(num_channels, affine=True)
        self.register_buffer("mask", 1 - torch.eye(num_channels, num_channels))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.pool(x)
        x = self.block(x)
        x = self.norm(x)

        keys = x.view(b, c, -1)
        queries = x.view(b, c, -1)

        attention = torch.matmul(queries, keys.transpose(2, 1))

        attention = torch.sigmoid(attention)
        attention = self.mask * attention
        attention = F.adaptive_avg_pool1d(attention, 1).flatten(1)
        return attention


class CWALayerv2(nn.Module):
    def __init__(self, num_channels, dim=128, spatial_dims=[128, 64, 32]):
        super(CWALayerv2, self).__init__()
        self.num_channels = num_channels
        self.block = ResBlock(num_channels, num_channels, 3, 1, 1, affine=True)
        self.pool = nn.AvgPool2d(8)
        self.norm = nn.InstanceNorm2d(num_channels, affine=True)
        self.keys = nn.Linear(32 * 32, dim)
        self.queries = nn.Linear(32 * 32, dim)

        self.combination = nn.Linear(num_channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.pool(x)
        x = self.block(x)
        x = self.norm(x)

        keys = self.keys(x.view(b, c, -1))
        queries = self.queries(x.view(b, c, -1))

        attention = torch.matmul(queries, keys.transpose(2, 1))
        attention = torch.sigmoid(attention)
        attention = F.adaptive_avg_pool1d(attention, 1).flatten(1)
        return attention


class CWALayer(nn.Module):
    def __init__(self, num_channels):
        super(CWALayer, self).__init__()
        self.num_channels = num_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(self.num_channels, self.num_channels * 2),
            nn.ReLU(),
            nn.Linear(self.num_channels * 2, self.num_channels),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        # print(y.shape)
        return y


Operations = [
    "sep_conv_1x1",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "sep_conv_7x7",
    "dil_conv_3x3",
    "dil_conv_5x5",
    "dil_conv_7x7",
    "conv_3x3",
    "conv_5x5",
    "conv_7x7",
    "avg_pool_3x3",
]

OPS = {
    "avg_pool_3x3": lambda c, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "sep_conv_1x1": lambda c, stride, affine: SepConv(
        c, c, 1, stride, 0, affine=affine
    ),
    "sep_conv_3x3": lambda c, stride, affine: SepConv(
        c, c, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5": lambda c, stride, affine: SepConv(
        c, c, 5, stride, 2, affine=affine
    ),
    "sep_conv_7x7": lambda c, stride, affine: SepConv(
        c, c, 7, stride, 3, affine=affine
    ),
    "dil_conv_3x3": lambda c, stride, affine: DilConv(
        c, c, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5": lambda c, stride, affine: DilConv(
        c, c, 5, stride, 4, 2, affine=affine
    ),
    "dil_conv_7x7": lambda c, stride, affine: DilConv(
        c, c, 7, stride, 6, 2, affine=affine
    ),
    "conv_3x3": lambda c, stride, affine: ReLUConv(c, c, 3, stride, 1, affine=affine),
    "conv_5x5": lambda c, stride, affine: ReLUConv(c, c, 5, stride, 2, affine=affine),
    "conv_7x7": lambda c, stride, affine: ReLUConv(c, c, 7, stride, 3, affine=affine),
}


class ReLUConvBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class ReLUConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(
                c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(
        self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            c_in,
            c_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=c_in,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            c_in,
            c_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=c_in,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class SepConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.op(x)


class OperationLayer(nn.Module):
    def __init__(self, c, stride):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        for o in Operations:
            op = OPS[o](c, stride, False)
            self._ops.append(op)

        self._out = nn.Sequential(
            nn.Conv2d(c * len(Operations), c, 1, padding=0, bias=False), nn.ReLU()
        )

    def forward(self, x, weights):
        weights = weights.transpose(1, 0)
        states = []
        for w, op in zip(weights, self._ops):
            # print(w.shape)
            states.append(op(x) * w.view([-1, 1, 1, 1]))
        return self._out(torch.cat(states[:], dim=1))


# a Group of operation layers
class GroupOLs(nn.Module):
    def __init__(self, steps, c):
        super(GroupOLs, self).__init__()
        self.preprocess = ReLUConv(c, c, 1, 1, 0, affine=False)
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        stride = 1

        for _ in range(self._steps):
            op = OperationLayer(c, stride)
            self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess(s0)
        for i in range(self._steps):
            res = s0
            # print(weights.shape,weights[:, i, :].shape,s0.shape)
            s0 = self._ops[i](s0, weights[:, i, :])
            s0 = self.relu(s0 + res)
        return s0


# Operation-wise Attention Layer (OWAL)
class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channel, self.output * 2),
            nn.ReLU(),
            nn.Linear(self.output * 2, self.k * self.num_ops),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y
