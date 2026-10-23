# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import time
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, GroupConv
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'MANet', 'RepMANetBaseline', 'RepMANet', 'C2f',
           'C3x', 'C3TR', 'C3Ghost','GhostBottleneck', 'Bottleneck', 'RepBottleneck', 'RepMixConv', 'RepMixBottleneck',
           'BottleneckCSP', 'Proto', 'RepC3', 'RFHM')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))





class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(Conv(c1, c2, k=7, s=2, p=3, act=True),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MANet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=0.5
    ):
        super().__init__(
            c1,
            c2,
            shortcut,
            g,
            k,
            e
        )

        c_ = int(c2 * e)

        self.cv1 = RepConv(
            c1,
            c_,
            k[0],
            1
        )


class RepMANetBaseline(nn.Module):
    """
    Simple 3x3 + 1x1 reparameterization ablation
    that preserves MANet topology.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5
    ):
        super().__init__()

        self.c = int(c2 * e)

        self.cv_first = Conv(
            c1,
            2 * self.c,
            1,
            1
        )

        self.cv_final = Conv(
            (4 + n) * self.c,
            c2,
            1
        )

        self.m = nn.ModuleList(
            RepBottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=((3, 3), (3, 3)),
                e=1.0
            )
            for _ in range(n)
        )

        self.cv_block_1 = Conv(
            2 * self.c,
            self.c,
            1,
            1
        )

        dim_hid = int(
            p * 2 * self.c
        )

        self.cv_block_2 = nn.Sequential(
            Conv(
                2 * self.c,
                dim_hid,
                1,
                1
            ),
            GroupConv(
                dim_hid,
                dim_hid,
                kernel_size,
                1
            ),
            Conv(
                dim_hid,
                self.c,
                1,
                1
            )
        )

    def forward(self, x):
        y = self.cv_first(x)

        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)

        y2, y3 = y.chunk(2, 1)

        y = [
            y0,
            y1,
            y2,
            y3
        ]

        y.extend(
            m(y[-1])
            for m in self.m
        )

        return self.cv_final(
            torch.cat(
                y,
                1
            )
        )


class RepMixConv(nn.Module):
    """
    Training structure:

        Full 3x3 + BN
            +
        1x1 + BN
        -> SUM
        -> activation

    Deployment structure:

        Single fused 3x3 convolution
        -> activation
    """

    def __init__(
        self,
        c1,
        c2,
        s=1,
        g=1,
        act=True
    ):
        super().__init__()

        if c1 % g != 0:
            raise ValueError(
                f"c1={c1} must be divisible by g={g}"
            )

        if c2 % g != 0:
            raise ValueError(
                f"c2={c2} must be divisible by g={g}"
            )

        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.g = g

        # =====================================================
        # Branch 1
        # Full 3x3 convolution
        # =====================================================

        self.rbr_3x3 = nn.Sequential(
            nn.Conv2d(
                c1,
                c2,
                kernel_size=3,
                stride=s,
                padding=1,
                groups=g,
                bias=False
            ),
            nn.BatchNorm2d(c2)
        )

        # =====================================================
        # Branch 2
        # 1x1 convolution
        # =====================================================

        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(
                c1,
                c2,
                kernel_size=1,
                stride=s,
                padding=0,
                groups=g,
                bias=False
            ),
            nn.BatchNorm2d(c2)
        )

        # =====================================================
        # Activation
        # =====================================================

        if act is True:
            self.act = (
                Conv.default_act
                if hasattr(
                    Conv,
                    "default_act"
                )
                else nn.SiLU()
            )

        elif isinstance(
            act,
            nn.Module
        ):
            self.act = act

        else:
            self.act = nn.Identity()

        # =====================================================
        # Stable initialization
        #
        # Initial state:
        #
        # RepMixConv(x)
        # ≈ Full3x3(x)
        #
        # The 1x1 branch gradually participates in optimization.
        # =====================================================

        nn.init.zeros_(
            self.rbr_1x1[1].weight
        )

        nn.init.zeros_(
            self.rbr_1x1[1].bias
        )

    def forward(self, x):

        if hasattr(
            self,
            "rbr_reparam"
        ):
            return self.forward_fuse(x)

        # =====================================================
        # Main full 3x3 branch
        # =====================================================

        branch_3x3 = self.rbr_3x3(x)

        # =====================================================
        # 1x1 branch
        # =====================================================

        branch_1x1 = self.rbr_1x1(x)

        # =====================================================
        # Multi-branch aggregation
        # =====================================================

        out = (
            branch_3x3
            +
            branch_1x1
        )

        return self.act(out)

    def forward_fuse(self, x):

        return self.act(
            self.rbr_reparam(x)
        )

    @staticmethod
    def _fuse_conv_bn_pair(
        conv,
        bn
    ):

        kernel = conv.weight

        if conv.bias is None:

            bias = torch.zeros(
                conv.out_channels,
                device=kernel.device,
                dtype=kernel.dtype
            )

        else:

            bias = conv.bias

        std = torch.sqrt(
            bn.running_var
            +
            bn.eps
        )

        scale = (
            bn.weight
            /
            std
        )

        kernel_fused = (
            kernel
            *
            scale.reshape(
                -1,
                1,
                1,
                1
            )
        )

        bias_fused = (
            bn.bias
            +
            (
                bias
                -
                bn.running_mean
            )
            *
            scale
        )

        return (
            kernel_fused,
            bias_fused
        )

    @staticmethod
    def _fuse_conv_bn(branch):

        return (
            RepMixConv
            ._fuse_conv_bn_pair(
                branch[0],
                branch[1]
            )
        )

    @staticmethod
    def _pad_1x1_to_3x3(
        kernel
    ):

        if (
            kernel.shape[-2:]
            !=
            (1, 1)
        ):
            raise ValueError(
                "Expected a 1x1 kernel."
            )

        return F.pad(
            kernel,
            [1, 1, 1, 1]
        )

    def get_equivalent_kernel_bias(
        self
    ):

        k3, b3 = (
            self._fuse_conv_bn(
                self.rbr_3x3
            )
        )

        k1, b1 = (
            self._fuse_conv_bn(
                self.rbr_1x1
            )
        )

        k1 = (
            self._pad_1x1_to_3x3(
                k1
            )
        )

        kernel = (
            k3
            +
            k1
        )

        bias = (
            b3
            +
            b1
        )

        return (
            kernel,
            bias
        )

    @torch.no_grad()
    def fuse_convs(self):

        if hasattr(
            self,
            "rbr_reparam"
        ):
            return

        self.eval()

        kernel, bias = (
            self
            .get_equivalent_kernel_bias()
        )

        # =====================================================
        # Single deploy convolution
        # =====================================================

        self.rbr_reparam = nn.Conv2d(
            self.c1,
            self.c2,
            kernel_size=3,
            stride=self.s,
            padding=1,
            groups=self.g,
            bias=True
        ).to(
            device=kernel.device,
            dtype=kernel.dtype
        )

        self.rbr_reparam.weight.copy_(
            kernel
        )

        self.rbr_reparam.bias.copy_(
            bias
        )

        # =====================================================
        # Remove training branches
        # =====================================================

        del self.rbr_3x3
        del self.rbr_1x1

        self.rbr_reparam.eval()

    def switch_to_deploy(self):

        self.fuse_convs()


class RepMixBottleneck(nn.Module):
    """
    Information-flow-enhanced bottleneck.

    Main path:

        X
        |
        RepMixConv
        |
        Conv3x3
        |
        F(X)

    Residual path:

        X --------------------+
                              |
                              +
                              |
                            Output

    Residual is enabled only when:

        shortcut=True
        and
        c1 == c2
    """

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=((3, 3), (3, 3)),
        e=0.5
    ):
        super().__init__()

        c_ = int(
            c2 * e
        )

        k1 = k[0]

        if k1 not in (
            3,
            (3, 3)
        ):
            raise ValueError(
                "RepMixBottleneck currently "
                "requires the first kernel "
                "to be 3x3, "
                f"but got {k1}"
            )

        self.cv1 = RepMixConv(
            c1,
            c_,
            s=1,
            g=1,
            act=True
        )

        self.cv2 = Conv(
            c_,
            c2,
            k[1],
            1,
            g=g
        )

        self.add = (
            shortcut
            and
            c1 == c2
        )

    def forward(self, x):

        y = self.cv2(
            self.cv1(x)
        )

        if self.add:
            return x + y

        return y


class LocalDetailResidual(nn.Module):
    """
    Lightweight local-detail enhancement.

    Structure:

        X ------------------------------+
        |                               |
        +-> DW3x3 -> BN ----------------+
                                        |
                                        v
                                      Output

    Initial state:

        BN gamma = 0

    therefore:

        Output ~= X

    at the beginning of training.

    No pointwise convolution is introduced.
    Parameter cost is approximately:

        9 * C

    convolution weights.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False
        )

        self.bn = nn.BatchNorm2d(
            channels
        )

        # =====================================================
        # Zero initialization
        #
        # Initial behavior:
        #
        # y = x + 0
        #
        # This preserves the original MANet information
        # pathway at the beginning of training.
        # =====================================================

        nn.init.zeros_(
            self.bn.weight
        )

        nn.init.zeros_(
            self.bn.bias
        )

    def forward(self, x):

        return (
            x
            +
            self.bn(
                self.dw(x)
            )
        )


class RepMANet(nn.Module):
    """
    Re-parameterized Mixed Aggregation Network (RepMANet).

    Four initial paths:

        y0:
            Original channel interaction
            1x1 Conv

        y1:
            Re-parameterized spatial feature extraction
            1x1
            -> RepConv
            -> 1x1

        y2:
            Original information-preserving path

        y3:
            Original information-preserving path

    Deep path:

        y3
        -> RepMixBottleneck
        -> RepMixBottleneck
        -> ...

    Final:

        Concat(
            y0,
            y1,
            y2,
            y3,
            y4,
            ...
        )
        -> 1x1 Conv
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        p=1,
        kernel_size=3,
        g=1,
        e=0.5
    ):
        super().__init__()

        self.c = int(
            c2 * e
        )

        # =====================================================
        # Initial channel projection
        # =====================================================

        self.cv_first = Conv(
            c1,
            2 * self.c,
            1,
            1
        )

        # =====================================================
        # Final aggregation
        #
        # y0
        # y1
        # y2
        # y3
        # +
        # n deep features
        #
        # total:
        #
        # (4 + n) * c
        # =====================================================

        self.cv_final = Conv(
            (4 + n) * self.c,
            c2,
            1
        )

        # =====================================================
        # Progressive deep extraction
        # =====================================================

        self.m = nn.ModuleList(
            RepMixBottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=(
                    (3, 3),
                    (3, 3)
                ),
                e=1.0
            )
            for _ in range(n)
        )

        # =====================================================
        # Branch y0
        #
        # Channel interaction branch
        # =====================================================

        self.cv_block_1 = Conv(
            2 * self.c,
            self.c,
            1,
            1
        )

        # =====================================================
        # Branch y1
        #
        # Re-parameterized spatial extraction branch
        # =====================================================

        dim_hid = int(
            p
            *
            2
            *
            self.c
        )

        self.cv_block_2 = nn.Sequential(
            Conv(
                2 * self.c,
                dim_hid,
                1,
                1
            ),

            RepConv(
                dim_hid,
                dim_hid,
                k=3,
                s=1,
                p=1,
                g=dim_hid
            ),

            Conv(
                dim_hid,
                self.c,
                1,
                1
            )
        )

    def forward(self, x):

        # =====================================================
        # Initial projection
        # =====================================================

        y = self.cv_first(x)

        # =====================================================
        # Branch 0
        # Channel interaction
        # =====================================================

        y0 = self.cv_block_1(y)

        # =====================================================
        # Branch 1
        # Re-parameterized spatial extraction
        # =====================================================

        y1 = self.cv_block_2(y)

        # =====================================================
        # Split original feature
        # =====================================================

        y2, y3 = y.chunk(
            2,
            1
        )

        # =====================================================
        # Branch 3
        #
        # y3 remains unchanged.
        #
        # It serves simultaneously as:
        #
        # 1. information-preserving output
        # 2. starting point of deep progressive path
        # =====================================================

        outputs = [
            y0,
            y1,
            y2,
            y3
        ]

        # =====================================================
        # Progressive serial deep extraction
        #
        # y3
        #  |
        #  v
        # M1 -> y4
        #  |
        #  v
        # M2 -> y5
        #  |
        #  v
        # ...
        # =====================================================

        for m in self.m:

            outputs.append(
                m(outputs[-1])
            )

        # =====================================================
        # Multi-path aggregation
        # =====================================================

        out = torch.cat(
            outputs,
            dim=1
        )

        # =====================================================
        # Final channel fusion
        # =====================================================

        return self.cv_final(
            out
        )
import math
import numbers

import torch
import torch.nn as nn

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageAgg(nn.Module):
    'MessageAgg.'

    def __init__(
        self,
        agg_method="mean",
        eps=1e-6,
    ):
        super().__init__()

        self.agg_method = agg_method
        self.eps = eps

    def forward(
        self,
        X,
        path,
    ):
        path = path.to(
            device=X.device,
            dtype=X.dtype,
        )

        out = torch.matmul(
            path,
            X,
        )

        if self.agg_method == "mean":
            norm = path.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(
                self.eps
            )

            out = out / norm

        elif self.agg_method == "sum":
            pass

        else:
            raise ValueError(
                f"Unsupported aggregation method: "
                f"{self.agg_method}"
            )

        return out


class HyPConv(nn.Module):
    'HyPConv.'

    def __init__(
        self,
        c1,
        c2,
    ):
        super().__init__()

        self.fc = nn.Linear(
            c1,
            c2,
        )

        self.v2e = MessageAgg(
            agg_method="mean"
        )

        self.e2v = MessageAgg(
            agg_method="mean"
        )

    def forward(
        self,
        x,
        H,
    ):
        'forward.'

        # [B, N, C1] -> [B, N, C2]
        x = self.fc(
            x
        )

        # --------------------------------------------------
        # vertex -> hyperedge
        #
        # H.transpose(1, 2):
        #     [B, M, N]
        # --------------------------------------------------
        edge_feature = self.v2e(
            x,
            H.transpose(
                1,
                2,
            ).contiguous(),
        )

        # --------------------------------------------------
        # hyperedge -> vertex
        #
        # H:
        #     [B, N, M]
        # --------------------------------------------------
        out = self.e2v(
            edge_feature,
            H,
        )

        return out


class RFHM(nn.Module):
    'RFHM: Relational Field Hypergraph Module with GSR, LSGM, AMC and LRG.'

    def __init__(
        self,
        c1,
        c2,
        threshold=None,
    ):
        super().__init__()

        self.c1 = c1
        self.c2 = c2

        
        self.threshold = threshold

        self.eps = 1e-6

        self.hgconv = HyPConv(
            c1,
            c2,
        )

        # --------------------------------------------------
        
        #
        # sigmoid(-1.5) ≈ 0.182
        #
        
        
        # --------------------------------------------------
        self.residual_scale_logit = nn.Parameter(
            torch.tensor(
                -2,
                dtype=torch.float32,
            )
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        if c1 == c2:
            self.shortcut = nn.Identity()

        else:
            self.shortcut = nn.Linear(
                c1,
                c2,
                bias=False,
            )

        self.bn = nn.BatchNorm2d(
            c2
        )

        self.act = nn.SiLU()

    def _global_threshold_sq(
        self,
        feature,
    ):
        'GSR: estimate the global squared relation scale.'

        batch_size, num_nodes, _ = feature.shape

        if num_nodes <= 1:
            return torch.zeros(
                batch_size,
                1,
                1,
                device=feature.device,
                dtype=feature.dtype,
            )

        
        global_mean = feature.mean(
            dim=1,
            keepdim=True,
        )

        
        global_radius_sq = (
            feature
            - global_mean
        ).pow(
            2
        ).sum(
            dim=-1
        )

        
        correction = (
            2.0
            * float(num_nodes)
            / float(
                num_nodes
                - 1
            )
        )

        global_threshold_sq = (
            correction
            * global_radius_sq.mean(
                dim=1,
                keepdim=True,
            )
        )

        # [B, 1] -> [B, 1, 1]
        return global_threshold_sq.unsqueeze(
            -1
        )

    def _local_threshold_sq(
        self,
        feature,
        h,
        w,
    ):
        'LSGM: estimate local squared scales in fixed 3x3 neighborhoods.'

        batch_size, num_nodes, channels = (
            feature.shape
        )

        if num_nodes != h * w:
            raise ValueError(
                f"Node number {num_nodes} does not match "
                f"spatial shape {h}x{w}."
            )

        # [B, N, C] -> [B, C, H, W]
        feature_2d = feature.transpose(
            1,
            2,
        ).contiguous().view(
            batch_size,
            channels,
            h,
            w,
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        local_mean = F.avg_pool2d(
            feature_2d,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        local_mean_sq = F.avg_pool2d(
            feature_2d.pow(
                2
            ),
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )

        # --------------------------------------------------
        # Var(x) = E[x²] - E[x]²
        # --------------------------------------------------
        local_variance = (
            local_mean_sq
            - local_mean.pow(
                2
            )
        ).clamp_min(
            0.0
        )

        
        local_radius_sq = local_variance.sum(
            dim=1,
            keepdim=True,
        )

        # --------------------------------------------------
        
        #
        
        
        
        # --------------------------------------------------
        count_input = torch.ones(
            1,
            1,
            h,
            w,
            device=feature.device,
            dtype=feature.dtype,
        )

        count_kernel = torch.ones(
            1,
            1,
            3,
            3,
            device=feature.device,
            dtype=feature.dtype,
        )

        local_count = F.conv2d(
            count_input,
            count_kernel,
            stride=1,
            padding=1,
        )

        
        local_correction = (
            2.0
            * local_count
            / (
                local_count
                - 1.0
            ).clamp_min(
                1.0
            )
        )

        local_threshold_sq = (
            local_correction
            * local_radius_sq
        )

        # [B, 1, H, W] -> [B, 1, N]
        return local_threshold_sq.flatten(
            2
        )

    def _adaptive_threshold(
        self,
        feature,
        h,
        w,
    ):
        'LSGM: modulate the global scale using relative local dispersion.'

        global_threshold_sq = (
            self._global_threshold_sq(
                feature
            )
        )

        local_threshold_sq = (
            self._local_threshold_sq(
                feature,
                h,
                w,
            )
        )

        # --------------------------------------------------
        
        #
        # [B, 1, N] -> [B, 1, 1]
        # --------------------------------------------------
        local_reference_sq = (
            local_threshold_sq.mean(
                dim=-1,
                keepdim=True,
            ).clamp_min(
                self.eps
            )
        )

        # --------------------------------------------------
        
        #
        
        #
        
        
        # --------------------------------------------------
        local_relative_sq = (
            local_threshold_sq
            / local_reference_sq
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        aligned_local_sq = (
            global_threshold_sq
            * local_relative_sq
        )

        # --------------------------------------------------
        
        #
        
        #
        
        #
        # threshold_sq =
        #     global_threshold_sq
        #     × sqrt(local_relative_sq)
        #
        
        
        # --------------------------------------------------
        threshold_sq = torch.sqrt(
            (
                global_threshold_sq
                * aligned_local_sq
            ).clamp_min(
                self.eps
            )
        )

        threshold = torch.sqrt(
            threshold_sq.clamp_min(
                self.eps
            )
        )

        return threshold

    def _build_hypergraph(
        self,
        feature,
        h,
        w,
    ):
        'AMC: construct sparse Gaussian memberships with self inclusion.'

        _, num_nodes, channels = feature.shape

        # --------------------------------------------------
        
        #
        
        
        # --------------------------------------------------
        graph_feature = F.layer_norm(
            feature.detach().float(),
            normalized_shape=(
                channels,
            ),
        )

        # [B, N, N]
        distance = torch.cdist(
            graph_feature,
            graph_feature,
            p=2,
        )

        # [B, 1, N]
        adaptive_threshold = (
            self._adaptive_threshold(
                graph_feature,
                h,
                w,
            )
        )

        threshold_sq = adaptive_threshold.pow(
            2
        ).clamp_min(
            self.eps
        )

        # --------------------------------------------------
        
        #
        
        # --------------------------------------------------
        threshold_support = (
            distance
            < adaptive_threshold
        )

        # --------------------------------------------------
        
        #
        
        # --------------------------------------------------
        weight = torch.exp(
            -distance.pow(
                2
            )
            / threshold_sq
        )

        # --------------------------------------------------
        
        #
        # N=6400 -> 80
        # N=1600 -> 40
        # N=400  -> 20
        #
        
        # --------------------------------------------------
        max_members = int(
            math.ceil(
                math.sqrt(
                    float(
                        max(
                            num_nodes,
                            1,
                        )
                    )
                )
            )
        )

        max_members = min(
            max_members,
            num_nodes,
        )

        # --------------------------------------------------
        
        #
        
        
        #
        
        #     [B, max_members, N]
        # --------------------------------------------------
        nearest_indices = torch.topk(
            distance,
            k=max_members,
            dim=1,
            largest=False,
            sorted=False,
        ).indices

        sparse_support = torch.zeros_like(
            distance
        )

        sparse_support.scatter_(
            dim=1,
            index=nearest_indices,
            value=1.0,
        )

        
        #
        
        
        
        hypergraph = (
            weight
            * threshold_support.to(
                dtype=weight.dtype
            )
            * sparse_support
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        identity = torch.eye(
            num_nodes,
            device=hypergraph.device,
            dtype=hypergraph.dtype,
        ).unsqueeze(
            0
        )

        hypergraph = torch.maximum(
            hypergraph,
            identity,
        )

        return hypergraph.to(
            device=feature.device,
            dtype=feature.dtype,
        )

    def forward(
        self,
        x,
    ):
        batch_size, _, height, width = (
            x.shape
        )

        # [B, C1, H, W] -> [B, N, C1]
        tokens = x.flatten(
            2
        ).transpose(
            1,
            2,
        ).contiguous()

        
        hypergraph = self._build_hypergraph(
            tokens,
            height,
            width,
        )

        # [B, N, C1] -> [B, N, C2]
        hyper_feature = self.hgconv(
            tokens,
            hypergraph,
        )

        
        identity = self.shortcut(
            tokens
        )

        
        residual_scale = torch.sigmoid(
            self.residual_scale_logit
        ).to(
            dtype=hyper_feature.dtype
        )

        tokens = (
            identity
            + residual_scale
            * hyper_feature
        )

        # [B, N, C2] -> [B, C2, H, W]
        out = tokens.transpose(
            1,
            2,
        ).contiguous().view(
            batch_size,
            self.c2,
            height,
            width,
        )

        out = self.act(
            self.bn(
                out
            )
        )

        return out






# Compatibility names for existing configurations and checkpoints.
RepMANetPP = RepMANet
HyperComputeModule = RFHM
