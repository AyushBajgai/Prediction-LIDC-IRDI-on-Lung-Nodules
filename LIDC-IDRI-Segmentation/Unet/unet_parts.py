"""
U-Net Component Modules
=================================

Paper Reference:
    Olaf Ronneberger, Philipp Fischer, Thomas Brox.
    "U-Net: Convolutional Networks for Biomedical Image Segmentation."
    In *Proceedings of MICCAI 2015 (Medical Image Computing and
    Computer-Assisted Intervention)*, Springer, 2015.
    DOI: https://doi.org/10.1007/978-3-319-24574-4_28

Description:
    This file implements the core modular components of the U-Net architecture:
        - DoubleConv: two consecutive convolutional layers with BatchNorm and ReLU.
        - Down: max-pooling followed by DoubleConv (encoder block).
        - Up: upsampling or transposed convolution followed by DoubleConv (decoder block).
        - OutConv: 1×1 convolution to project features into the target number of classes.

License & Attribution:
    This implementation is a re-creation for academic and educational use,
    inspired by public open-source U-Net implementations under permissive licenses.
    Please retain this citation if reused for research or coursework.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive (Conv → BatchNorm → ReLU) blocks"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool followed by DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: Upsample or Transposed Conv + DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            # Bilinear interpolation for upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Learnable upsampling (Transposed Convolution)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        # Pad to align shapes before concatenation
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1×1 convolution for class prediction"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

