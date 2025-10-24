import torch
import torch.nn as nn
import torch.nn.functional as F

# Import modular components of U-Net (DoubleConv, Down, Up, OutConv)
from .unet_parts import *


class UNet(nn.Module):
    """
    U-Net: Fully Convolutional Network for Biomedical Image Segmentation
    (Ronneberger et al., MICCAI 2015)

    Model architecture:
        Encoder (Downsampling): DoubleConv + MaxPool
        Decoder (Upsampling): Upsample/TransposeConv + Skip Connection + DoubleConv
        Output layer: 1×1 convolution mapping to the number of classes

    Args:
        n_channels:  Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        n_classes:   Number of output segmentation classes
        bilinear:    Whether to use bilinear interpolation for upsampling
                     (if False, transposed convolution is used)
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path (feature extraction)
        self.inc = DoubleConv(n_channels, 64)     # Input block
        self.down1 = Down(64, 128)                # Downsampling 1
        self.down2 = Down(128, 256)               # Downsampling 2
        self.down3 = Down(256, 512)               # Downsampling 3

        # If bilinear upsampling is used, reduce the number of channels in the bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)    # Bottleneck layer

        # Decoder path (upsampling + skip connections)
        self.up1 = Up(1024, 512 // factor, bilinear)   # Upsampling 1
        self.up2 = Up(512, 256 // factor, bilinear)    # Upsampling 2
        self.up3 = Up(256, 128 // factor, bilinear)    # Upsampling 3
        self.up4 = Up(128, 64, bilinear)               # Upsampling 4

        # Output layer (1×1 convolution to map feature maps to class logits)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward propagation:
        1. Encode: progressively downsample and extract hierarchical features.
        2. Decode: upsample while merging corresponding encoder features (skip connections).
        3. Output: predict per-pixel class scores.
        """
        # Encoder path
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 1024, H/16, W/16]

        # Decoder path with skip connections
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64,  H,  W]

        # Final segmentation map (class logits for each pixel)
        logits = self.outc(x)
        return logits
