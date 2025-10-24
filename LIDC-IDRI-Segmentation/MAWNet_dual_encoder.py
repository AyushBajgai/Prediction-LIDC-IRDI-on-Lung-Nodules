# Dual-encoder MAW-Net: Wavelet encoder + ResNet encoder + attention-gated fusion + UNet-style decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from pytorch_wavelets import DWTForward  # pip install pytorch-wavelets
except Exception as e:
    raise ImportError(
        "Missing dependency 'pytorch-wavelets'. Please install:\n"
        "  pip install pytorch-wavelets\n"
        f"Original error: {e}"
    )

from torchvision.models import resnet34


# ---------------------------
# Attention Blocks (CBAM)
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg + mx)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ---------------------------
# Basic Conv Blocks
# ---------------------------
def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Up-conv + concat skip + DoubleConv
    in_ch:  channels of the incoming tensor before upsample
    skip_ch: channels of the skip connection tensor
    out_ch: output channels of this decoder stage
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------
# Helpers
# ---------------------------
def cat_orients(yh_level: torch.Tensor | list | tuple) -> torch.Tensor:
    """
    Merge LH/HL/HH along channel dim.
    Accepts:
      - Tensor [B, C, 3, H, W]  (pytorch_wavelets default)
      - Iterable of 3 tensors [LH, HL, HH] each [B, C, H, W]
      - Tensor already merged [B, C', H, W] (returned as-is)
    """
    if isinstance(yh_level, (list, tuple)):
        parts = list(yh_level)
        assert len(parts) == 3, f"Expect 3 parts for orientations, got {len(parts)}"
    elif isinstance(yh_level, torch.Tensor):
        if yh_level.dim() == 5 and yh_level.size(2) == 3:
            parts = [yh_level[:, :, i, ...] for i in range(3)]
        elif yh_level.dim() == 4:
            return yh_level
        else:
            raise ValueError(f"Unexpected yh_level shape: {tuple(yh_level.shape)}")
    else:
        raise TypeError(f"Unsupported type for yh_level: {type(yh_level)}")
    return torch.cat(parts, dim=1).contiguous()


# ---------------------------
# Wavelet Encoder (per scale)
# ---------------------------
class WaveletEncoder(nn.Module):
    """
    Multi-level DWT encoder.
    For each level, concat highpass (LH, HL, HH) -> 1x1 reduce -> DoubleConv + CBAM.
    Returns 4-scale features aligned to UNet stages: /1, /2, /4, /8.
    """
    def __init__(self, in_ch=1, base_ch=64, levels=3, wave='haar'):
        super().__init__()
        self.levels = levels
        self.dwt = DWTForward(J=levels, wave=wave)

        # stem at full resolution
        self.stem = DoubleConv(in_ch, base_ch)

        # target channels per scale
        self.chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]  # 64,128,256,512

        # 1x1 reducers for high-frequency groups (3*base_ch -> target_ch)
        self.reduce2 = nn.Conv2d(base_ch * 3, self.chs[1], kernel_size=1, bias=False)
        self.reduce3 = nn.Conv2d(base_ch * 3, self.chs[2], kernel_size=1, bias=False)
        self.reduce4 = nn.Conv2d(base_ch * 3, self.chs[3], kernel_size=1, bias=False)

        # per-scale blocks + CBAM
        self.blocks = nn.ModuleList([DoubleConv(c, c) for c in self.chs])
        self.cbams = nn.ModuleList([CBAM(c) for c in self.chs])

    def forward(self, x):
        # full-res stem
        x0 = self.stem(x)                # [B, 64, H, W]
        LL, highs = self.dwt(x0)         # highs[0]/2, highs[1]/4, highs[2]/8

        # s1: /1
        s1 = self.cbams[0](self.blocks[0](x0))  # [B, 64, H, W]

        # s2: /2 from highs[0]
        h1 = cat_orients(highs[0])              # [B, 192, H/2, W/2]
        h1 = self.reduce2(h1)                   # [B, 128, H/2, W/2]
        s2 = self.cbams[1](self.blocks[1](h1))  # [B, 128, H/2, W/2]

        # s3: /4 from highs[1]
        h2 = cat_orients(highs[1])              # [B, 192, H/4, W/4]
        h2 = self.reduce3(h2)                   # [B, 256, H/4, W/4]
        s3 = self.cbams[2](self.blocks[2](h2))  # [B, 256, H/4, W/4]

        # s4: /8 from highs[2]
        h3 = cat_orients(highs[2])              # [B, 192, H/8, W/8]
        h3 = self.reduce4(h3)                   # [B, 512, H/8, W/8]
        s4 = self.cbams[3](self.blocks[3](h3))  # [B, 512, H/8, W/8]

        return [s1, s2, s3, s4]  # /1, /2, /4, /8


# ---------------------------
# ResNet Encoder (C1..C4)
# ---------------------------
class ResNetEncoder(nn.Module):
    """
    Outputs feature maps aligned with 4 UNet stages:
    After tweaks: /2, /4, /8, /8 (keep last at /8 by removing the stride).
    Then project channels to match wavelet branch: 64,128,256,512.
    """
    def __init__(self, in_ch=1, pretrained=False, freeze_bn=False):
        super().__init__()
        net = resnet34(weights=None if not pretrained else None)
        net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # /2
        self.pool = net.maxpool             # -> /4
        self.layer1 = net.layer1            # /4
        self.layer2 = net.layer2            # /8
        self.layer3 = net.layer3            # /16 normally

        # keep layer3 at /8
        for m in self.layer3.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (1, 1)

        # projection heads registered as modules
        self.proj1 = ConvBNReLU(64, 64)
        self.proj2 = ConvBNReLU(64, 128)
        self.proj3 = ConvBNReLU(128, 256)
        self.proj4 = ConvBNReLU(256, 512)

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)          # [B, 64, H/2, W/2]
        c1 = x
        x = self.pool(x)          # [B, 64, H/4, W/4]
        c2 = self.layer1(x)       # [B, 64, H/4, W/4]
        c3 = self.layer2(c2)      # [B, 128, H/8, W/8]
        c4 = self.layer3(c3)      # [B, 256, H/8, W/8]

        p1 = self.proj1(c1)       # 64
        p2 = self.proj2(c2)       # 128
        p3 = self.proj3(c3)       # 256
        p4 = self.proj4(c4)       # 512
        return [p1, p2, p3, p4]


# ---------------------------
# Fusion + Decoder
# ---------------------------
class FusionGate(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.align = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.cbam = CBAM(out_ch)

    def forward(self, w, e):
        # concat and align, then CBAM
        z = torch.cat([w, e], dim=1)
        z = self.align(z)
        z = self.cbam(z)
        return z


class MAWNetDualEncoder(nn.Module):
    """
    Wavelet encoder + ResNet encoder -> gated fusion per-scale -> UNet decoder -> 1x1 head
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=64, wave='haar'):
        super().__init__()
        self.wavelet_enc = WaveletEncoder(in_ch=in_channels, base_ch=base_ch, levels=3, wave=wave)
        self.resnet_enc = ResNetEncoder(in_ch=in_channels, pretrained=False)

        # per-scale fusion (C1..C4)
        self.fuse1 = FusionGate(64 + 64, 64)
        self.fuse2 = FusionGate(128 + 128, 128)
        self.fuse3 = FusionGate(256 + 256, 256)
        self.fuse4 = FusionGate(512 + 512, 512)

        # decoder:
        # fused4: 512 (/8), fused3: 256 (/4), fused2: 128 (/2), fused1: 64 (/1)
        self.up3 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)  # 512 -> 256; concat with 256
        self.up2 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)  # 256 -> 128; concat with 128
        self.up1 = UpBlock(in_ch=128, skip_ch=64,  out_ch=64)   # 128 -> 64;  concat with 64

        self.tail = DoubleConv(64, 64)
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    @staticmethod
    def _align(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Align spatial size of 'a' to 'ref' using bilinear interpolation."""
        if a.shape[-2:] != ref.shape[-2:]:
            a = F.interpolate(a, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return a

    def forward(self, x):
        # wavelet: /1, /2, /4, /8
        w1, w2, w3, w4 = self.wavelet_enc(x)
        # resnet:  /2, /4, /8, /8  (after stride tweaks)
        e1, e2, e3, e4 = self.resnet_enc(x)

        # align resnet maps to wavelet scales
        e1 = self._align(e1, w1)  # /2 -> /1
        e2 = self._align(e2, w2)  # /4 -> /2
        e3 = self._align(e3, w3)  # /8 -> /4
        e4 = self._align(e4, w4)  # /8 -> /8 (usually already aligned)

        # gated fusion at each scale
        fused1 = self.fuse1(w1, e1)  # [B, 64,  H,   W]
        fused2 = self.fuse2(w2, e2)  # [B, 128, H/2, W/2]
        fused3 = self.fuse3(w3, e3)  # [B, 256, H/4, W/4]
        fused4 = self.fuse4(w4, e4)  # [B, 512, H/8, W/8]

        # decoder
        d3 = self.up3(fused4, fused3)  # -> [B, 256, H/4, W/4]
        d2 = self.up2(d3, fused2)      # -> [B, 128, H/2, W/2]
        d1 = self.up1(d2, fused1)      # -> [B, 64,  H,   W]

        y = self.tail(d1)
        y = self.head(y)
        return y


# simple factory
def create_model(in_channels=1, out_channels=1):
    return MAWNetDualEncoder(in_channels=in_channels, out_channels=out_channels)



