"""
MESNet: Multi-scale Edge-aware Saliency Network

This file implements the core architecture of MESNet for fine-grained
urban tree canopy (UTC) segmentation.

The network follows a resolution-preserving design and consists of
stacked MES blocks, each integrating:
    - Multiscale-aware features (MAU)
    - Edge-aware features (EAU)
    - Saliency-aware features (SAU)

Author: Shuang Zhang
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.transforms.functional import gaussian_blur


# =========================================================
# 1. Spatial Attention
# =========================================================
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module

    Computes spatial attention using channel-wise average pooling
    and max pooling, followed by a convolution layer.
    """

    def __init__(self, kernel_size=7, only_map=False):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
        self.only_map = only_map

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate
        spatial_feature = torch.cat([avg_out, max_out], dim=1)

        # Generate attention map
        attention = self.sigmoid(self.conv(spatial_feature))

        if self.only_map:
            return attention
        return x * attention


# =========================================================
# 2. Local Entropy (Texture Descriptor)
# =========================================================
def local_entropy(x, kernel_size=3, eps=1e-6):
    """
    Approximate local entropy for texture representation.

    Used in SAU to describe local texture complexity of canopy regions.
    """
    local_hist = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)
    local_hist = torch.clamp(local_hist, min=eps, max=1.0)

    entropy = -local_hist * torch.log(local_hist + eps)
    return entropy


# =========================================================
# 3. Saliency Attention Unit (SAU)
# =========================================================
class SAU(nn.Module):
    """
    Saliency Attention Unit

    Combines:
    - Texture saliency (local entropy)
    - Feature saliency (spatial attention)
    """

    def __init__(self, in_channels, kernel_size=3, eps=1e-6):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.kernel_size = kernel_size
        self.eps = eps

        self.spatial_att = SpatialAttention(only_map=True)
        self.channel_reduce = nn.Conv2d(in_channels, 1, kernel_size=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.norm(x)

        # Texture saliency
        texture = local_entropy(x_norm, self.kernel_size, self.eps)
        texture = self.channel_reduce(texture)

        # Feature saliency
        feature = self.spatial_att(x)

        # Fuse
        saliency = self.fusion(torch.cat([feature, texture], dim=1))
        return saliency


# =========================================================
# 4. Multi-scale Aggregation Unit (MAU)
# =========================================================
class MAU(nn.Module):
    """
    Multiscale-Aware Unit (MAU)

    Captures multiscale contextual information via hierarchical
    receptive field aggregation.

    Combines:
        - Sequential feature propagation (context modeling)
        - Parallel feature extraction (multi-scale detail)

    Designed to handle scale diversity of UTCs.
    """

    def __init__(self, channels, bn_momentum=0.1, negative_slope=0.01):
        super().__init__()

        def dw_block(k, pad):
            return nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=pad, groups=channels),
                nn.BatchNorm2d(channels, momentum=bn_momentum),
                nn.LeakyReLU(negative_slope)
            )

        # Sequential blocks
        self.t1 = dw_block(1, 0)
        self.t2 = dw_block(3, 1)
        self.t3 = dw_block(5, 2)

        # Parallel blocks
        self.p1 = dw_block(3, 1)
        self.p2 = dw_block(5, 2)

    def forward(self, x):
        f1 = self.t1(x)
        f2 = self.t2(f1) + f1
        f3 = self.t3(f2) + f2

        out = x + f1 + f2 + f3
        out += self.p1(x) + self.p2(x)

        return out


# =========================================================
# 5. Edge Attention Unit (EAU)
# =========================================================
class EAU(nn.Module):
    """
    Edge-Aware Unit (EAU)

    Enhances boundary representation by explicitly modeling edge features.

    Combines:
        - Pre-difined gradient extraction (Sobel operator)
        - Learnable feature fusion

    Enables accurate delineation of complex canopy boundaries.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.sobel_x = self._sobel_kernel('x')
        self.sobel_y = self._sobel_kernel('y')

        self.sobel_x.requires_grad = False
        self.sobel_y.requires_grad = False

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * in_channels, 1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

    def _sobel_kernel(self, direction):
        if direction == 'x':
            k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        else:
            k = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        k = k.float().view(1, 1, 3, 3)
        return k.repeat(self.in_channels, 1, 1, 1)

    def forward(self, x):
        x = gaussian_blur(x, [3, 3])

        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        gx = F.conv2d(x, sobel_x, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, sobel_y, padding=1, groups=self.in_channels)

        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        edge = self.fusion(torch.cat([gx, gy, mag], dim=1))
        return edge


# =========================================================
# 6. MES Block
# =========================================================
class MESBlock(nn.Module):
    """
    MES Block

    The fundamental building unit of MESNet.

    Integrates three complementary cues:
        - Multiscale features (MAU)
        - Edge information (EAU)
        - Saliency cues (SAU)

    A refinement mechanism is applied to progressively enhance edge quality
    across layers.
    """

    def __init__(self, channels, has_last_edge=True):
        super().__init__()

        self.mau = MAU(channels)
        self.eau = EAU(channels)
        self.sau = SAU(channels)

        self.has_last_edge = has_last_edge

        if has_last_edge:
            self.edge_refine = nn.Sequential(
                nn.Conv2d(2, 1, 3, padding=1),
                nn.BatchNorm2d(1),
                nn.LeakyReLU(0.01)
            )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x, prev_edge=None):
        f = self.mau(x)
        edge = self.eau(x)
        sal = self.sau(x)

        if self.has_last_edge and prev_edge is not None:
            f = f - edge
            edge = self.edge_refine(torch.cat([edge, prev_edge], dim=1))
            f = f + edge

        out = self.fuse(torch.cat([f, edge, sal], dim=1)) + x
        return out, edge


# =========================================================
# 7. MESNet
# =========================================================
class MESNet(nn.Module):
    """
    MESNet

    A resolution-preserving segmentation network composed of stacked MES blocks.

    Key characteristics:
        - No spatial downsampling (preserves fine details)
        - Progressive feature refinement
        - Joint modeling of scale, boundary, and saliency

    Designed for fine-grained UTC mapping with coarse supervision.
    """

    def __init__(self, width=64, image_band=3, num_blocks=5, need_seg=True):
        super().__init__()

        self.need_seg = need_seg

        self.stem = nn.Conv2d(image_band, width, 3, padding=1)

        self.blocks = nn.ModuleList([
            MESBlock(width, has_last_edge=(i != 0))
            for i in range(num_blocks)
        ])

        self.head = SegmentationHead(64, 2) if need_seg else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.stem(x)

        edge = None
        for blk in self.blocks:
            x, edge = blk(x, edge)

        if self.need_seg:
            return self.head(x)
        return x


# =========================================================
# 8. Utilities
# =========================================================
class SegmentationHead(nn.Sequential):
    """
    Simple segmentation head.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )


def freeze_layers(model, keywords):
    """
    Freeze layers containing specific keywords.
    """
    for name, param in model.named_parameters():
        if any(k in name for k in keywords):
            param.requires_grad = False
            print(f"[Frozen] {name}")

if __name__ == '__main__':
    net = MESNet(64, image_band=3)
    # net.load_state_dict(torch.load(r"J:\UTCMapping\pretrained\SH_epoch_49.pth"))
    for name, param in net.named_parameters():
        print(f"{name}: {param.shape}")
