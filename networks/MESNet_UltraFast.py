"""
MESNet-fast: an accelerated implementation of the original MESNet idea.

Main goal:
- Preserve the core MESNet idea:
  Stem -> stacked MESBlock -> SegmentationHead
  MESBlock = MAU + EAU + SAU + fuse + residual
- Keep the same modeling principle as networks/MESNet.py: combine
  multiscale-aware features, edge-aware features, and saliency-aware features
  for fine-resolution UTC segmentation.
- Optimize slow implementation details:
  1) remove torchvision gaussian_blur from forward
  2) register Sobel / Gaussian kernels as buffers
  3) avoid repeated .to(device) calls in forward
  4) use bias=False before normalization layers
  5) use inplace activations where safe
  6) fix SegmentationHead width hard-coding

Implementation changes relative to networks/MESNet.py:
1. Training speed:
   - Uses a ResNet34 stem to downsample early, so most MES blocks operate on
     H/4 feature maps instead of full-resolution tensors.
   - Adds a lightweight decoder to restore H/4 predictions back to the input
     resolution.
   - Replaces the slow Gaussian-blur + Sobel edge path with a channel-mean
     Laplacian edge unit.
   - Provides fast_texture=True, which replaces log-entropy texture extraction
     with a cheaper local absolute-contrast descriptor.
   - Supports channels_last memory format and optional torch.compile.

2. Memory and stability:
   - Supports gradient checkpointing during training to reduce activation
     memory.
   - Uses GroupNorm in the decoder, which is more stable for small batches.
   - Uses bias=False before normalization layers and in-place activations where
     safe.

3. Engineering usability:
   - Adds from_config() so train.py and test.py can build the network directly
     from YAML/CLI configuration.
   - Keeps the public class name MESNet so existing training code can import
     from networks.MESNet_UltraFast without changing downstream calls.

The core idea remains the same as the original MESNet: MAU captures multiscale
context, EAU injects edge cues, and SAU emphasizes salient canopy structures
inside repeated MES blocks. The changes here are mainly implementation-level
and runtime-level modifications for faster training, lower memory pressure, and
cleaner configuration in the release pipeline. Small numerical differences from
networks/MESNet.py are expected because the stem, edge unit, and optional
texture descriptor are not bitwise-identical to the legacy implementation.

Author: Shuang Zhang
"""

import importlib.util

import torch
import torch.nn.functional as F
from torch.nn import init
from types import SimpleNamespace
import torchvision.models as models
import torch.nn as nn

import torch.nn as nn


def GN(channels, max_groups=8):
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)



class ResNet34Stem(nn.Module):
    """
    Replace original startconv with ResNet34 stem.
    Output: H/4 feature map
    """

    def __init__(self, out_channels=64, pretrained=True):
        super().__init__()

        resnet = models.resnet34(pretrained=pretrained)

        # conv1 + bn1 + relu + maxpool => /4
        self.stem = nn.Sequential(
            resnet.conv1,   # /2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # /4
        )

        # project 64 -> width
        self.proj = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.proj(x)
        return x

class SimpleDecoder(nn.Module):
    """
    Restore H/4 features to the input resolution with two 2x upsampling steps.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            GN(in_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1, bias=False),
            GN(in_channels // 4),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )

        self.head = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return self.head(x)

# =========================================================
# 1. Spatial Attention
# =========================================================
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Computes spatial attention using channel-wise average pooling
    and max pooling, followed by a convolution layer.
    """

    def __init__(self, kernel_size=7, only_map=False):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.only_map = only_map

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)

        spatial_feature = torch.cat((avg_out, max_out), dim=1)
        attention = torch.sigmoid(self.conv(spatial_feature))

        if self.only_map:
            return attention
        return x * attention


# =========================================================
# 2. Local Entropy / Texture Descriptor
# =========================================================
def local_entropy(x, kernel_size=3, eps=1e-6):
    """
    Approximate local entropy for texture representation.

    Kept mathematically consistent with the original version:
        local_hist = avg_pool(x)
        entropy = -p * log(p)

    Note:
    - This is still relatively expensive because of log().
    - If you want maximum speed, use SAU(..., fast_texture=True).
    """
    local_hist = F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    local_hist = torch.clamp(local_hist, min=eps, max=1.0)
    entropy = -local_hist * torch.log(local_hist + eps)
    return entropy


# =========================================================
# 3. Saliency Attention Unit
# =========================================================
class SAU(nn.Module):
    """
    Saliency Attention Unit.

    Combines:
    - texture saliency
    - feature saliency

    fast_texture=False:
        uses the original entropy-like texture descriptor.
    fast_texture=True:
        uses local absolute contrast instead of log entropy. This is faster,
        but slightly changes the texture descriptor.
    """

    def __init__(
        self,
        in_channels,
        kernel_size=3,
        eps=1e-6,
        fast_texture=False,
    ):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.kernel_size = kernel_size
        self.eps = eps
        self.fast_texture = fast_texture

        self.spatial_att = SpatialAttention(only_map=True)
        self.channel_reduce = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            GN(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_norm = self.norm(x)

        if self.fast_texture:
            # Faster texture approximation:
            # local absolute contrast = |x - local_mean(x)|
            local_mean = F.avg_pool2d(
                x_norm,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
            texture = torch.abs(x_norm - local_mean)
        else:
            texture = local_entropy(
                x_norm,
                kernel_size=self.kernel_size,
                eps=self.eps,
            )

        texture = self.channel_reduce(texture)
        feature = self.spatial_att(x)

        saliency = self.fusion(torch.cat((feature, texture), dim=1))
        return saliency


# =========================================================
# 4. Multi-scale Aggregation Unit
# =========================================================
class MAU(nn.Module):
    """
    Multiscale-Aware Unit.

    Same topology as the original MAU:
    - sequential depthwise branches: 1x1 -> 3x3 -> 5x5
    - parallel depthwise branches: 3x3 and 5x5
    """

    def __init__(self, channels, bn_momentum=0.1, negative_slope=0.01):
        super().__init__()

        def dw_block(k, pad):
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=k,
                    padding=pad,
                    groups=channels,
                    bias=False,
                ),
                GN(channels),
                nn.LeakyReLU(negative_slope, inplace=True),
            )

        self.t1 = dw_block(1, 0)
        self.t2 = dw_block(3, 1)
        self.t3 = dw_block(5, 2)

        self.p1 = dw_block(3, 1)
        self.p2 = dw_block(5, 2)

    def forward(self, x):
        f1 = self.t1(x)
        f2 = self.t2(f1) + f1
        f3 = self.t3(f2) + f2

        out = x + f1 + f2 + f3
        out = out + self.p1(x) + self.p2(x)
        return out


# =========================================================
# 5. Edge Attention Unit
# =========================================================
class EAU(nn.Module):
    """
    Edge-Aware Unit (Channel Mean + Laplacian Residual)

    Core idea:
        1. Average channels into x_gray
        2. Apply learnable Laplacian
        3. Residual high-frequency extraction
    """

    def __init__(self, in_channels, use_learnable_scale=True):
        super().__init__()

        self.in_channels = in_channels

        # -----------------------------------------------------
        # Depthwise Laplacian (applied AFTER channel mean)
        # -----------------------------------------------------
        self.laplace_conv = nn.Conv2d(
            1, 1,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self._init_laplacian()

        # -----------------------------------------------------
        # Scale (stabilization)
        # -----------------------------------------------------
        self.use_learnable_scale = use_learnable_scale
        if use_learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scale", torch.tensor(1.0))

        # -----------------------------------------------------
        # Fusion (unchanged, but input is single-channel now)
        # -----------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def _init_laplacian(self):
        """
        Initialize Laplacian kernel (fixed 3x3 operator)
        """
        with torch.no_grad():
            kernel = torch.tensor(
                [[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)

            self.laplace_conv.weight.copy_(kernel)

    def forward(self, x):
        """
        Steps:
            1. Channel mean
            2. Laplacian
            3. Residual high-frequency extraction
            4. Attention map
        """

        # -----------------------------------------------------
        # 1. Channel average (key change)
        # -----------------------------------------------------
        x_gray = x.mean(dim=1, keepdim=True)

        # -----------------------------------------------------
        # 2. Laplacian
        # -----------------------------------------------------
        lap = self.laplace_conv(x_gray)

        # -----------------------------------------------------
        # 3. High-frequency residual
        # -----------------------------------------------------
        edge_feat = x_gray - self.scale * lap

        # optional stability nonlinearity
        edge_feat = torch.abs(edge_feat)

        # -----------------------------------------------------
        # 4. Attention map
        # -----------------------------------------------------
        edge = self.fusion(edge_feat)

        return edge


# =========================================================
# 6. MES Block
# =========================================================
class MESBlock(nn.Module):
    """
    MES Block.

    Same overall logic as the original:
        f = MAU(x)
        edge = EAU(x)
        sal = SAU(x)
        optional edge refinement using previous edge
        fuse([f, edge, sal]) + x
    """

    def __init__(
        self,
        channels,
        has_last_edge=True,
        fast_texture=False,
    ):
        super().__init__()

        self.mau = MAU(channels)
        self.eau = EAU(channels)
        self.sau = SAU(channels, fast_texture=fast_texture)

        self.has_last_edge = has_last_edge

        if has_last_edge:
            self.edge_refine = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
                GN(1),
                nn.LeakyReLU(0.01, inplace=True),
            )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels + 2, channels, kernel_size=1, bias=False),
            GN(channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x, prev_edge=None):
        f = self.mau(x)
        edge = self.eau(x)
        sal = self.sau(x)

        if self.has_last_edge and prev_edge is not None:
            f = f - edge
            edge = self.edge_refine(torch.cat((edge, prev_edge), dim=1))
            f = f + edge

        f = F.group_norm(f, num_groups=8)
        edge = F.group_norm(edge, num_groups=1)
        sal = F.group_norm(sal, num_groups=1)

        out = self.fuse(torch.cat((f, edge, sal), dim=1)) + x
        return out, edge


# =========================================================
# 7. MESNet
# =========================================================
class MESNet(nn.Module):
    """
    MESNet-fast (clean version)

    Config-driven architecture:
    - model_cfg: structure
    - optim_cfg: training optim hints
    - runtime_cfg: execution optim (compile / checkpoint / memory format)
    """

    def __init__(
        self,
        width=64,
        image_band=3,
        num_blocks=5,
        need_seg=True,
        fast_texture=False,
        optim_cfg=None,
        runtime_cfg=None,
    ):
        super().__init__()

        # =====================================================
        # Core config
        # =====================================================
        self.need_seg = need_seg
        self.width = width

        self.stem = ResNet34Stem(out_channels=width, pretrained=True)

        self.blocks = nn.ModuleList([
            MESBlock(
                width,
                has_last_edge=(i != 0),
                fast_texture=fast_texture,
            )
            for i in range(num_blocks)
        ])

        self.decoder = SimpleDecoder(width, width)

        self.head = SegmentationHead(width, 2) if need_seg else None

        self._init_weights()

        # =====================================================
        # Optim config (training-related)
        # =====================================================
        self.optim_cfg = optim_cfg or {}

        # =====================================================
        # Runtime config (execution-related)
        # =====================================================
        self.runtime_cfg = runtime_cfg or {
            "checkpoint": True,
            "channels_last": True,
            "compile": False,
            "compile_mode": "default",
        }

        self.use_checkpoint = self.runtime_cfg.get("checkpoint", True)
        self.use_channels_last = self.runtime_cfg.get("channels_last", True)
        self.enable_compile = self.runtime_cfg.get("compile", False)
        self.compile_mode = self.runtime_cfg.get("compile_mode", "default")

    # =========================================================
    # Config factory
    # =========================================================
    @classmethod
    def from_config(cls, args):
        def namespace_to_dict(value):
            if isinstance(value, SimpleNamespace):
                return {
                    key: namespace_to_dict(item)
                    for key, item in vars(value).items()
                }

            if isinstance(value, dict):
                return {
                    key: namespace_to_dict(item)
                    for key, item in value.items()
                }

            return value

        # -------------------------
        # Model config
        # -------------------------
        model_cfg = getattr(args, "model", None)

        width = 64
        image_band = 3
        num_blocks = 5
        need_seg = True
        fast_texture = False


        if model_cfg is not None:
            width = getattr(model_cfg, "width", width)
            image_band = getattr(model_cfg, "image_band", image_band)
            num_blocks = getattr(model_cfg, "num_blocks", num_blocks)
            need_seg = getattr(model_cfg, "need_seg", need_seg)
            fast_texture = getattr(model_cfg, "fast_texture", fast_texture)

        # -------------------------
        # Optim config
        # -------------------------
        optim_cfg = getattr(args, "optim", None)
        optim_cfg = namespace_to_dict(optim_cfg) if optim_cfg is not None else {}

        # -------------------------
        # Runtime config (clean merge)
        # -------------------------
        runtime_cfg = getattr(args, "runtime", None)
        runtime_cfg = namespace_to_dict(runtime_cfg) if runtime_cfg is not None else {}

        runtime_defaults = {
            "checkpoint": True,
            "channels_last": True,
            "compile": False,
            "compile_mode": "default",
        }

        runtime_cfg = {**runtime_defaults, **runtime_cfg}

        # -------------------------
        # Build model
        # -------------------------
        return cls(
            width=width,
            image_band=image_band,
            num_blocks=num_blocks,
            need_seg=need_seg,
            fast_texture=fast_texture,
            optim_cfg=optim_cfg,
            runtime_cfg=runtime_cfg,
        )

    # =========================================================
    # Init weights
    # =========================================================
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if getattr(m, "weight", None) is not None:
                    init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    init.zeros_(m.bias)

    # =========================================================
    # Forward
    # =========================================================
    def forward(self, x):

        # channels_last (safe inside forward)
        if self.use_channels_last:
            x = x.to(memory_format=torch.channels_last)

        x = self.stem(x)

        edge = None
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x, edge = checkpoint(blk, x, edge)
            else:
                x, edge = blk(x, edge)

        x=self.decoder(x)

        return self.head(x) if self.need_seg else x

    # =========================================================
    # Runtime build (compile only)
    # =========================================================
    def build_runtime(self):
        """
        Apply execution-time optimizations.
        Call AFTER model.to(device)
        """

        if self.enable_compile:
            if importlib.util.find_spec("triton") is None:
                print(
                    "WARNING: runtime.compile=True but Triton is unavailable. "
                    "Use eager mode instead."
                )
                return self

            try:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                return torch.compile(self, mode=self.compile_mode)
            except Exception as exc:
                print(
                    "WARNING: torch.compile failed during setup. "
                    f"Use eager mode instead. Reason: {exc}"
                )
                return self

        return self


# =========================================================
# 8. Utilities
# =========================================================
class SegmentationHead(nn.Sequential):
    """
    Simple segmentation head.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )


def freeze_layers(model, keywords):
    """
    Freeze layers containing specific keywords.
    """
    for name, param in model.named_parameters():
        if any(k in name for k in keywords):
            param.requires_grad = False
            print(f"[Frozen] {name}")



if __name__ == "__main__":
    import argparse
    import time

    from MESNet import MESNet as LegacyMESNet

    def benchmark_model(model, x, warmup_iters, test_iters, use_amp):
        model.eval()

        with torch.no_grad():
            _ = model(x)

        if x.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        with torch.no_grad():
            for _ in range(warmup_iters):
                if x.device.type == "cuda" and use_amp:
                    with torch.amp.autocast("cuda"):
                        _ = model(x)
                else:
                    _ = model(x)

        if x.device.type == "cuda":
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()

            with torch.no_grad():
                for _ in range(test_iters):
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            _ = model(x)
                    else:
                        _ = model(x)

            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender)
        else:
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(test_iters):
                    _ = model(x)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

        latency_ms = elapsed_ms / max(test_iters, 1)
        fps = x.shape[0] * 1000.0 / latency_ms

        return {
            "fps": fps,
        }

    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy MESNet.py with the accelerated "
            "MESNet_UltraFast.py implementation."
        )
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_band", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--test_iters", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--fast_texture", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    x = torch.randn(
        args.batch_size,
        args.image_band,
        args.image_size,
        args.image_size,
        device=device,
    )

    models_to_test = [
        (
            "MESNet.py legacy",
            LegacyMESNet(
                width=args.width,
                image_band=args.image_band,
                num_blocks=args.num_blocks,
                need_seg=True,
            ),
        ),
        (
            "MESNet_UltraFast.py",
            MESNet(
                width=args.width,
                image_band=args.image_band,
                num_blocks=args.num_blocks,
                need_seg=True,
                fast_texture=args.fast_texture,
                runtime_cfg={
                    "checkpoint": False,
                    "channels_last": device.type == "cuda",
                    "compile": False,
                    "compile_mode": "default",
                },
            ),
        ),
    ]

    print("=" * 88)
    print("MESNet efficiency comparison")
    print(f"Device       : {device}")
    print(f"Input shape  : {tuple(x.shape)}")
    print(f"AMP enabled  : {args.amp and device.type == 'cuda'}")
    print(f"Fast texture : {args.fast_texture}")
    print("=" * 88)

    results = []
    for name, model in models_to_test:
        model = model.to(device)
        if name == "MESNet_UltraFast.py" and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        result = benchmark_model(
            model=model,
            x=x,
            warmup_iters=args.warmup_iters,
            test_iters=args.test_iters,
            use_amp=args.amp and device.type == "cuda",
        )
        result["name"] = name
        results.append(result)

    header = (
        f"{'Model':<24}"
        f"{'FPS':>12}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['name']:<24}"
            f"{result['fps']:>12.2f}"
        )

    print("=" * 88)
