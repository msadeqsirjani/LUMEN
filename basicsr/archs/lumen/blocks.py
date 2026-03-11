"""
LUMEN core building blocks.

Design philosophy for STM32 Cortex-M deployment:
  - All ops are INT8-quantization friendly
  - BatchNorm folds into conv weights at inference (zero overhead)
  - ReLU6 maps directly to CMSIS-NN clipped ReLU
  - Hard-sigmoid avoids expensive exp/division
  - Depthwise convolutions leverage CMSIS-NN optimized kernels
  - No FFT, no softmax over spatial tokens, no dynamic ops
"""

import torch
import torch.nn as nn


class MultiScaleDepthwiseMixer(nn.Module):
    """Multi-Scale Depthwise Mixer (MSDM).

    Hardware-friendly replacement for window self-attention.
    Captures local context (3x3, 5x5) and semi-global context
    (strip convolutions 1x7, 7x1) using only depthwise operations.

    Complexity: O(C * H * W) — linear, vs O(w^2 * H * W) for window attention.
    MCU-friendly: all depthwise ops, no quadratic attention, no softmax.

    Args:
        dim: Number of input channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Local context at two scales
        self.dw3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        # Semi-global context via strip (horizontal + vertical)
        self.dw_h = nn.Conv2d(dim, dim, (1, 7), 1, (0, 3), groups=dim, bias=False)
        self.dw_v = nn.Conv2d(dim, dim, (7, 1), 1, (3, 0), groups=dim, bias=False)
        # Channel fusion
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.dw3(x) + self.dw5(x)
        strip = self.dw_h(x) + self.dw_v(x)
        return self.act(self.bn(self.pw(local + strip)))


class LightweightChannelRecalibrator(nn.Module):
    """Lightweight Channel Recalibrator (LCR).

    Hardware-friendly replacement for FTCA (FFT-based cross-attention).
    Uses global average pooling + small FC + hard-sigmoid for channel
    recalibration. This is SE-Net style but with:
      - Hard-sigmoid instead of sigmoid (no exp, just clamp+scale)
      - ReLU6 activation (CMSIS-NN native)
      - Minimal squeeze ratio to keep FC layers small

    Complexity: O(C^2 / squeeze) per sample — negligible for small C.
    MCU-friendly: pool, two FC, clamp — all CMSIS-NN supported ops.

    Args:
        dim: Number of input channels.
        squeeze_ratio: Squeeze factor for the hidden FC layer.
    """

    def __init__(self, dim: int, squeeze_ratio: int = 4):
        super().__init__()
        hidden = max(4, dim // squeeze_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Note: no BN on pooled (1x1) features — BN is undefined for spatial size 1.
        # The bias in fc1 provides the affine shift; fc2 bias handles the scale shift.
        self.fc1 = nn.Conv2d(dim, hidden, 1, bias=True)
        self.act = nn.ReLU6(inplace=True)
        self.fc2 = nn.Conv2d(hidden, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = self.act(self.fc1(s))
        s = self.fc2(s)
        # Hard-sigmoid: clamp(x/6 + 0.5, 0, 1)
        # Avoids expensive sigmoid, maps linearly — INT8 friendly
        s = torch.clamp(s * (1.0 / 6.0) + 0.5, 0.0, 1.0)
        return x * s


class LumenFFN(nn.Module):
    """Lumen Feed-Forward Network.

    Inverted bottleneck with depthwise conv and integrated channel
    recalibration. Replaces GDFN (gated depthwise FFN) from FTANet
    with a simpler, more MCU-friendly design:
      - No gating (avoids chunking and elementwise multiply)
      - ReLU6 instead of GELU
      - LCR for channel recalibration (replaces gating)
      - BatchNorm (folds at inference)

    Args:
        dim: Number of input channels.
        expand_ratio: Hidden dimension expansion factor.
    """

    def __init__(self, dim: int, expand_ratio: float = 2.5):
        super().__init__()
        hidden = int(dim * expand_ratio)

        self.pw_in = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False)
        self.bn_dw = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU6(inplace=True)
        self.lcr = LightweightChannelRecalibrator(hidden, squeeze_ratio=4)
        self.pw_out = nn.Conv2d(hidden, dim, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn_in(self.pw_in(x)))
        x = self.act(self.bn_dw(self.dw(x)))
        x = self.lcr(x)
        x = self.bn_out(self.pw_out(x))
        return x


class LumenBlock(nn.Module):
    """Core building block of LUMEN.

    Two sub-layers connected with learnable residual scales:
      1. MSDM: multi-scale spatial mixing (replaces window attention)
      2. LumenFFN: channel mixing with recalibration (replaces GDFN+FTCA)

    Learnable scales initialized to 1e-6 for stable training from scratch,
    following MambaIRv2/FTANet practice.

    Args:
        dim: Number of input channels.
        expand_ratio: FFN expansion ratio.
    """

    def __init__(self, dim: int, expand_ratio: float = 2.5):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(dim)
        self.msdm = MultiScaleDepthwiseMixer(dim)

        self.bn2 = nn.BatchNorm2d(dim)
        self.ffn = LumenFFN(dim, expand_ratio)

        # Learnable residual scales (tiny init for stable training)
        self.scale1 = nn.Parameter(torch.ones(dim, 1, 1) * 1e-6)
        self.scale2 = nn.Parameter(torch.ones(dim, 1, 1) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale1 * self.msdm(self.bn1(x))
        x = x + self.scale2 * self.ffn(self.bn2(x))
        return x


class LumenGroup(nn.Module):
    """Group of LumenBlocks with a group-level residual connection.

    Analogous to FTAG in FTANet: multiple blocks + a conv + residual.
    The group-level residual ensures stable gradient flow.

    Args:
        dim: Number of input channels.
        num_blocks: Number of LumenBlocks in this group.
        expand_ratio: FFN expansion ratio.
    """

    def __init__(self, dim: int, num_blocks: int, expand_ratio: float = 2.5):
        super().__init__()
        self.blocks = nn.ModuleList([
            LumenBlock(dim, expand_ratio) for _ in range(num_blocks)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for block in self.blocks:
            x = block(x)
        return self.conv(x) + residual
