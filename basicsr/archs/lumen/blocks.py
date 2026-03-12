"""
LUMEN core building blocks.

GPU-efficient design with minimal kernel launches per block:
  - DCCM: two 3x3 convs for channel mixing (same as PLKSR)
  - PMSDM: partial multi-scale depthwise with decomposed large kernels
    - Local branch: 5x5 depthwise (fine detail)
    - Global branch: Kx1 + 1xK separable depthwise (large receptive field)
    - Only on partial channels (rest = identity)
  - ~5 ops per block (matching PLKSR's efficiency)
  - GELU activation (well-optimized on GPU)
"""

import torch
import torch.nn as nn


class DCCM(nn.Module):
    """Double Convolutional Channel Mixer.

    Args:
        dim: Number of input/output channels.
        expand_ratio: Channel expansion ratio for the hidden layer.
    """

    def __init__(self, dim: int, expand_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * expand_ratio)
        self.conv1 = nn.Conv2d(dim, hidden, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))


class PMSDM(nn.Module):
    """Partial Multi-Scale Depthwise Mixer.

    Two-branch multi-scale spatial mixing on partial channels:
      - Local branch: 5x5 depthwise (captures fine textures)
      - Global branch: decomposed Kx1 + 1xK separable depthwise
        (captures structure/edges with large receptive field)

    The decomposed large kernel is the key differentiator from PLKSR:
    instead of a single KxK conv, we use separable factorization which
    gives the same effective receptive field with fewer params and
    naturally captures horizontal/vertical structures.

    Args:
        dim: Total number of channels.
        partial_ch: Number of channels to process (rest = identity).
        large_kernel: Size of the decomposed large kernel.
    """

    def __init__(self, dim: int, partial_ch: int = 16, large_kernel: int = 17):
        super().__init__()
        self.partial_ch = partial_ch
        pad = large_kernel // 2
        # Local: 5x5 depthwise for fine detail
        self.dw_local = nn.Conv2d(partial_ch, partial_ch, 5, 1, 2,
                                  groups=partial_ch, bias=False)
        # Global: decomposed Kx1 + 1xK for large receptive field
        self.dw_h = nn.Conv2d(partial_ch, partial_ch, (1, large_kernel), 1,
                              (0, pad), groups=partial_ch, bias=False)
        self.dw_v = nn.Conv2d(partial_ch, partial_ch, (large_kernel, 1), 1,
                              (pad, 0), groups=partial_ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_part = x[:, :self.partial_ch, :, :]
        x_id = x[:, self.partial_ch:, :, :]
        # Local detail + global structure (decomposed large kernel)
        out = self.dw_local(x_part) + self.dw_v(self.dw_h(x_part))
        return torch.cat([out, x_id], dim=1)


class EA(nn.Module):
    """Element-wise Attention.

    Instance-dependent modulation via sigmoid-gated attention.
    Only used in the full model, excluded in tiny variants.

    Args:
        dim: Number of channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.conv(x))


class LumenBlock(nn.Module):
    """LUMEN building block.

    Structure: x → DCCM → PMSDM → [EA] → Conv1x1 → + x

    Kernel launches per block (tiny, no EA):
      conv3x3 → gelu → conv3x3 → dw5x5 → dw1xK → dwKx1 → cat → conv1x1
      = 8 ops (vs PLKSR's ~7)

    Args:
        dim: Number of channels.
        partial_ch: Channels for multi-scale depthwise processing.
        large_kernel: Decomposed large kernel size.
        expand_ratio: DCCM channel expansion ratio.
        use_ea: Whether to include Element-wise Attention.
    """

    def __init__(self, dim: int, partial_ch: int = 16, large_kernel: int = 17,
                 expand_ratio: float = 2.0, use_ea: bool = False):
        super().__init__()
        self.dccm = DCCM(dim, expand_ratio)
        self.pmsdm = PMSDM(dim, partial_ch, large_kernel)
        self.ea = EA(dim) if use_ea else nn.Identity()
        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dccm(x)
        out = self.pmsdm(out)
        out = self.ea(out)
        return x + self.pw(out)
