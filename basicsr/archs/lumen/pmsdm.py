"""
PMSDM: Partial Multi-Scale Depthwise Mixer (original 2-scale version).

Two-branch spatial mixing on partial channels:
  - Local branch:  5x5 depthwise (fine textures)
  - Global branch: Kx1 + 1xK decomposed depthwise (large-scale structure)
  - Identity:      remaining channels pass through

Used as the ablation baseline for EPMSDM (3-scale version).
"""

import torch
import torch.nn as nn


class PMSDM(nn.Module):
    """Original Partial Multi-Scale Depthwise Mixer (2-scale).

    Splits channels into two groups:
        [0 : p]       → local 5x5 DW + global Kx1/1xK DW (summed)
        [p : dim]     → identity (pass-through)

    Args:
        dim: Total number of channels.
        partial_ch: Channels to process (rest = identity).
        large_kernel: Size of the decomposed large kernel. Default: 17.
    """

    def __init__(self, dim: int, partial_ch: int = 8, large_kernel: int = 17):
        super().__init__()
        self.partial_ch = partial_ch
        pad = large_kernel // 2

        # Local: 5x5 depthwise
        self.dw_local = nn.Conv2d(
            partial_ch, partial_ch, 5, 1, 2,
            groups=partial_ch, bias=False,
        )
        # Global: decomposed Kx1 + 1xK
        self.dw_h = nn.Conv2d(
            partial_ch, partial_ch, (1, large_kernel), 1, (0, pad),
            groups=partial_ch, bias=False,
        )
        self.dw_v = nn.Conv2d(
            partial_ch, partial_ch, (large_kernel, 1), 1, (pad, 0),
            groups=partial_ch, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_part = x[:, :self.partial_ch, :, :]
        x_id = x[:, self.partial_ch:, :, :]
        out = self.dw_local(x_part) + self.dw_v(self.dw_h(x_part))
        return torch.cat([out, x_id], dim=1)
