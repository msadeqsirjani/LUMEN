"""
EPMSDM: Enhanced Partial Multi-Scale Depthwise Mixer.

Three-scale spatial mixing on partial channels:
  - Fine branch:   3x3 depthwise (texture, noise)
  - Medium branch: 5x5 depthwise (edges, small structures)
  - Global branch: Kx1 + 1xK separable depthwise (large-scale structure)
  - Identity:      remaining channels pass through unchanged

Compared to the original PMSDM (2-scale: 5x5 + decomposed K):
  - Adds a fine 3x3 branch for better texture capture
  - Each branch operates on its own channel slice (no redundant compute)
  - Three-scale design inspired by MAN's multi-scale large kernel attention
"""

import torch
import torch.nn as nn


class EPMSDM(nn.Module):
    """Enhanced Partial Multi-Scale Depthwise Mixer.

    Splits channels into four groups:
        [0 : p]           → fine 3x3 depthwise
        [p : 2p]          → medium 5x5 depthwise
        [2p : 3p]         → global Kx1 + 1xK decomposed depthwise
        [3p : dim]        → identity (pass-through)

    Args:
        dim: Total number of channels.
        partial_ch: Channels per active branch (3 branches total).
            Must satisfy 3 * partial_ch <= dim.
        large_kernel: Size of the decomposed large kernel. Default: 17.
    """

    def __init__(self, dim: int, partial_ch: int = 8, large_kernel: int = 17):
        super().__init__()
        assert 3 * partial_ch <= dim, (
            f"3 * partial_ch ({3 * partial_ch}) must be <= dim ({dim})"
        )
        self.p = partial_ch
        pad_lg = large_kernel // 2

        # Fine: 3x3 depthwise
        self.dw_fine = nn.Conv2d(
            partial_ch, partial_ch, 3, 1, 1,
            groups=partial_ch, bias=False,
        )
        # Medium: 5x5 depthwise
        self.dw_medium = nn.Conv2d(
            partial_ch, partial_ch, 5, 1, 2,
            groups=partial_ch, bias=False,
        )
        # Global: decomposed Kx1 + 1xK separable depthwise
        self.dw_h = nn.Conv2d(
            partial_ch, partial_ch, (1, large_kernel), 1, (0, pad_lg),
            groups=partial_ch, bias=False,
        )
        self.dw_v = nn.Conv2d(
            partial_ch, partial_ch, (large_kernel, 1), 1, (pad_lg, 0),
            groups=partial_ch, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        x_fine = x[:, :p, :, :]
        x_med = x[:, p:2*p, :, :]
        x_glob = x[:, 2*p:3*p, :, :]
        x_id = x[:, 3*p:, :, :]

        y_fine = self.dw_fine(x_fine)
        y_med = self.dw_medium(x_med)
        y_glob = self.dw_v(self.dw_h(x_glob))

        return torch.cat([y_fine, y_med, y_glob, x_id], dim=1)
