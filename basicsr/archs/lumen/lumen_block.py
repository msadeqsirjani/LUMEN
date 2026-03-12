"""
LumenBlock: core building block for the LUMEN network.

Default: RepDCCM → EPMSDM → PFA → Conv1x1 + residual

Supports ablation via constructor flags:
  - use_reparam: RepDCCM (True) vs plain DCCM (False)
  - spatial_mixer: 'epmsdm' (3-scale) vs 'pmsdm' (2-scale)
  - attention: 'pfa' (parameter-free) vs 'none'
  - activation: 'star_relu' vs 'gelu'
"""

import torch
import torch.nn as nn

from .rep_dccm import RepDCCM
from .dccm import DCCM
from .epmsdm import EPMSDM
from .pmsdm import PMSDM
from .pfa import PFA


class LumenBlock(nn.Module):
    """LUMEN building block with configurable components for ablation.

    Args:
        dim: Number of channels.
        partial_ch: Channels per active branch in spatial mixer.
        large_kernel: Decomposed large kernel size.
        use_reparam: Use RepDCCM (True) or plain DCCM (False).
        spatial_mixer: 'epmsdm' (3-scale) or 'pmsdm' (2-scale original).
        attention: 'pfa' (parameter-free) or 'none'.
        activation: 'star_relu' or 'gelu' (only affects plain DCCM).
        deploy: If True, use fused convolutions. Default: False.
    """

    def __init__(
        self,
        dim: int,
        partial_ch: int = 8,
        large_kernel: int = 17,
        use_reparam: bool = True,
        spatial_mixer: str = 'epmsdm',
        attention: str = 'pfa',
        activation: str = 'star_relu',
        deploy: bool = False,
    ):
        super().__init__()
        self.use_reparam = use_reparam

        # Channel mixer
        if use_reparam:
            self.dccm = RepDCCM(dim, deploy=deploy)
        else:
            self.dccm = DCCM(dim, activation=activation)

        # Spatial mixer
        if spatial_mixer == 'epmsdm':
            self.spatial = EPMSDM(dim, partial_ch, large_kernel)
        elif spatial_mixer == 'pmsdm':
            self.spatial = PMSDM(dim, partial_ch, large_kernel)
        else:
            raise ValueError(f"Unknown spatial_mixer: {spatial_mixer}")

        # Attention
        if attention == 'pfa':
            self.attn = PFA()
        elif attention == 'none':
            self.attn = nn.Identity()
        else:
            raise ValueError(f"Unknown attention: {attention}")

        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dccm(x)
        out = self.spatial(out)
        out = self.attn(out)
        return x + self.pw(out)

    def fuse(self):
        """Collapse re-param branches for inference."""
        if self.use_reparam:
            self.dccm.fuse()
