"""
RepDCCM: Re-parameterizable Double Convolutional Channel Mixer.

Training:  RepConv(3x3+1x1+identity) → StarReLU → RepConv(3x3+1x1+identity)
Inference: Conv3x3 → StarReLU → Conv3x3  (after fuse)

The re-param branches provide free expressiveness at training time
(better gradient flow, implicit regularisation from BN) with zero
inference cost — the dominant technique across NTIRE 2024-2025 winners.

Unlike the original DCCM which used expand_ratio=2.0 (doubling channels),
RepDCCM keeps channels constant (expand_ratio=1.0) because the multi-branch
training topology already provides sufficient representational capacity.
This cuts per-block channel-mixer params roughly in half.
"""

import torch
import torch.nn as nn

from .rep_conv import RepConv
from .star_relu import StarReLU


class RepDCCM(nn.Module):
    """Re-parameterizable Double Convolutional Channel Mixer.

    Args:
        dim: Number of input/output channels.
        deploy: If True, initialize with fused convolutions. Default: False.
    """

    def __init__(self, dim: int, deploy: bool = False):
        super().__init__()
        self.rep1 = RepConv(dim, dim, deploy=deploy)
        self.act = StarReLU()
        self.rep2 = RepConv(dim, dim, deploy=deploy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rep2(self.act(self.rep1(x)))

    def fuse(self):
        """Collapse re-param branches for inference."""
        self.rep1.fuse()
        self.rep2.fuse()
