"""
PFA: Parameter-Free Attention.

Inspired by SPAN (arXiv:2311.12770, NTIRE 2024 winner):
    Attention(x) = x * sigmoid(|x|)

Zero learnable parameters — attention is derived purely from feature
magnitudes. Features with larger absolute values are amplified while
near-zero features are suppressed.

This replaces the original EA module (Conv3x3 + Sigmoid) which added
dim*dim*9 parameters per block. PFA achieves comparable gating with
no extra parameters and no extra kernel launches (sigmoid + multiply
fuse trivially on GPU).
"""

import torch
import torch.nn as nn


class PFA(nn.Module):
    """Parameter-Free Attention.

    Computes: x * sigmoid(|x|)

    No learnable parameters. Always enabled (no param cost to justify
    toggling off like the original EA).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x.abs())
