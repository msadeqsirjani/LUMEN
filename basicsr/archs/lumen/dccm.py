"""
DCCM: Double Convolutional Channel Mixer (plain, non-re-parameterizable).

Used as the ablation baseline for RepDCCM. Two 3x3 convolutions with
configurable activation (GELU or StarReLU), no BatchNorm branches.

Same architecture as PLKSR's channel mixer.
"""

import torch
import torch.nn as nn

from .star_relu import StarReLU


def _build_activation(name: str) -> nn.Module:
    """Build activation by name."""
    if name == 'star_relu':
        return StarReLU()
    elif name == 'gelu':
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class DCCM(nn.Module):
    """Plain Double Convolutional Channel Mixer.

    Args:
        dim: Number of input/output channels.
        activation: Activation function name ('star_relu' or 'gelu').
    """

    def __init__(self, dim: int, activation: str = 'gelu'):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act = _build_activation(activation)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))
