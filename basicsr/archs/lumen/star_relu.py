"""
StarReLU activation function.

From MetaFormer Baselines (arXiv:2210.13452):
    StarReLU(x) = scale * ReLU(x)^2 + bias

Only 4 FLOPs per element vs GELU's 14 FLOPs (71% reduction).
Learnable scale and bias alleviate distribution shift from squaring.
"""

import torch
import torch.nn as nn


class StarReLU(nn.Module):
    """StarReLU: s * ReLU(x)^2 + b.

    Args:
        scale_value: Initial value for learnable scale. Default: 1.0.
        bias_value: Initial value for learnable bias. Default: 0.0.
        inplace: Whether to use inplace ReLU. Default: False.
    """

    def __init__(
        self,
        scale_value: float = 1.0,
        bias_value: float = 0.0,
        inplace: bool = False,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(torch.full((), scale_value))
        self.bias = nn.Parameter(torch.full((), bias_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.relu(x) ** 2 + self.bias
