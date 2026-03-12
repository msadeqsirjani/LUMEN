"""
Re-parameterizable Convolution.

Training: parallel 3x3+BN, 1x1+BN, identity+BN branches summed together.
Inference: algebraically merged into a single 3x3 convolution.

Based on RepVGG (arXiv:2101.03697) and ECBSR (ACM MM 2021).
This is the single most impactful technique for GPU-efficient SR — every
NTIRE 2024-2025 winner uses re-parameterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepConv(nn.Module):
    """Re-parameterizable 3x3 convolution.

    During training, uses three parallel branches:
        - Conv3x3 + BN
        - Conv1x1 + BN
        - BN (identity, only when in_ch == out_ch)

    At inference (after calling fuse()), collapses to a single Conv3x3.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolution stride. Default: 1.
        groups: Number of groups for grouped convolution. Default: 1.
        deploy: If True, initialize as fused single conv. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        deploy: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy

        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, 3, stride, 1,
                groups=groups, bias=True,
            )
        else:
            self.fused_conv = None

            # Branch 1: 3x3 conv + BN
            self.conv3x3 = nn.Conv2d(
                in_channels, out_channels, 3, stride, 1,
                groups=groups, bias=False,
            )
            self.bn3x3 = nn.BatchNorm2d(out_channels)

            # Branch 2: 1x1 conv + BN
            self.conv1x1 = nn.Conv2d(
                in_channels, out_channels, 1, stride, 0,
                groups=groups, bias=False,
            )
            self.bn1x1 = nn.BatchNorm2d(out_channels)

            # Branch 3: identity + BN (only when dimensions match)
            if in_channels == out_channels and stride == 1:
                self.bn_identity = nn.BatchNorm2d(out_channels)
            else:
                self.bn_identity = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_conv is not None:
            return self.fused_conv(x)

        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.bn_identity is not None:
            out = out + self.bn_identity(x)
        return out

    def _fuse_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """Fuse Conv + BN into equivalent Conv weights and bias."""
        kernel = conv.weight
        gamma = bn.weight
        beta = bn.bias
        mu = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        # Scale each output channel's kernel by gamma/std
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        fused_weight = kernel * scale
        fused_bias = beta - mu * gamma / std
        return fused_weight, fused_bias

    def _get_identity_kernel_bias(self):
        """Create a 3x3 identity kernel from BN parameters."""
        # Identity as 3x3: zeros with 1 in center
        channels_per_group = self.in_channels // self.groups
        kernel = torch.zeros(
            self.in_channels, channels_per_group, 3, 3,
            device=self.bn_identity.weight.device,
        )
        for i in range(self.in_channels):
            kernel[i, i % channels_per_group, 1, 1] = 1.0

        gamma = self.bn_identity.weight
        beta = self.bn_identity.bias
        mu = self.bn_identity.running_mean
        var = self.bn_identity.running_var
        eps = self.bn_identity.eps

        std = torch.sqrt(var + eps)
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * scale, beta - mu * gamma / std

    def _pad_1x1_to_3x3(self, kernel):
        """Pad a 1x1 kernel to 3x3."""
        return F.pad(kernel, [1, 1, 1, 1])

    def fuse(self):
        """Collapse multi-branch into single Conv3x3. Call before inference."""
        if self.fused_conv is not None:
            return  # Already fused

        # Branch 1: 3x3
        k3, b3 = self._fuse_bn(self.conv3x3, self.bn3x3)

        # Branch 2: 1x1 padded to 3x3
        k1, b1 = self._fuse_bn(self.conv1x1, self.bn1x1)
        k1 = self._pad_1x1_to_3x3(k1)

        # Branch 3: identity
        if self.bn_identity is not None:
            ki, bi = self._get_identity_kernel_bias()
            k_final = k3 + k1 + ki
            b_final = b3 + b1 + bi
        else:
            k_final = k3 + k1
            b_final = b3 + b1

        # Replace with single conv
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, 3, self.stride, 1,
            groups=self.groups, bias=True,
        ).to(k_final.device)
        self.fused_conv.weight.data = k_final
        self.fused_conv.bias.data = b_final

        # Remove training branches
        for attr in ('conv3x3', 'bn3x3', 'conv1x1', 'bn1x1', 'bn_identity'):
            if hasattr(self, attr):
                delattr(self, attr)

        self.deploy = True
