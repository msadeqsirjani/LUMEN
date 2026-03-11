"""
LUMEN: Ultra-Lightweight Super-Resolution for STM32 Cortex-M.

Architecture:
    Input → Conv3x3 [shallow] → N × LumenGroup → Conv3x3 [deep]
          → + global skip → Upsample → Output

Default config: embed_dim=32, depths=(4,4), expand_ratio=2.5 → ~141K params
"""

import math
import torch
import torch.nn as nn

from .blocks import LumenGroup


def _init_weights(m: nn.Module):
    """Initialize weights for stable training."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class LUMEN(nn.Module):
    """LUMEN: Ultra-Lightweight Super-Resolution Network.

    Designed for deployment on STM32 Cortex-M MCUs via TFLite Micro.
    All operations are INT8-quantization friendly and CMSIS-NN compatible.

    Architecture:
        Input (LR) → shallow conv → N×LumenGroup → deep conv
                   → + global skip → PixelShuffle → Output (HR)

    Args:
        num_in_ch: Number of input image channels (3 for RGB, 1 for gray).
        num_out_ch: Number of output image channels.
        embed_dim: Feature channel dimension throughout the network.
        depths: Tuple of ints, number of LumenBlocks per group.
        expand_ratio: FFN expansion ratio (hidden = dim * expand_ratio).
        upscale: Super-resolution scale factor (2 or 4).
        img_range: Image value range for normalization (1.0 for [0,1]).
        upsampler: Upsampling mode:
            'pixelshuffledirect' - lightweight single PixelShuffle (default)
            'pixelshuffle' - progressive PixelShuffle for larger scales
            '' - no upsampling (for denoising / artifact removal)
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        embed_dim: int = 32,
        depths: tuple = (4, 4),
        expand_ratio: float = 2.5,
        upscale: int = 4,
        img_range: float = 1.0,
        upsampler: str = 'pixelshuffledirect',
    ):
        super().__init__()
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler

        # RGB normalization buffer (following SwinIR convention)
        if num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.register_buffer('mean', torch.Tensor(rgb_mean).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, 1, 1, 1))

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # Deep feature extraction: N × LumenGroup
        self.groups = nn.ModuleList([
            LumenGroup(embed_dim, depths[i], expand_ratio)
            for i in range(len(depths))
        ])

        # Deep feature aggregation
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Upsampler / reconstruction head
        if upsampler == 'pixelshuffledirect':
            # Lightweight: single PixelShuffle (best for MCU, minimal memory)
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_ch * (upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale),
            )
        elif upsampler == 'pixelshuffle':
            # Progressive PixelShuffle for power-of-2 scales
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.ReLU6(inplace=True),
            )
            upsample_layers = []
            for _ in range(int(math.log2(upscale))):
                upsample_layers += [
                    nn.Conv2d(embed_dim, 4 * embed_dim, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.ReLU6(inplace=True),
                ]
            self.upsample = nn.Sequential(*upsample_layers)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        elif upsampler == '':
            # No upsampling: for denoising / JPEG artifact removal
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(_init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for group in self.groups:
            x = group(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffledirect':
            feat = self.conv_first(x)
            body = self.forward_features(feat)
            body = self.conv_after_body(body) + feat
            x = self.upsample(body)

        elif self.upsampler == 'pixelshuffle':
            feat = self.conv_first(x)
            body = self.forward_features(feat)
            body = self.conv_after_body(body) + feat
            body = self.conv_before_upsample(body)
            x = self.conv_last(self.upsample(body))

        elif self.upsampler == '':
            feat = self.conv_first(x)
            body = self.forward_features(feat)
            body = self.conv_after_body(body) + feat
            x = self.conv_last(body)

        # De-normalize
        x = x / self.img_range + self.mean
        return x

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


