"""
LUMEN: Lightweight Multi-Scale Network for Efficient Super-Resolution.

Architecture:
    Input (LR) → Conv3x3 → N × LumenBlock → Conv3x3 + skip → Upsample → Output (HR)

Each LumenBlock: RepDCCM → EPMSDM → PFA → Conv1x1 + residual

Configs:
    LUMEN:      embed_dim=48, num_blocks=16, partial_ch=12, large_kernel=21
    LUMEN-tiny: embed_dim=32, num_blocks=8,  partial_ch=8,  large_kernel=17
"""

import torch
import torch.nn as nn

from .lumen_block import LumenBlock


def _init_weights(m: nn.Module):
    """Initialize weights for stable training."""
    if isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LUMEN(nn.Module):
    """LUMEN: Lightweight Multi-Scale Network for Efficient Super-Resolution.

    GPU-efficient architecture combining:
      - RepDCCM: re-parameterizable channel mixer (train multi-branch, infer single conv)
      - EPMSDM: three-scale partial depthwise spatial mixer (fine/medium/global)
      - PFA: parameter-free attention (zero extra params)
      - StarReLU: efficient activation (4 FLOPs vs GELU's 14)

    Args:
        num_in_ch: Number of input image channels (3 for RGB).
        num_out_ch: Number of output image channels.
        embed_dim: Feature channel dimension.
        num_blocks: Total number of LumenBlocks.
        partial_ch: Channels per active branch in spatial mixer.
        large_kernel: Size of decomposed large kernel.
        upscale: Super-resolution scale factor (2, 3, or 4).
        img_range: Image value range for normalization (1.0 for [0,1]).
        upsampler: Upsampling mode:
            'pixelshuffledirect' - single PixelShuffle (default, lightweight)
            'pixelshuffle' - progressive PixelShuffle for larger scales
            '' - no upsampling (for denoising / artifact removal)
        use_reparam: Use RepDCCM (True) or plain DCCM (False).
        spatial_mixer: 'epmsdm' (3-scale) or 'pmsdm' (2-scale original).
        attention: 'pfa' (parameter-free) or 'none'.
        activation: 'star_relu' or 'gelu'.
        deploy: If True, initialize with fused re-param convolutions.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        embed_dim: int = 32,
        num_blocks: int = 8,
        partial_ch: int = 8,
        large_kernel: int = 17,
        upscale: int = 4,
        img_range: float = 1.0,
        upsampler: str = 'pixelshuffledirect',
        use_reparam: bool = True,
        spatial_mixer: str = 'epmsdm',
        attention: str = 'pfa',
        activation: str = 'star_relu',
        deploy: bool = False,
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

        # Deep feature extraction: N × LumenBlock (flat, no groups)
        self.blocks = nn.Sequential(*[
            LumenBlock(
                embed_dim, partial_ch, large_kernel,
                use_reparam=use_reparam,
                spatial_mixer=spatial_mixer,
                attention=attention,
                activation=activation,
                deploy=deploy,
            )
            for _ in range(num_blocks)
        ])

        # Deep feature aggregation
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Upsampler / reconstruction head
        if upsampler == 'pixelshuffledirect':
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_ch * (upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale),
            )
        elif upsampler == '':
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize
        x = (x - self.mean) * self.img_range

        # Feature extraction with global skip
        feat = self.conv_first(x)
        body = self.conv_after_body(self.blocks(feat)) + feat

        # Reconstruction
        if self.upsampler == 'pixelshuffledirect':
            x = self.upsample(body)
        elif self.upsampler == '':
            x = self.conv_last(body)

        # De-normalize
        x = x / self.img_range + self.mean
        return x

    def fuse(self):
        """Collapse all re-param branches for inference deployment."""
        for block in self.blocks:
            block.fuse()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
