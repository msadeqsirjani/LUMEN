"""
SVD-based low-rank compression for LUMEN.

Inspired by MambaLiteSR (arXiv:2502.14090), which achieved 42% training power
reduction by applying low-rank factorization (rank=2) to Mamba mixer weights.

We apply the same idea to Conv2d pointwise (1x1) layers, which dominate
parameter count in LUMEN. A 1x1 Conv2d with weight (C_out, C_in, 1, 1)
is equivalent to a linear transform W ∈ R^(C_out x C_in), amenable to SVD.

Factorization: W ≈ U @ diag(S[:r]) @ V[:r]^T
Implemented as two sequential Conv2d layers:
    Conv2d(C_in, r, 1)   [V^T]
    Conv2d(r, C_out, 1)  [U * S]

Args for low_rank_compress:
    model: the LUMEN model to compress
    rank: target rank (e.g. 2, 4, 8, 16)
    target_layers: list of layer name substrings to compress (e.g. ['pw_in', 'pw_out'])
                   if None, compress all eligible Conv2d(1x1) layers
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class LowRankConv2d(nn.Module):
    """Two-layer factorized replacement for a 1x1 Conv2d.

    Decomposes W (C_out x C_in) into:
        W ≈ W_high (C_out x rank) @ W_low (rank x C_in)
    stored as two Conv2d(1x1) layers.

    Forward: x -> conv_low -> conv_high -> output
    Equivalent to original 1x1 conv but with rank * (C_in + C_out)
    parameters instead of C_in * C_out.
    """

    def __init__(self, in_channels: int, out_channels: int, rank: int,
                 bias: bool = False):
        super().__init__()
        self.rank = rank
        # Low: projects C_in -> rank
        self.conv_low = nn.Conv2d(in_channels, rank, 1, bias=False)
        # High: projects rank -> C_out (carries bias if needed)
        self.conv_high = nn.Conv2d(rank, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_high(self.conv_low(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _svd_init_low_rank(conv: nn.Conv2d, rank: int) -> LowRankConv2d:
    """Initialize a LowRankConv2d from an existing Conv2d via SVD.

    Only valid for 1x1 convolutions (kernel_size=1).
    The weight W has shape (C_out, C_in, 1, 1) → squeezed to (C_out, C_in).
    SVD: W = U @ diag(S) @ V^T
    Low-rank approximation: W_r = U[:, :r] @ diag(S[:r]) @ V[:, :r]^T

    We initialize:
        conv_low.weight  = V[:, :r]^T  shape (rank, C_in)
        conv_high.weight = U[:, :r] * S[:r]  shape (C_out, rank)
    """
    assert conv.kernel_size == (1, 1), "Only 1x1 convolutions can be low-rank factorized"

    W = conv.weight.data.squeeze(-1).squeeze(-1)  # (C_out, C_in)
    C_out, C_in = W.shape
    rank = min(rank, C_out, C_in)

    # SVD decomposition on CPU for stability
    W_cpu = W.float().cpu()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    # U: (C_out, min(C_out,C_in)), S: (min,), Vh: (min, C_in)

    lr_conv = LowRankConv2d(C_in, C_out, rank,
                             bias=(conv.bias is not None))

    # conv_low: weight shape (rank, C_in, 1, 1)
    # Initialize with V^T rows (Vh[:rank])
    lr_conv.conv_low.weight.data = Vh[:rank].unsqueeze(-1).unsqueeze(-1).to(conv.weight.device)

    # conv_high: weight shape (C_out, rank, 1, 1)
    # Initialize with U[:, :rank] * S[:rank]  (absorb singular values into high)
    W_high = U[:, :rank] * S[:rank].unsqueeze(0)  # (C_out, rank)
    lr_conv.conv_high.weight.data = W_high.unsqueeze(-1).unsqueeze(-1).to(conv.weight.device)

    if conv.bias is not None:
        lr_conv.conv_high.bias.data = conv.bias.data.clone()

    return lr_conv


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module):
    """Replace a child module by name, supporting dotted paths."""
    parts = name.split('.')
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def low_rank_compress(
    model: nn.Module,
    rank: int = 4,
    target_layers: Optional[list] = None,
    verbose: bool = True,
) -> nn.Module:
    """Apply SVD low-rank compression to eligible Conv2d(1x1) layers.

    Args:
        model: LUMEN model (modified in-place).
        rank: Target rank for factorization. Lower = smaller/faster but lower quality.
              Recommended values: 2 (aggressive), 4 (balanced), 8 (conservative).
        target_layers: List of layer name substrings to target. If None, all 1x1 convs.
              e.g. ['pw_in', 'pw_out', 'fc1', 'fc2']
        verbose: Print compression statistics.

    Returns:
        The compressed model (same object, modified in-place).
    """
    original_params = sum(p.numel() for p in model.parameters())

    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.kernel_size != (1, 1):
            continue
        if module.groups != 1:
            continue  # skip depthwise

        # Filter by target layers if specified
        if target_layers is not None:
            if not any(t in name for t in target_layers):
                continue

        C_out, C_in = module.weight.shape[:2]
        # Only compress if factorization actually saves parameters
        # rank * (C_in + C_out) < C_in * C_out
        if rank * (C_in + C_out) >= C_in * C_out:
            if verbose:
                print(f"  Skip {name} ({C_in}→{C_out}): rank={rank} not beneficial")
            continue

        replacements.append((name, module))

    # Apply replacements (after collecting to avoid modifying dict during iteration)
    for name, module in replacements:
        C_out, C_in = module.weight.shape[:2]
        lr_conv = _svd_init_low_rank(module, rank)
        _replace_module(model, name, lr_conv)
        if verbose:
            orig_p = C_in * C_out
            new_p = rank * (C_in + C_out)
            print(f"  Compressed {name}: ({C_in}→{C_out}) "
                  f"{orig_p:,} params → {new_p:,} params "
                  f"(rank={rank}, saved {100*(1-new_p/orig_p):.1f}%)")

    compressed_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\nTotal: {original_params:,} → {compressed_params:,} params "
              f"({100*(1-compressed_params/original_params):.1f}% reduction)")

    return model


def measure_compression(model_orig: nn.Module, model_compressed: nn.Module,
                         input_size: tuple = (1, 3, 32, 32)) -> dict:
    """Measure parameter count and inference MACs before/after compression."""
    p_orig = sum(p.numel() for p in model_orig.parameters())
    p_comp = sum(p.numel() for p in model_compressed.parameters())

    # Byte sizes (float32 = 4 bytes, int8 = 1 byte)
    size_fp32_orig = p_orig * 4
    size_fp32_comp = p_comp * 4
    size_int8_comp = p_comp * 1  # after INT8 quantization

    return {
        'params_original': p_orig,
        'params_compressed': p_comp,
        'param_reduction_pct': 100 * (1 - p_comp / p_orig),
        'size_fp32_original_kb': size_fp32_orig / 1024,
        'size_fp32_compressed_kb': size_fp32_comp / 1024,
        'size_int8_compressed_kb': size_int8_comp / 1024,
    }
