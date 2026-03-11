#!/usr/bin/env python3
"""
SVD low-rank compression for LUMEN.

Applies low-rank factorization to 1x1 Conv2d layers to reduce parameter
count and inference MACs for tighter MCU deployment targets.

Usage:
    python scripts/compress.py --rank 4
    python scripts/compress.py --weights pretrained/LUMEN_x4.pth --rank 4 --output compressed/LUMEN_x4_r4.pth
"""

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from basicsr.archs.lumen.model import LUMEN
from basicsr.compress import low_rank_compress, measure_compression
from basicsr.utils import get_logger, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='LUMEN Low-Rank Compression')
    parser.add_argument('--weights', type=str, default=None, help='Pre-trained weights')
    parser.add_argument('--rank', type=int, default=4, help='Target rank for SVD factorization')
    parser.add_argument('--output', type=str, default=None, help='Output path for compressed model')
    parser.add_argument('--upscale', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger('lumen_compress')

    model = LUMEN(upscale=args.upscale)
    if args.weights:
        load_checkpoint(args.weights, model)
        logger.info(f"Loaded weights: {args.weights}")

    orig = copy.deepcopy(model)
    logger.info(f"Applying SVD compression (rank={args.rank})...")
    low_rank_compress(model, rank=args.rank, verbose=True)

    stats = measure_compression(orig, model)
    logger.info(f"\nParam reduction: {stats['param_reduction_pct']:.1f}%")
    logger.info(f"Size (FP32): {stats['size_fp32_original_kb']:.1f} KB -> {stats['size_fp32_compressed_kb']:.1f} KB")
    logger.info(f"Size (INT8): {stats['size_int8_compressed_kb']:.1f} KB")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        torch.save({'model': model.state_dict()}, args.output)
        logger.info(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
