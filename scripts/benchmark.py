#!/usr/bin/env python3
"""
LUMEN MACs / FLOPs / parameter counter.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --input-size 1 3 64 64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from thop import profile as thop_profile

from basicsr.archs.lumen.model import LUMEN


def _fmt(n):
    if n >= 1e9: return f"{n/1e9:.2f} G"
    if n >= 1e6: return f"{n/1e6:.2f} M"
    if n >= 1e3: return f"{n/1e3:.2f} K"
    return str(int(n))


def benchmark(model, input_size):
    dummy = torch.zeros(input_size)
    params = sum(p.numel() for p in model.parameters())
    macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    flops   = macs * 2

    print(f"  LUMEN         "
          f"Params: {_fmt(params):<10}  "
          f"MACs: {_fmt(macs):<10}  "
          f"FLOPs: {_fmt(flops)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--upscale', type=int, default=4)
    p.add_argument('--input-size', type=int, nargs=4, default=[1, 3, 32, 32],
                   metavar=('B', 'C', 'H', 'W'))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n  Input: {args.input_size}  Upscale: x{args.upscale}\n")
    benchmark(LUMEN(upscale=args.upscale), args.input_size)
    print()


if __name__ == '__main__':
    main()
