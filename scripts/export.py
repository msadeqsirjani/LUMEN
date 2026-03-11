#!/usr/bin/env python3
"""
Export LUMEN to ONNX and TorchScript for MCU/edge deployment.

Usage:
    python scripts/export.py --weights pretrained/LUMEN_x4.pth
    python scripts/export.py --weights pretrained/LUMEN_x4.pth --onnx --torchscript
    python scripts/export.py --weights pretrained/LUMEN_x4.pth --input-size 1 3 32 32
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from basicsr.archs.lumen.model import LUMEN
from basicsr.utils import get_logger, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='LUMEN Export')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--upscale', type=int, default=4)
    parser.add_argument('--input-size', type=int, nargs=4, default=[1, 3, 32, 32],
                        metavar=('B', 'C', 'H', 'W'))
    parser.add_argument('--output-dir', type=str, default='exported')
    parser.add_argument('--onnx', action='store_true', default=True)
    parser.add_argument('--torchscript', action='store_true', default=True)
    parser.add_argument('--opset', type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger('lumen_export')
    os.makedirs(args.output_dir, exist_ok=True)

    model = LUMEN(upscale=args.upscale).eval()
    load_checkpoint(args.weights, model)
    logger.info(f"Loaded: {args.weights}")

    dummy = torch.zeros(args.input_size)
    stem = f"LUMEN_x{args.upscale}"

    if args.onnx:
        try:
            import onnx
            out_path = os.path.join(args.output_dir, f"{stem}.onnx")
            torch.onnx.export(
                model, dummy, out_path,
                input_names=['lq'], output_names=['sr'],
                dynamic_axes={'lq': {0: 'batch', 2: 'height', 3: 'width'},
                              'sr': {0: 'batch', 2: 'height', 3: 'width'}},
                opset_version=args.opset,
                do_constant_folding=True,
            )
            logger.info(f"ONNX saved: {out_path}")
        except ImportError:
            logger.warning("onnx not installed; skipping ONNX export")

    if args.torchscript:
        out_path = os.path.join(args.output_dir, f"{stem}.pt")
        traced = torch.jit.trace(model, dummy)
        torch.jit.save(traced, out_path)
        logger.info(f"TorchScript saved: {out_path}")


if __name__ == '__main__':
    main()
