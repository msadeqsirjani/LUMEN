#!/usr/bin/env python3
"""
LUMEN benchmark: MACs, FLOPs, parameters, GPU latency, and MGO.

Follows the PLKSR benchmark protocol:
  - "Restoring an HD (1280x720) image" = OUTPUT is HD
  - Input size = HD / scale (e.g., x4: 320x180 input → 1280x720 output)
  - Latency & MGO measured on GPU at FP16 precision
  - MGO = Maximum GPU memory Occupancy (torch.cuda.max_memory_allocated)

Usage:
    # Default: all scales, HD output, FP16 (matches PLKSR Table 4)
    python scripts/benchmark.py

    # Single scale
    python scripts/benchmark.py --upscale 4

    # With latency + MGO measurement
    python scripts/benchmark.py --latency

    # Custom output resolution
    python scripts/benchmark.py --output-size 720 1280 --latency

    # Custom LR input directly (overrides --output-size)
    python scripts/benchmark.py --input-size 1 3 180 320 --upscale 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basicsr.archs.lumen.model import LUMEN

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} G"
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.2f} K"
    return str(int(n))


def _lr_size(output_h: int, output_w: int, scale: int) -> tuple:
    """Compute LR input size from desired HR output size and scale."""
    return (1, 3, output_h // scale, output_w // scale)


# ---------------------------------------------------------------------------
# MACs / FLOPs / Parameters
# ---------------------------------------------------------------------------

def count_macs(model: nn.Module, input_size: tuple) -> dict:
    """Count MACs, FLOPs, and parameters.

    Prefers fvcore (more accurate), falls back to thop.
    """
    dummy = torch.zeros(input_size)
    params = sum(p.numel() for p in model.parameters())

    if HAS_FVCORE:
        flop_counter = FlopCountAnalysis(model, dummy)
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)
        macs = flop_counter.total()
    elif HAS_THOP:
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        macs = int(macs)
    else:
        raise RuntimeError(
            "Neither fvcore nor thop is installed. "
            "Install one: pip install fvcore  or  pip install thop"
        )

    return {"macs": macs, "flops": macs * 2, "params": params}


# ---------------------------------------------------------------------------
# MGO (Maximum GPU memory Occupancy) — matches PLKSR protocol
# ---------------------------------------------------------------------------

@torch.inference_mode()
def measure_mgo(
    model: nn.Module,
    input_size: tuple,
    device: str = "cuda",
    use_fp16: bool = False,
) -> float:
    """Measure Maximum GPU memory Occupancy in MB.

    Uses torch.cuda.max_memory_allocated() — same as PLKSR.
    Matches PLKSR protocol: model.half() for FP16 (not autocast).
    """
    if use_fp16:
        model = model.half().to(device).eval()
    else:
        model = model.to(device).eval()

    # Clean GPU state — must happen AFTER model is on device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dtype = torch.float16 if use_fp16 else torch.float32
    dummy = torch.randn(input_size, device=device, dtype=dtype)

    _ = model(dummy)

    torch.cuda.synchronize()
    mgo_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del dummy
    torch.cuda.empty_cache()

    return mgo_mb


# ---------------------------------------------------------------------------
# GPU latency measurement (PLKSR-style with CUDA events)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def measure_latency(
    model: nn.Module,
    input_size: tuple,
    device: str = "cuda",
    warmup: int = 50,
    repetitions: int = 300,
    use_fp16: bool = False,
) -> dict:
    """Measure GPU latency using CUDA event timing.

    Matches PLKSR protocol: model.half() for FP16 (not autocast).
    """
    if use_fp16:
        model = model.half().to(device).eval()
    else:
        model = model.to(device).eval()

    dtype = torch.float16 if use_fp16 else torch.float32
    dummy = torch.randn(input_size, device=device, dtype=dtype)

    use_cuda = device.startswith("cuda")

    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup):
        _ = model(dummy)

    if use_cuda:
        torch.cuda.synchronize()

    # Timed runs
    timings = np.zeros(repetitions)

    for i in range(repetitions):
        if use_cuda:
            starter.record()
            _ = model(dummy)
            ender.record()
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender)
        else:
            import time
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            timings[i] = (t1 - t0) * 1000.0

    avg_ms = np.mean(timings)
    median_ms = np.median(timings)
    std_ms = np.std(timings)
    fps = 1000.0 / avg_ms

    return {
        "avg_ms": avg_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_table_header(has_latency: bool):
    cols = f"{'Scale':<8} {'Params':<12} {'MACs':<12} {'FLOPs':<12} {'Input':<16}"
    if has_latency:
        cols += f" {'Latency(ms)':<12} {'MGO(mb)':<10} {'FPS':<10}"
    print(f"  {cols}")
    print(f"  {'-' * len(cols)}")


def print_table_row(scale: int, info: dict, input_size: tuple,
                    lat: dict = None, mgo_mb: float = 0.0):
    inp_str = f"{input_size[2]}x{input_size[3]}"
    row = (f"  x{scale:<7} "
           f"{_fmt(info['params']):<12} "
           f"{_fmt(info['macs']):<12} "
           f"{_fmt(info['flops']):<12} "
           f"{inp_str:<16}")
    if lat:
        row += (f" {lat['avg_ms']:<12.2f} "
                f"{mgo_mb:<10.1f} "
                f"{lat['fps']:<10.1f}")
    print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LUMEN benchmark (PLKSR protocol)")
    p.add_argument("--upscale", type=int, default=None,
                   help="Single scale to benchmark (default: all scales)")
    p.add_argument("--output-size", type=int, nargs=2, default=[720, 1280],
                   metavar=("H", "W"),
                   help="HR output resolution (default: 720 1280 = HD)")
    p.add_argument("--input-size", type=int, nargs=4, default=None,
                   metavar=("B", "C", "H", "W"),
                   help="Override: direct LR input size (ignores --output-size)")
    p.add_argument("--latency", action="store_true",
                   help="Measure GPU latency and MGO")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use FP16 (default: True, matching PLKSR protocol)")
    p.add_argument("--no-fp16", action="store_true",
                   help="Disable FP16, use FP32")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repetitions", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_fp16:
        args.fp16 = False

    scales = [args.upscale] if args.upscale else [2, 3, 4]
    out_h, out_w = args.output_size

    backend = "fvcore" if HAS_FVCORE else ("thop" if HAS_THOP else "none")
    print(f"\n  LUMEN Benchmark  (MAC counter: {backend})")
    print(f"  Output HD: {out_h}x{out_w}  FP16: {args.fp16}")
    if args.latency:
        print(f"  Device: {args.device}  Warmup: {args.warmup}  Reps: {args.repetitions}")
    print()

    print_table_header(args.latency)
    for s in scales:
        # Compute correct LR input size: HD / scale
        if args.input_size:
            input_size = tuple(args.input_size)
        else:
            input_size = _lr_size(out_h, out_w, s)

        model = LUMEN(upscale=s)
        model.fuse()
        model.eval()

        info = count_macs(model, input_size)
        lat = None
        mgo = 0.0
        if args.latency:
            mgo = measure_mgo(model, input_size, args.device, args.fp16)
            lat = measure_latency(
                model, input_size, args.device,
                args.warmup, args.repetitions, args.fp16,
            )

        print_table_row(s, info, input_size, lat, mgo)

    print()


if __name__ == "__main__":
    main()
