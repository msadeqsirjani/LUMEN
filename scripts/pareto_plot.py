#!/usr/bin/env python3
"""
Pareto scatter plot: Latency vs PSNR for efficient SR methods.

Generates a publication-ready scatter plot showing the trade-off between
GPU latency and reconstruction quality, with bubble size proportional
to parameter count. Follows PLKSR's Figure 1 format.

Usage:
    # Use built-in baseline data + LUMEN benchmark
    python scripts/pareto_plot.py

    # With custom data from JSON
    python scripts/pareto_plot.py --data results/benchmark_data.json

    # Custom output
    python scripts/pareto_plot.py --output results/pareto.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Baseline methods data from published papers (x4 scale, BSD100, HD output)
# Sources: PLKSR Table 3/4, SPAN paper, MAN paper
BASELINES = {
    "FSRCNN": {
        "params_k": 12, "latency_ms": None,
        "psnr_set5": 30.72, "psnr_bsd100": 26.98,
    },
    "ESPCN": {
        "params_k": 20, "latency_ms": None,
        "psnr_set5": 30.90, "psnr_bsd100": 27.03,
    },
    "ShuffleMixer-T": {
        "params_k": 50, "latency_ms": None,
        "psnr_set5": 31.90, "psnr_bsd100": 27.42,
    },
    "ABPN": {
        "params_k": 20, "latency_ms": None,
        "psnr_set5": 32.01, "psnr_bsd100": 27.48,
    },
    "SeemoRe-T": {
        "params_k": 220, "latency_ms": None,
        "psnr_set5": 32.31, "psnr_bsd100": 27.65,
    },
    "SPAN-S": {
        "params_k": 411, "latency_ms": 12.22,
        "psnr_set5": 32.20, "psnr_bsd100": 27.60,
    },
    "PLKSR-tiny": {
        "params_k": 250, "latency_ms": 3.6,
        "psnr_set5": 32.33, "psnr_bsd100": 27.68,
    },
    "SwinIR-light": {
        "params_k": 878, "latency_ms": 34.2,
        "psnr_set5": 32.44, "psnr_bsd100": 27.69,
    },
    "SRFormer-light": {
        "params_k": 853, "latency_ms": 34.2,
        "psnr_set5": 32.51, "psnr_bsd100": 27.73,
    },
    "ELAN-light": {
        "params_k": 582, "latency_ms": 18.9,
        "psnr_set5": 32.43, "psnr_bsd100": 27.69,
    },
}


def load_data(data_path: str = None) -> dict:
    """Load method data from JSON or use built-in baselines."""
    methods = dict(BASELINES)

    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            custom = json.load(f)
        methods.update(custom)

    return methods


def create_pareto_plot(
    methods: dict,
    highlight: str = "LUMEN",
    metric: str = "psnr_bsd100",
    output_path: str = "results/pareto_plot.png",
):
    """Create Pareto scatter plot.

    X-axis: latency (ms) or params (K)
    Y-axis: PSNR
    Bubble size: params
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: Params vs PSNR ---
    ax1 = axes[0]
    for name, data in methods.items():
        params_k = data["params_k"]
        psnr = data.get(metric, 0)
        if not psnr:
            continue

        is_ours = highlight.lower() in name.lower()
        color = "#E63946" if is_ours else "#457B9D"
        marker = "*" if is_ours else "o"
        size = 200 if is_ours else 80
        zorder = 10 if is_ours else 5

        ax1.scatter(
            params_k, psnr,
            c=color, s=size, marker=marker,
            edgecolors="black", linewidths=0.5,
            zorder=zorder, label=name,
        )
        offset_y = 0.03 if is_ours else 0.02
        ax1.annotate(
            name, (params_k, psnr),
            textcoords="offset points",
            xytext=(5, 5 + (10 if is_ours else 0)),
            fontsize=7, fontweight="bold" if is_ours else "normal",
        )

    ax1.set_xlabel("Parameters (K)", fontsize=11)
    ax1.set_ylabel(f"PSNR (dB) — {metric.split('_')[-1]}", fontsize=11)
    ax1.set_title("Parameters vs Quality", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Latency vs PSNR ---
    ax2 = axes[1]
    for name, data in methods.items():
        lat = data.get("latency_ms")
        psnr = data.get(metric, 0)
        if not lat or not psnr:
            continue

        is_ours = highlight.lower() in name.lower()
        color = "#E63946" if is_ours else "#457B9D"
        marker = "*" if is_ours else "o"
        size = max(data["params_k"] / 3, 40)
        zorder = 10 if is_ours else 5

        ax2.scatter(
            lat, psnr,
            c=color, s=size, marker=marker,
            edgecolors="black", linewidths=0.5,
            zorder=zorder,
        )
        ax2.annotate(
            name, (lat, psnr),
            textcoords="offset points",
            xytext=(5, 5 + (10 if is_ours else 0)),
            fontsize=7, fontweight="bold" if is_ours else "normal",
        )

    ax2.set_xlabel("Latency (ms) — HD output, FP16", fontsize=11)
    ax2.set_ylabel(f"PSNR (dB) — {metric.split('_')[-1]}", fontsize=11)
    ax2.set_title("Latency vs Quality (bubble = params)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Efficient Super-Resolution — x4 Scale", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Pareto plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pareto plot for efficient SR")
    parser.add_argument("--data", type=str, default=None,
                        help="JSON file with method data (augments built-in baselines)")
    parser.add_argument("--highlight", type=str, default="LUMEN",
                        help="Method name to highlight")
    parser.add_argument("--metric", type=str, default="psnr_bsd100",
                        choices=["psnr_set5", "psnr_bsd100"],
                        help="PSNR metric to plot")
    parser.add_argument("--output", type=str, default="results/pareto_plot.png")
    args = parser.parse_args()

    methods = load_data(args.data)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    create_pareto_plot(methods, args.highlight, args.metric, args.output)


if __name__ == "__main__":
    main()
