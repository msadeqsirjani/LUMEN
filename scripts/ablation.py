#!/usr/bin/env python3
"""
Automated ablation runner for LUMEN.

Iterates through ablation YAML configs, launches training via BasicSR,
and collects results into a summary table.

Usage:
    # Dry run — list configs without training
    python scripts/ablation.py --dry-run

    # Run all ablation experiments
    python scripts/ablation.py

    # Run a specific subset
    python scripts/ablation.py --filter kernel

    # Only collect results from completed experiments
    python scripts/ablation.py --collect-only
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ABLATION_DIR = ROOT / "options" / "train" / "ablation"
EXPERIMENTS_DIR = ROOT / "experiments" / "ablation"


def find_configs(filter_str: str = None):
    """Find all ablation YAML configs, optionally filtered."""
    configs = sorted(glob.glob(str(ABLATION_DIR / "ablation_*.yml")))
    if filter_str:
        configs = [c for c in configs if filter_str in os.path.basename(c)]
    return configs


def parse_experiment_name(config_path: str) -> str:
    """Extract experiment name from YAML config."""
    with open(config_path) as f:
        opt = yaml.safe_load(f)
    return opt.get("name", Path(config_path).stem)


def get_best_psnr(exp_dir: Path) -> dict:
    """Parse validation logs to find best PSNR/SSIM from an experiment."""
    log_files = list(exp_dir.rglob("*.log"))
    best_psnr = 0.0
    best_ssim = 0.0
    best_iter = 0

    for log_file in log_files:
        with open(log_file) as f:
            for line in f:
                if "psnr:" in line and "ssim:" in line:
                    try:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "psnr:":
                                psnr = float(parts[i + 1].rstrip(","))
                            if p == "ssim:":
                                ssim = float(parts[i + 1].rstrip(","))
                            if p == "iter:":
                                it = int(parts[i + 1].rstrip(","))
                        if psnr > best_psnr:
                            best_psnr = psnr
                            best_ssim = ssim
                            best_iter = it
                    except (ValueError, IndexError):
                        continue

    return {"psnr": best_psnr, "ssim": best_ssim, "iter": best_iter}


def count_params(config_path: str) -> dict:
    """Instantiate model from config and count parameters."""
    import torch
    from basicsr.archs.lumen.model import LUMEN

    with open(config_path) as f:
        opt = yaml.safe_load(f)

    net_opt = opt["network_g"].copy()
    net_opt.pop("type", None)
    model = LUMEN(**net_opt)

    train_params = sum(p.numel() for p in model.parameters())

    # Fuse if re-param is enabled
    has_reparam = net_opt.get("use_reparam", True)
    if has_reparam:
        # Need forward passes to update BN stats for meaningful fuse
        model.train()
        dummy = torch.randn(2, 3, 32, 32)
        for _ in range(5):
            model(dummy)
        model.eval()
        model.fuse()

    fused_params = sum(p.numel() for p in model.parameters())

    return {"train_params": train_params, "fused_params": fused_params}


def run_training(config_path: str, gpu_id: int = 0):
    """Launch BasicSR training for a single config."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-m", "basicsr.train",
        "-opt", config_path,
    ]

    print(f"\n{'='*60}")
    print(f"  Training: {Path(config_path).stem}")
    print(f"  Config:   {config_path}")
    print(f"  GPU:      {gpu_id}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    return result.returncode == 0


def collect_results(configs: list) -> list:
    """Collect results from all experiments."""
    rows = []
    for cfg in configs:
        name = parse_experiment_name(cfg)
        exp_dir = EXPERIMENTS_DIR / name

        # Count params
        try:
            params = count_params(cfg)
        except Exception as e:
            params = {"train_params": 0, "fused_params": 0}
            print(f"  Warning: could not count params for {name}: {e}")

        # Get best metrics
        if exp_dir.exists():
            metrics = get_best_psnr(exp_dir)
        else:
            metrics = {"psnr": 0.0, "ssim": 0.0, "iter": 0}

        rows.append({
            "name": name,
            "train_params": params["train_params"],
            "fused_params": params["fused_params"],
            "best_psnr": metrics["psnr"],
            "best_ssim": metrics["ssim"],
            "best_iter": metrics["iter"],
        })

    return rows


def print_table(rows: list):
    """Pretty-print results table."""
    print(f"\n{'='*90}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'='*90}")
    header = (
        f"  {'Name':<30} {'Train Params':>13} {'Fused Params':>13}"
        f" {'PSNR':>8} {'SSIM':>8} {'Best Iter':>10}"
    )
    print(header)
    print(f"  {'-'*84}")

    for r in rows:
        tp = f"{r['train_params']/1e3:.1f}K" if r['train_params'] else "—"
        fp = f"{r['fused_params']/1e3:.1f}K" if r['fused_params'] else "—"
        psnr = f"{r['best_psnr']:.2f}" if r['best_psnr'] else "—"
        ssim = f"{r['best_ssim']:.4f}" if r['best_ssim'] else "—"
        it = f"{r['best_iter']}" if r['best_iter'] else "—"
        print(f"  {r['name']:<30} {tp:>13} {fp:>13} {psnr:>8} {ssim:>8} {it:>10}")

    print()


def save_csv(rows: list, path: str):
    """Save results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="LUMEN ablation runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="List configs without training")
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect results, no training")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter configs by substring (e.g., 'kernel')")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--output", type=str, default="experiments/ablation/results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    configs = find_configs(args.filter)
    if not configs:
        print("No ablation configs found.")
        return

    print(f"\n  Found {len(configs)} ablation configs:")
    for c in configs:
        name = parse_experiment_name(c)
        print(f"    - {name}")

    if args.dry_run:
        # Show param counts
        rows = collect_results(configs)
        print_table(rows)
        return

    if not args.collect_only:
        # Run training for each config
        for cfg in configs:
            name = parse_experiment_name(cfg)
            exp_dir = EXPERIMENTS_DIR / name
            if exp_dir.exists():
                print(f"  Skipping {name} (already exists). Delete to re-run.")
                continue
            success = run_training(cfg, args.gpu)
            if not success:
                print(f"  WARNING: Training failed for {name}")

    # Collect and display results
    rows = collect_results(configs)
    print_table(rows)

    # Save CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_csv(rows, args.output)


if __name__ == "__main__":
    main()
