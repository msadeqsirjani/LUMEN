#!/usr/bin/env python3
"""
Side-by-side visual comparison for super-resolution results.

Generates publication-ready comparison figures with zoomed crop regions
and per-crop PSNR annotations, following PLKSR's Figure 7 format.

Usage:
    # Compare models on an image with auto-detected challenging crop
    python scripts/visual_compare.py \
        --gt datasets/Benchmarks/Urban100/HR/img_004.png \
        --lr datasets/Benchmarks/Urban100/LR_bicubic/X4/img_004.png \
        --checkpoint pretrained/LUMEN_x4.pth pretrained/PLKSR_x4.pth \
        --labels LUMEN PLKSR-tiny \
        --scale 4

    # Custom crop region (in HR coordinates)
    python scripts/visual_compare.py \
        --gt img_hr.png --lr img_lr.png \
        --checkpoint model_a.pth model_b.pth \
        --labels A B --scale 4 \
        --crop 200 300 128 128
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from basicsr.archs.lumen.model import LUMEN
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def load_model(checkpoint: str, scale: int, device: str = "cuda") -> LUMEN:
    """Load a pretrained LUMEN model."""
    model = LUMEN(upscale=scale)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "params_ema" in state:
        model.load_state_dict(state["params_ema"], strict=True)
    elif "params" in state:
        model.load_state_dict(state["params"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_image(path: str) -> np.ndarray:
    """Load image as uint8 RGB."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_sr(model: LUMEN, lr_img: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Run SR inference, return uint8 RGB."""
    img_f = lr_img.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)

    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def find_interesting_crop(
    gt: np.ndarray,
    lr: np.ndarray,
    scale: int,
    crop_size: int = 128,
) -> tuple:
    """Find a high-variance crop region (challenging for SR).

    Returns (y, x) in HR coordinates.
    """
    # Compute local variance on GT
    gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    best_var = 0
    best_y, best_x = 0, 0
    step = crop_size // 2

    for y in range(0, h - crop_size, step):
        for x in range(0, w - crop_size, step):
            patch = gray[y:y+crop_size, x:x+crop_size]
            var = patch.var()
            if var > best_var:
                best_var = var
                best_y, best_x = y, x

    return best_y, best_x


def compute_crop_psnr(sr: np.ndarray, gt: np.ndarray, crop: tuple) -> float:
    """Compute PSNR on a specific crop region."""
    y, x, h, w = crop
    sr_crop = sr[y:y+h, x:x+w]
    gt_crop = gt[y:y+h, x:x+w]
    return calculate_psnr(sr_crop, gt_crop, crop_border=0, test_y_channel=True)


def create_comparison(
    gt: np.ndarray,
    bicubic: np.ndarray,
    sr_results: list,
    labels: list,
    psnrs: list,
    crop: tuple,
    output_path: str,
    image_name: str = "",
):
    """Create publication-ready comparison figure.

    Layout: [Full image with crop box] [GT crop] [Bicubic crop] [Model1 crop] ...
    """
    y, x, h, w = crop
    n_cols = len(sr_results) + 2  # GT crop + Bicubic crop + N models

    fig, axes = plt.subplots(1, n_cols + 1, figsize=(3.2 * (n_cols + 1), 3.5))

    # Full GT image with crop rectangle
    axes[0].imshow(gt)
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=2, edgecolor='red', facecolor='none',
    )
    axes[0].add_patch(rect)
    axes[0].set_title(image_name or "Ground Truth", fontsize=9)
    axes[0].axis("off")

    # GT crop
    axes[1].imshow(gt[y:y+h, x:x+w])
    axes[1].set_title("HR (GT)", fontsize=9)
    axes[1].axis("off")

    # Bicubic crop
    bic_psnr = calculate_psnr(
        bicubic[y:y+h, x:x+w], gt[y:y+h, x:x+w],
        crop_border=0, test_y_channel=True,
    )
    axes[2].imshow(bicubic[y:y+h, x:x+w])
    axes[2].set_title(f"Bicubic\n{bic_psnr:.2f} dB", fontsize=9)
    axes[2].axis("off")

    # Model crops
    for i, (sr, label, psnr) in enumerate(zip(sr_results, labels, psnrs)):
        ax = axes[3 + i]
        ax.imshow(sr[y:y+h, x:x+w])
        ax.set_title(f"{label}\n{psnr:.2f} dB", fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visual comparison for SR methods",
    )
    parser.add_argument("--gt", required=True, help="HR ground truth image")
    parser.add_argument("--lr", required=True, help="LR input image")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="Model checkpoint(s)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Model labels")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop", type=int, nargs=4, default=None,
                        metavar=("Y", "X", "H", "W"),
                        help="Crop region in HR coordinates")
    parser.add_argument("--crop-size", type=int, default=128,
                        help="Auto-crop size if --crop not specified")
    parser.add_argument("--output", type=str,
                        default="results/visual_compare.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    gt = load_image(args.gt)
    lr = load_image(args.lr)
    labels = args.labels or [Path(c).stem for c in args.checkpoint]

    # Bicubic upscale for reference
    bicubic = cv2.resize(
        lr, (gt.shape[1], gt.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    # Determine crop region
    if args.crop:
        crop = tuple(args.crop)
    else:
        cy, cx = find_interesting_crop(gt, lr, args.scale, args.crop_size)
        crop = (cy, cx, args.crop_size, args.crop_size)
        print(f"  Auto-selected crop: y={cy}, x={cx}, "
              f"h={args.crop_size}, w={args.crop_size}")

    # Run SR for each model
    sr_results, psnrs = [], []
    for ckpt, label in zip(args.checkpoint, labels):
        print(f"  Running SR with {label} ...")
        model = load_model(ckpt, args.scale, args.device)
        sr = run_sr(model, lr, args.device)
        psnr = compute_crop_psnr(sr, gt, crop)
        sr_results.append(sr)
        psnrs.append(psnr)
        print(f"    Crop PSNR: {psnr:.2f} dB")
        del model
        torch.cuda.empty_cache()

    image_name = Path(args.gt).stem
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    create_comparison(
        gt, bicubic, sr_results, labels, psnrs,
        crop, args.output, image_name,
    )


if __name__ == "__main__":
    main()
