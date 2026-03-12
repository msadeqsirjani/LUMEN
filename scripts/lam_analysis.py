#!/usr/bin/env python3
"""
LAM: Local Attribution Map analysis for LUMEN.

Implements the diffusion index (DI) and LAM visualization from
"Interpreting Super-Resolution Networks with Local Attribution Maps"
(Gu & Dong, CVPR 2021).

Algorithm: Integrated gradients — accumulate gradients of a central
output patch w.r.t. all input pixels over multiple interpolation steps
from a blurred baseline to the original input.

Usage:
    # Single model
    python scripts/lam_analysis.py \
        --checkpoint pretrained/LUMEN_x4.pth \
        --image datasets/Benchmarks/Urban100/LR_bicubic/X4/img_004.png \
        --scale 4

    # Compare multiple models
    python scripts/lam_analysis.py \
        --checkpoint pretrained/LUMEN_x4.pth pretrained/PLKSR_x4.pth \
        --labels LUMEN PLKSR \
        --image datasets/Benchmarks/Urban100/LR_bicubic/X4/img_004.png \
        --scale 4

    # Custom crop region
    python scripts/lam_analysis.py \
        --checkpoint pretrained/LUMEN_x4.pth \
        --image img.png --scale 4 \
        --crop 100 100 48 48
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from basicsr.archs.lumen.model import LUMEN


def load_model(checkpoint: str, scale: int, device: str = "cuda") -> LUMEN:
    """Load a pretrained LUMEN model."""
    model = LUMEN(upscale=scale)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Handle both raw state_dict and wrapped formats
    if "params_ema" in state:
        model.load_state_dict(state["params_ema"], strict=True)
    elif "params" in state:
        model.load_state_dict(state["params"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_image(path: str) -> np.ndarray:
    """Load image as float32 RGB [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def compute_lam(
    model: LUMEN,
    lr_img: np.ndarray,
    crop_y: int,
    crop_x: int,
    crop_h: int = 48,
    crop_w: int = 48,
    steps: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Compute Local Attribution Map via integrated gradients.

    Args:
        model: Pretrained SR model.
        lr_img: LR input image, float32 [H, W, 3], range [0, 1].
        crop_y, crop_x: Top-left corner of the output crop (in LR coords).
        crop_h, crop_w: Size of the output crop (in LR coords).
        steps: Number of interpolation steps.
        device: Compute device.

    Returns:
        Attribution map, shape [H, W], higher = more influential.
    """
    scale = model.upscale

    # Prepare input tensor
    img_t = torch.from_numpy(lr_img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Baseline: heavily blurred version of input
    baseline = F.avg_pool2d(F.pad(img_t, [11]*4, mode='reflect'), 23, 1)

    # Output crop region (in HR coordinates)
    oy, ox = crop_y * scale, crop_x * scale
    oh, ow = crop_h * scale, crop_w * scale

    attribution = torch.zeros_like(img_t)

    for step in range(steps):
        alpha = (step + 0.5) / steps
        interp = baseline + alpha * (img_t - baseline)
        interp = interp.detach().requires_grad_(True)

        out = model(interp)

        # Sum of output pixels in the crop region
        target = out[:, :, oy:oy+oh, ox:ox+ow].sum()
        target.backward()

        attribution += interp.grad * (img_t - baseline) / steps

    # Aggregate across channels → spatial attribution
    lam = attribution.squeeze(0).abs().sum(dim=0).cpu().numpy()

    # Normalize to [0, 1]
    lam = lam - lam.min()
    if lam.max() > 0:
        lam = lam / lam.max()

    return lam


def compute_di(lam: np.ndarray, percentile: float = 95.0) -> float:
    """Compute Diffusion Index — percentage of pixels above threshold.

    Higher DI = model uses wider input context (better long-range modeling).
    """
    threshold = np.percentile(lam, percentile)
    return float((lam >= threshold).sum()) / lam.size * 100.0


def visualize_lam(
    lr_img: np.ndarray,
    lams: list,
    labels: list,
    dis: list,
    crop_rect: tuple,
    output_path: str,
):
    """Create side-by-side LAM visualization.

    Args:
        lr_img: Original LR image [H, W, 3].
        lams: List of attribution maps.
        labels: List of model names.
        dis: List of diffusion indices.
        crop_rect: (y, x, h, w) of the analyzed crop.
        output_path: Path to save the figure.
    """
    n = len(lams) + 1  # +1 for input image
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 2:
        axes = [axes[0], axes[1]]

    # Input image with crop rectangle
    axes[0].imshow(lr_img)
    y, x, h, w = crop_rect
    rect = plt.Rectangle((x, y), w, h, linewidth=2,
                          edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].set_title("Input (LR)")
    axes[0].axis("off")

    # LAM heatmaps
    for i, (lam, label, di) in enumerate(zip(lams, labels, dis)):
        ax = axes[i + 1]
        ax.imshow(lr_img, alpha=0.3)
        im = ax.imshow(lam, cmap="jet", alpha=0.7, vmin=0, vmax=1)
        ax.set_title(f"{label}\nDI={di:.1f}%")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  LAM saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LAM analysis for LUMEN")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="Model checkpoint(s)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Model labels (default: checkpoint filenames)")
    parser.add_argument("--image", required=True,
                        help="LR input image path")
    parser.add_argument("--scale", type=int, default=4,
                        help="SR scale factor")
    parser.add_argument("--crop", type=int, nargs=4, default=None,
                        metavar=("Y", "X", "H", "W"),
                        help="Crop region in LR coords (default: center 48x48)")
    parser.add_argument("--steps", type=int, default=64,
                        help="Integrated gradient steps")
    parser.add_argument("--output", type=str, default="results/lam_analysis.png",
                        help="Output figure path")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    lr_img = load_image(args.image)
    h, w = lr_img.shape[:2]

    # Default crop: center
    if args.crop is None:
        ch, cw = 48, 48
        cy, cx = (h - ch) // 2, (w - cw) // 2
        args.crop = [cy, cx, ch, cw]

    labels = args.labels or [Path(c).stem for c in args.checkpoint]

    lams, dis = [], []
    for ckpt, label in zip(args.checkpoint, labels):
        print(f"  Computing LAM for {label} ...")
        model = load_model(ckpt, args.scale, args.device)
        lam = compute_lam(
            model, lr_img,
            args.crop[0], args.crop[1], args.crop[2], args.crop[3],
            steps=args.steps, device=args.device,
        )
        di = compute_di(lam)
        lams.append(lam)
        dis.append(di)
        print(f"    DI = {di:.1f}%")
        del model
        torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    visualize_lam(lr_img, lams, labels, dis, tuple(args.crop), args.output)


if __name__ == "__main__":
    main()
