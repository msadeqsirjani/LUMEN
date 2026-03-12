#!/usr/bin/env python3
"""
Fourier feature visualization for LUMEN.

Hooks into intermediate blocks to extract feature maps, then computes
the 2D FFT log-amplitude spectrum. Follows PLKSR's Figure 4 protocol
for comparing frequency content across methods.

Usage:
    # Single model — visualize features after each block
    python scripts/fourier_viz.py \
        --checkpoint pretrained/LUMEN_x4.pth \
        --image datasets/Benchmarks/Urban100/LR_bicubic/X4/img_004.png \
        --scale 4

    # Compare two models at a specific block
    python scripts/fourier_viz.py \
        --checkpoint pretrained/LUMEN_x4.pth pretrained/PLKSR_x4.pth \
        --labels LUMEN PLKSR \
        --image img.png --scale 4 --block-idx -1
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from basicsr.archs.lumen.model import LUMEN


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
    """Load image as float32 RGB [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def extract_block_features(
    model: LUMEN,
    lr_img: np.ndarray,
    block_indices: list = None,
    device: str = "cuda",
) -> dict:
    """Extract feature maps after specified blocks via hooks.

    Args:
        model: Pretrained LUMEN model.
        lr_img: LR input, float32 [H, W, 3].
        block_indices: Which block indices to capture (default: all).
        device: Compute device.

    Returns:
        Dict mapping block_idx → feature tensor [C, H, W].
    """
    num_blocks = len(model.blocks)
    if block_indices is None:
        block_indices = list(range(num_blocks))
    # Handle negative indices
    block_indices = [i % num_blocks for i in block_indices]

    features = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            features[idx] = output.detach().cpu()
        return hook_fn

    for idx in block_indices:
        h = model.blocks[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    img_t = torch.from_numpy(lr_img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        model(img_t)

    for h in hooks:
        h.remove()

    return {k: v.squeeze(0) for k, v in features.items()}


def compute_fft_spectrum(features: torch.Tensor) -> np.ndarray:
    """Compute average log-amplitude FFT spectrum across channels.

    Args:
        features: Feature tensor [C, H, W].

    Returns:
        Log-amplitude spectrum [H, W], shifted so DC is center.
    """
    # FFT per channel
    fft = torch.fft.fft2(features)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    magnitude = fft_shifted.abs()

    # Average across channels
    avg_mag = magnitude.mean(dim=0).numpy()

    # Log scale
    log_amp = np.log1p(avg_mag)

    # Normalize to [0, 1]
    log_amp = log_amp - log_amp.min()
    if log_amp.max() > 0:
        log_amp = log_amp / log_amp.max()

    return log_amp


def visualize_spectra(
    spectra: list,
    labels: list,
    block_indices: list,
    output_path: str,
):
    """Create Fourier spectrum visualization grid.

    Args:
        spectra: List of lists — spectra[model_idx][block_idx].
        labels: Model names.
        block_indices: Block indices visualized.
        output_path: Save path.
    """
    n_models = len(spectra)
    n_blocks = len(block_indices)

    fig, axes = plt.subplots(
        n_models, n_blocks,
        figsize=(3 * n_blocks, 3 * n_models),
        squeeze=False,
    )

    for i, (model_spectra, label) in enumerate(zip(spectra, labels)):
        for j, (spec, bidx) in enumerate(zip(model_spectra, block_indices)):
            ax = axes[i][j]
            im = ax.imshow(spec, cmap="inferno", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(f"Block {bidx}", fontsize=10)
            if j == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Fourier Feature Spectra (log-amplitude)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Fourier visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fourier feature visualization")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="Model checkpoint(s)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Model labels")
    parser.add_argument("--image", required=True,
                        help="LR input image path")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--block-idx", type=int, nargs="+", default=None,
                        help="Block indices to visualize (default: first, mid, last)")
    parser.add_argument("--output", type=str, default="results/fourier_viz.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    lr_img = load_image(args.image)
    labels = args.labels or [Path(c).stem for c in args.checkpoint]

    all_spectra = []
    all_block_indices = None

    for ckpt, label in zip(args.checkpoint, labels):
        print(f"  Extracting features from {label} ...")
        model = load_model(ckpt, args.scale, args.device)

        num_blocks = len(model.blocks)
        if args.block_idx is not None:
            block_indices = args.block_idx
        else:
            # Default: first, middle, last
            block_indices = [0, num_blocks // 2, num_blocks - 1]

        if all_block_indices is None:
            all_block_indices = block_indices

        features = extract_block_features(model, lr_img, block_indices, args.device)
        model_spectra = [compute_fft_spectrum(features[i % num_blocks]) for i in block_indices]
        all_spectra.append(model_spectra)

        del model
        torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    visualize_spectra(all_spectra, labels, all_block_indices, args.output)


if __name__ == "__main__":
    main()
