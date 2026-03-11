"""
Lightweight training utilities for standalone LUMEN scripts
(scripts/train.py, scripts/test.py, scripts/train_distill.py, etc.)

These complement BasicSR's heavier pipeline utilities with simple,
script-friendly helpers that don't require the full BasicSR config system.
"""

import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = 'lumen', log_file: str = None) -> logging.Logger:
    """Get a logger writing to stdout and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '[%(asctime)s %(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_psnr: float,
    path: str,
    is_best: bool = False,
):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'best_psnr': best_psnr,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, path)
    if is_best:
        torch.save(state, path.replace('.pth', '_best.pth'))


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = 'cpu',
) -> dict:
    """Load a checkpoint. Returns {'epoch': int, 'best_psnr': float}."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model'])
    if optimizer and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler and state.get('scheduler'):
        scheduler.load_state_dict(state['scheduler'])
    return {'epoch': state.get('epoch', 0), 'best_psnr': state.get('best_psnr', 0.0)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 10.0 * torch.log10(max_val ** 2 / mse).item()


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, max_val: float = 1.0) -> float:
    """SSIM on Y-channel (luminance), following standard SR evaluation."""
    import torch.nn.functional as F

    with torch.no_grad():
        if pred.dim() == 3:
            pred, target = pred.unsqueeze(0), target.unsqueeze(0)
        if pred.shape[1] == 3:
            coeff = pred.new_tensor([0.257, 0.504, 0.098]).view(1, 3, 1, 1)
            pred   = (pred   * coeff).sum(1, keepdim=True) + 16.0 / 255.0
            target = (target * coeff).sum(1, keepdim=True) + 16.0 / 255.0

        sigma = 1.5
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        pad = window_size // 2
        C1, C2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2

        mu_x = F.conv2d(pred,   window, padding=pad)
        mu_y = F.conv2d(target, window, padding=pad)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sig_x2 = F.conv2d(pred   * pred,   window, padding=pad) - mu_x2
        sig_y2 = F.conv2d(target * target, window, padding=pad) - mu_y2
        sig_xy = F.conv2d(pred   * target, window, padding=pad) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
        return (num / den).mean().item()


class AverageMeter:
    """Running average tracker for loss/metric logging."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum, self.count = 0.0, 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
