"""
SR Dataset loaders for LUMEN training.

Supports the standard DIV2K/DF2K directory structure used by FTANet:
    datasets/
    ├── DIV2K/
    │   ├── HR/Train/         (high-resolution training images)
    │   └── LR_bicubic/Train/X4/  (low-resolution inputs)
    └── Benchmarks/
        ├── Set5/
        │   ├── HR/
        │   └── LR_bicubic/X4/
        └── Set14/, BSDS100/, Urban100/, Manga109/
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_img(path: str) -> np.ndarray:
    """Read image as float32 RGB in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert (H, W, C) float32 numpy to (C, H, W) float32 tensor."""
    return torch.from_numpy(img.transpose(2, 0, 1))


def augment(hr: np.ndarray, lr: np.ndarray, hflip: bool = True, vflip: bool = True, rot: bool = True):
    """Apply random flips and rotation to paired HR/LR images."""
    if hflip and random.random() < 0.5:
        hr = hr[:, ::-1, :].copy()
        lr = lr[:, ::-1, :].copy()
    if vflip and random.random() < 0.5:
        hr = hr[::-1, :, :].copy()
        lr = lr[::-1, :, :].copy()
    if rot and random.random() < 0.5:
        hr = hr.transpose(1, 0, 2).copy()
        lr = lr.transpose(1, 0, 2).copy()
    return hr, lr


class SRPairDataset(Dataset):
    """Paired HR/LR Super-Resolution Dataset.

    Supports two modes:
      - 'folder': HR and LR in separate directories (DIV2K style)
      - 'single': Only HR provided, LR created by bicubic downsampling

    Args:
        hr_dir: Path to HR image directory.
        lr_dir: Path to LR image directory. If None, LR is generated on-the-fly.
        patch_size: HR patch size for random crop. 0 for full image (testing).
        scale: Upscaling factor.
        augment: Apply random flips/rotation during training.
        cache: Load all images into RAM (fast training, requires enough RAM).
        extensions: Valid image file extensions.
    """

    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        hr_dir: str,
        lr_dir: Optional[str] = None,
        patch_size: int = 256,
        scale: int = 4,
        augment: bool = True,
        cache: bool = False,
        extensions: Optional[set] = None,
    ):
        super().__init__()
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.patch_size = patch_size
        self.scale = scale
        self.do_augment = augment
        self.cache = cache
        self.extensions = extensions or self.EXTENSIONS

        # Collect image paths
        self.hr_paths = sorted([
            p for p in self.hr_dir.iterdir()
            if p.suffix.lower() in self.extensions
        ])
        if not self.hr_paths:
            raise RuntimeError(f"No images found in {hr_dir}")

        if self.lr_dir is not None:
            self.lr_paths = [
                self._find_lr(p) for p in self.hr_paths
            ]
        else:
            self.lr_paths = [None] * len(self.hr_paths)

        # Optional: cache all images in RAM
        self._hr_cache = {}
        self._lr_cache = {}
        if self.cache:
            print(f"Caching {len(self.hr_paths)} images...")
            for i, (hp, lp) in enumerate(zip(self.hr_paths, self.lr_paths)):
                self._hr_cache[i] = read_img(str(hp))
                if lp is not None:
                    self._lr_cache[i] = read_img(str(lp))

    def _find_lr(self, hr_path: Path) -> Path:
        """Find the corresponding LR image for a given HR path."""
        # Try exact name match first
        lr_path = self.lr_dir / hr_path.name
        if lr_path.exists():
            return lr_path
        # Try without 'x4' suffix convention (e.g., '0001x4.png' for HR '0001.png')
        stem = hr_path.stem
        for ext in self.extensions:
            for suffix in [f'x{self.scale}', f'X{self.scale}', '']:
                candidate = self.lr_dir / f"{stem}{suffix}{ext}"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"Cannot find LR pair for {hr_path} in {self.lr_dir}"
        )

    def _load(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load HR (and optionally LR) image, using cache if available."""
        if idx in self._hr_cache:
            hr = self._hr_cache[idx]
            lr = self._lr_cache.get(idx)
        else:
            hr = read_img(str(self.hr_paths[idx]))
            lp = self.lr_paths[idx]
            lr = read_img(str(lp)) if lp is not None else None
        return hr, lr

    def _make_lr(self, hr: np.ndarray) -> np.ndarray:
        """Bicubic downsampling to generate LR from HR."""
        h, w = hr.shape[:2]
        lr_h, lr_w = h // self.scale, w // self.scale
        lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        return np.clip(lr, 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> dict:
        hr, lr = self._load(idx)

        # Generate LR on-the-fly if not provided
        if lr is None:
            lr = self._make_lr(hr)

        # Random crop for training
        if self.patch_size > 0:
            hr_h, hr_w = hr.shape[:2]
            lr_ps = self.patch_size // self.scale
            lr_h, lr_w = lr.shape[:2]

            # Random top-left corner in LR space
            x0 = random.randint(0, lr_w - lr_ps)
            y0 = random.randint(0, lr_h - lr_ps)

            lr = lr[y0:y0 + lr_ps, x0:x0 + lr_ps, :]
            hr = hr[y0 * self.scale:(y0 + lr_ps) * self.scale,
                    x0 * self.scale:(x0 + lr_ps) * self.scale, :]

        # Augmentation
        if self.do_augment:
            hr, lr = augment(hr, lr)

        return {
            'lq': img_to_tensor(lr),
            'gt': img_to_tensor(hr),
            'path': str(self.hr_paths[idx]),
        }


class SRSingleDataset(Dataset):
    """Dataset for inference: loads single LR images without ground truth.

    Args:
        lr_dir: Directory containing LR images.
        extensions: Valid image extensions.
    """

    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, lr_dir: str, extensions: Optional[set] = None):
        super().__init__()
        self.lr_dir = Path(lr_dir)
        self.extensions = extensions or self.EXTENSIONS
        self.paths = sorted([
            p for p in self.lr_dir.iterdir()
            if p.suffix.lower() in self.extensions
        ])
        if not self.paths:
            raise RuntimeError(f"No images found in {lr_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = read_img(str(self.paths[idx]))
        return {
            'lq': img_to_tensor(img),
            'path': str(self.paths[idx]),
        }
