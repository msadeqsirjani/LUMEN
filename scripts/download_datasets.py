"""
Download datasets for LUMEN training and testing.

Usage:
    python scripts/download_datasets.py --task sr        # ClassicSR / Lightweight SR
    python scripts/download_datasets.py --task car       # JPEG Compression Artifact Reduction
    python scripts/download_datasets.py --task colordn   # Gaussian Color Image Denoising
    python scripts/download_datasets.py --task all       # Download all datasets
"""

import argparse
from pathlib import Path
import shutil
import zipfile
import requests
from tqdm import tqdm

BASE = "https://utsacloud-my.sharepoint.com/:u:/g/personal/mohammadsadegh_sirjani_utsa_edu/"

DATASETS = {
    # ClassicSR & LightweightSR datasets
    "DIV2K": ("IQCiMggqDoXfQJsfHbDwETN5AbvoiLhOIzRJ7voldBh970E?e=2kNBJA", "*.png"),
    "DF2K": ("IQDlarek0SAGT4lm3_pGiBq6AQXmoRdu1a-K1ZOmP4m1tvk?e=BaMTKE", "*.png"),
    "Benchmarks": ("IQCF-4NR-_hRSIU0MuZmaPyZAdzcKLbsRxGXT92KGV5UnD8?e=c9oHYN", "*.png"),
    # JPEG Compression Artifact Reduction datasets
    "DFWB": ("IQBAPTgoa88DSI8d-hpaXO5PAb8Xw4mxR59DRfTPDvrSnaQ?e=E5P8aQ", "*"),
    "CAR": ("IQAyjaMRXKyNSJ4-9HR82hWtAbbm72aypAIWaMxYiYShJGs?e=7HQGTG", "*"),
    # Gaussian Color Image Denoising datasets
    "DFWB_RGB": ("IQD22rG80pBKTp-768mbpIf4AaMbYvMmmtIlaaKeB9WFrVE?e=snZYyL", "*"),
    "ColorDN": ("IQA7YpJ6xkXIT7DhvGi7Jcm3AbFmCfy4_SG4qPXhxbE2sEk?e=ZBdZMt", "*"),
}

TASKS = {
    "sr": ["DF2K", "DIV2K", "Benchmarks"],
    "car": ["DFWB", "CAR"],
    "colordn": ["DFWB_RGB", "ColorDN"],
}

# Folder renames after extraction: {dataset: {old_name: new_name}}
RENAMES = {
    "ColorDN": {
        "CBSD68HQ": "CBSD68",
        "Kodak24HQ": "Kodak24",
        "McMasterHQ": "McMaster",
        "Urban100HQ": "Urban100",
    },
}

# Datasets that need restructuring to match DIV2K format (with Train subfolder)
RESTRUCTURE = {"DF2K"}


def download_file(uid: str, path: Path) -> bool:
    """Download a file from OneDrive."""
    url = BASE + uid
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url + '&download=1', headers={'User-Agent': 'Mozilla/5.0'}, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return False

        total = int(r.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=path.name) as pbar:
            for chunk in r.iter_content(32768):
                f.write(chunk)
                pbar.update(len(chunk))

        zipfile.ZipFile(path).testzip()
        return True
    except Exception:
        if path.exists():
            path.unlink()
        return False


def extract_and_flatten(zip_path: Path, target_dir: Path, name: str):
    """Extract zip and flatten if nested folder exists."""
    temp_dir = target_dir / "_temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)

    zipfile.ZipFile(zip_path).extractall(temp_dir)

    nested = temp_dir / name
    if nested.exists() and nested.is_dir():
        for item in nested.iterdir():
            dest = target_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(target_dir))
    else:
        for item in temp_dir.iterdir():
            if item.name.startswith("_") or item.name == "__MACOSX":
                continue
            dest = target_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(target_dir))

    shutil.rmtree(temp_dir, ignore_errors=True)
    macosx = target_dir / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx, ignore_errors=True)


def apply_renames(target_dir: Path, name: str):
    """Rename folders after extraction if needed."""
    if name not in RENAMES:
        return
    for old_name, new_name in RENAMES[name].items():
        old_path = target_dir / old_name
        new_path = target_dir / new_name
        if old_path.exists() and not new_path.exists():
            shutil.move(str(old_path), str(new_path))


def restructure_dataset(target_dir: Path, name: str):
    """Restructure dataset to match DIV2K format (with Train subfolder)."""
    if name not in RESTRUCTURE:
        return

    # Restructure HR: HR/*.png -> HR/Train/*.png
    hr_dir = target_dir / "HR"
    if hr_dir.exists() and not (hr_dir / "Train").exists():
        train_dir = hr_dir / "Train"
        train_dir.mkdir(exist_ok=True)
        for img in hr_dir.glob("*.png"):
            shutil.move(str(img), str(train_dir / img.name))
        print(f"  Restructured {name}/HR -> HR/Train/")

    # Restructure LR_bicubic: LR_bicubic/X* -> LR_bicubic/Train/X*
    lr_dir = target_dir / "LR_bicubic"
    if lr_dir.exists() and not (lr_dir / "Train").exists():
        train_dir = lr_dir / "Train"
        train_dir.mkdir(exist_ok=True)
        for scale_dir in ["X2", "X3", "X4"]:
            src = lr_dir / scale_dir
            if src.exists():
                shutil.move(str(src), str(train_dir / scale_dir))
        print(f"  Restructured {name}/LR_bicubic -> LR_bicubic/Train/")


def download_dataset(data_dir: Path, name: str):
    """Download and extract a single dataset."""
    uid, check_ext = DATASETS[name]
    d = data_dir / name

    if d.exists() and list(d.rglob(check_ext)):
        print(f"{name} already exists, skipping...")
        return

    d.mkdir(parents=True, exist_ok=True)
    z = d / f"{name}.zip"

    if z.exists():
        print(f"Extracting {name}...")
        extract_and_flatten(z, d, name)
        apply_renames(d, name)
        restructure_dataset(d, name)
        z.unlink()
        return

    print(f"Downloading {name}...")
    if download_file(uid, z):
        extract_and_flatten(z, d, name)
        apply_renames(d, name)
        restructure_dataset(d, name)
        z.unlink()
        print(f"  [SUCCESS] {name} downloaded successfully")
    else:
        print(f"  [FAILED] Failed to download {name}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for LUMEN")
    parser.add_argument(
        "--task",
        type=str,
        choices=["sr", "car", "colordn", "all"],
        default="all",
        help="Task type: sr (ClassicSR/Lightweight), car (JPEG CAR), colordn (Color Denoising), all"
    )
    args = parser.parse_args()

    data = Path(__file__).parent.parent / "datasets"

    if args.task == "all":
        datasets = [name for names in TASKS.values() for name in names]
    else:
        datasets = TASKS[args.task]

    print(f"Downloading datasets for task: {args.task}")
    print(f"Datasets: {', '.join(datasets)}\n")

    for name in datasets:
        download_dataset(data, name)

    print("\nDone!")


if __name__ == "__main__":
    main()
