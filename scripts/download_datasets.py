"""
Download datasets for LUMEN training and testing.

Usage:
    python scripts/download_datasets.py
"""

from pathlib import Path
import shutil
import zipfile
import requests
from tqdm import tqdm

BASE = "https://utsacloud-my.sharepoint.com/:u:/g/personal/mohammadsadegh_sirjani_utsa_edu/"

DATASETS = {
    "DIV2K":      ("IQCiMggqDoXfQJsfHbDwETN5AbvoiLhOIzRJ7voldBh970E?e=2kNBJA", "*.png"),
    "Benchmarks": ("IQCF-4NR-_hRSIU0MuZmaPyZAdzcKLbsRxGXT92KGV5UnD8?e=c9oHYN", "*.png"),
}



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
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(target_dir))
    else:
        for item in temp_dir.iterdir():
            if item.name.startswith("_") or item.name == "__MACOSX":
                continue
            dest = target_dir / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(target_dir))

    shutil.rmtree(temp_dir, ignore_errors=True)
    macosx = target_dir / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx, ignore_errors=True)


def cleanup(dataset_dir: Path):
    """Remove X3 scale folders and LR_unknown (not needed for x2/x4 training)."""
    for x3 in dataset_dir.rglob("X3"):
        if x3.is_dir():
            shutil.rmtree(x3)
    for lr_unknown in dataset_dir.rglob("LR_unknown"):
        if lr_unknown.is_dir():
            shutil.rmtree(lr_unknown)


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
        cleanup(d)
        z.unlink()
        return

    print(f"Downloading {name}...")
    if download_file(uid, z):
        extract_and_flatten(z, d, name)
        cleanup(d)
        z.unlink()
        print(f"  [SUCCESS] {name} downloaded successfully")
    else:
        print(f"  [FAILED] Failed to download {name}")


def main():
    data = Path(__file__).parent.parent / "datasets"
    datasets = list(DATASETS.keys())

    print(f"Datasets: {', '.join(datasets)}\n")
    for name in datasets:
        download_dataset(data, name)

    print("\nDone!")


if __name__ == "__main__":
    main()
