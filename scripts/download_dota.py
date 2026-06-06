#!/usr/bin/env python3
"""
Download DOTA v1.0 images and annotations from the official Google Drive folders.

Source: https://captain-whu.github.io/DOTA/dataset.html (DOTA-v1.0 section).

The dataset is academic-use only. By running this script you agree to the
terms on the DOTA website.

Usage:
    python scripts/download_dota.py                  # default: data/raw/DOTA
    python scripts/download_dota.py --out <dir>
    python scripts/download_dota.py --splits train val
    python scripts/download_dota.py --skip-extract   # download only, no unzip
"""

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import gdown

# Official DOTA-v1.0 Google Drive folder URLs (from the dataset webpage).
# Each folder is a *folder*, not a single file — gdown --folder lists its
# contents and downloads each entry (typically: image part archives plus
# a labelTxt-v1.0 archive).
DOTA_FOLDERS = {
    "train": "https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK",
    "val":   "https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224",
    "test":  "https://drive.google.com/drive/folders/1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl",
}


def _extract(archive: Path, dest: Path) -> None:
    """Extract zip or tar.gz into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)
    else:
        print(f"  Skipping unknown archive type: {archive.name}")


def download_split(split: str, dest: Path, skip_extract: bool) -> None:
    url = DOTA_FOLDERS[split]
    split_dir = dest / split
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading DOTA-v1.0 {split} → {split_dir} ===")
    gdown.download_folder(
        url=url,
        output=str(split_dir),
        quiet=False,
        use_cookies=False,
    )

    if skip_extract:
        return

    print(f"  Extracting archives in {split_dir} ...")
    for archive in sorted(split_dir.iterdir()):
        if archive.is_file() and (archive.suffix == ".zip" or archive.name.endswith((".tar.gz", ".tgz"))):
            print(f"    {archive.name}")
            _extract(archive, split_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/DOTA", help="Output directory (default: data/raw/DOTA)")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"], choices=list(DOTA_FOLDERS),
        help="Which splits to download (default: train val). Add 'test' if you need testing images.",
    )
    parser.add_argument("--skip-extract", action="store_true", help="Download archives only, do not unzip")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        download_split(split, out, args.skip_extract)

    print("\nDone. Expected layout after extraction:")
    print(f"  {out}/")
    print( "    train/images/   *.png")
    print( "    train/labelTxt/ *.txt")
    print( "    val/images/     *.png")
    print( "    val/labelTxt/   *.txt")


if __name__ == "__main__":
    main()
