#!/usr/bin/env python3
"""
Download DOTA v1.0 images and annotations.

DOTA requires a one-time free registration at:
    https://captain-whu.github.io/DOTA/dataset.html

After registering you receive Google Drive download links.
Paste them below (or use the known public IDs) and run this script.

Usage:
    python scripts/download_dota.py --out data/raw/DOTA
"""

import argparse
import zipfile
from pathlib import Path

import gdown

# Known public Google Drive file IDs for DOTA v1.0 (part archives).
# If these links are expired, download manually from the DOTA website
# and place the zip files inside --out before running with --skip-download.
DOTA_FILES = {
    # images
    "part1.zip":        "1UdluMXnx4VFqfFtMY7vToYMYQaJW0O6p",
    "part2.zip":        "1KmACrLsABHWO7bBIqgpqLwT9K0rBMW_s",
    "part3.zip":        "1OSS3y2CtCgNH3F_BuFsVXSTQ6mV_XVWF",
    "val_images.zip":   "1FYKVhDl3Xm2YK9CeX5n3i4dFDi0LoD3k",
    # annotations
    "train_labels.zip": "1HE6aSGBXIIiAv0Mxqt41DI_2FqGxCK8V",
    "val_labels.zip":   "1OGCbTm5zNSIKJBBhOeMcAiOZWnxGbHYI",
}


def download_and_extract(file_id: str, dest_zip: Path, extract_to: Path) -> None:
    if not dest_zip.exists():
        print(f"  Downloading {dest_zip.name} ...")
        gdown.download(id=file_id, output=str(dest_zip), quiet=False)
    else:
        print(f"  {dest_zip.name} already exists, skipping download.")

    print(f"  Extracting {dest_zip.name} ...")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(extract_to)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/DOTA", help="Output directory")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download; only extract zips already present in --out",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for fname, file_id in DOTA_FILES.items():
        dest_zip = out / fname
        if args.skip_download:
            if dest_zip.exists():
                print(f"Extracting {fname} ...")
                with zipfile.ZipFile(dest_zip, "r") as zf:
                    zf.extractall(out)
            else:
                print(f"  WARNING: {dest_zip} not found, skipping.")
        else:
            try:
                download_and_extract(file_id, dest_zip, out)
            except Exception as e:
                print(f"  ERROR downloading {fname}: {e}")
                print("  Download manually from https://captain-whu.github.io/DOTA/dataset.html")
                print(f"  and place the zip at {dest_zip}")

    print("\nDone. Expected layout:")
    print("  data/raw/DOTA/")
    print("    train/images/   *.png")
    print("    train/labelTxt/ *.txt")
    print("    val/images/     *.png")
    print("    val/labelTxt/   *.txt")


if __name__ == "__main__":
    main()
