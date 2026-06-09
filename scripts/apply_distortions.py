"""Materialize the distortion sweep on the clean test split.

Layout:
  out_root/{distortion}/{level}/test/images/<name>.png

Writes:
  --manifest path → CSV with one row per (image, distortion, level) and SNR.

Usage:
  python -m scripts.apply_distortions \\
      --clean-root data/clean \\
      --out-root   data/distorted \\
      --manifest   results/distortion_manifest.csv
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

from src.distortions import (
    apply_haze,
    apply_jpeg,
    apply_noise,
    seed_for_tile,
    snr_db,
)
from src.config import HAZE_LEVELS, JPEG_LEVELS, NOISE_LEVELS


def _format_level(distortion: str, lvl) -> str:
    """Folder-name format for a level: haze keeps one decimal, jpeg/noise int."""
    if distortion == "haze":
        return f"{float(lvl):.1f}"
    return str(int(lvl))


def _imread_rgb(path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _imwrite_rgb(path: Path, rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"failed to write {path}")


def apply_one(distortion: str, level, img_rgb, name: str):
    if distortion == "haze":
        return apply_haze(img_rgb, beta=float(level))
    if distortion == "jpeg":
        return apply_jpeg(img_rgb, q=int(level))
    if distortion == "noise":
        # Same seed regardless of sigma so noise patterns are "anchored" to the
        # tile, not the sweep level — better for before/after visualization.
        return apply_noise(img_rgb, sigma_g=float(level), seed=seed_for_tile(name))
    raise ValueError(f"unknown distortion {distortion!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--out-root",   type=Path, required=True)
    parser.add_argument("--manifest",   type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="Re-write distorted PNGs even if they exist.")
    args = parser.parse_args()

    # Path convention: clean_path and distorted_path in the manifest are
    # written as the *string form* of whatever Path the CLI received. If the
    # caller passes `--clean-root data/clean` (relative), manifest paths are
    # relative-to-cwd; if they pass an absolute path, manifest paths are
    # absolute. Downstream W8+ readers should either run from the repo root,
    # or wrap with Path(repo_root) / row["clean_path"].

    clean_imgs = sorted((args.clean_root / "test" / "images").glob("*.png"))
    if not clean_imgs:
        print(f"No tiles found under {args.clean_root}/test/images", file=sys.stderr)
        return 1

    sweep = [
        ("haze",  HAZE_LEVELS),
        ("jpeg",  JPEG_LEVELS),
        ("noise", NOISE_LEVELS),
    ]

    rows = []
    n_written, n_skipped = 0, 0
    for clean_path in clean_imgs:
        name = clean_path.stem
        clean_rgb = _imread_rgb(clean_path)
        for distortion, levels in sweep:
            for lvl in levels:
                lvl_str = _format_level(distortion, lvl)
                out_path = args.out_root / distortion / lvl_str / "test" / "images" / f"{name}.png"
                if out_path.exists() and not args.force:
                    distorted_rgb = _imread_rgb(out_path)
                    n_skipped += 1
                else:
                    distorted_rgb = apply_one(distortion, lvl, clean_rgb, name)
                    _imwrite_rgb(out_path, distorted_rgb)
                    n_written += 1
                rows.append({
                    "name": name,
                    "distortion": distortion,
                    "level": lvl_str,
                    "level_numeric": float(lvl),
                    "snr_db": snr_db(clean_rgb, distorted_rgb),
                    "clean_path": str(clean_path),
                    "distorted_path": str(out_path),
                })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.manifest, index=False)
    print(f"wrote {n_written} new tiles, reused {n_skipped} existing; manifest at {args.manifest}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
