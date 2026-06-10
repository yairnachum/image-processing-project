"""Build fine-tune data: distorted OBB tiles per distortion family.

Reproduces the 160-tile train subset (seed=7), splits it into train/val,
distorts each tile at every family level, and writes distorted PNGs + OBB
labels + a per-family `task: obb` data yaml. The 40-tile test split is never
touched.

    python -m scripts.build_finetune_data --distortion all \\
        --raw-root data/raw/DOTA --out-root data/finetune
"""

import argparse
from pathlib import Path

import cv2

from src.config import (
    DATA_ROOT, HAZE_LEVELS, JPEG_LEVELS, NOISE_LEVELS,
    FINETUNE_N_TRAIN,
)
from src.dota_utils import (
    DOTA_CLASSES, load_dota_split, select_subset, train_test_split,
    crop_to_tile, write_yolo_obb_label,
)
from scripts.apply_distortions import apply_one, _format_level

FAMILY_LEVELS = {"haze": HAZE_LEVELS, "jpeg": JPEG_LEVELS, "noise": NOISE_LEVELS}


def reproduce_source_train(raw_root: Path):
    """Reproduce the exact 160-sample train subset used for the clean tiles."""
    train = load_dota_split(raw_root / "train" / "images",
                            raw_root / "train" / "labelTxt-v1.0")
    val = load_dota_split(raw_root / "val" / "images",
                          raw_root / "val" / "labelTxt-v1.0")
    subset = select_subset(train + val, n=200, seed=7)
    source_train, _source_test = train_test_split(subset, n_train=160, seed=7)
    return source_train


def _write_yaml(fam_dir: Path) -> None:
    lines = [
        f"# DOTA-OBB fine-tune data ({fam_dir.name})",
        f"path: {fam_dir.resolve()}",
        "train: train/images",
        "val: val/images",
        "task: obb",
        "",
        "names:",
        *(f"  {i}: {c}" for i, c in enumerate(DOTA_CLASSES)),
    ]
    (fam_dir / "dota_obb.yaml").write_text("\n".join(lines) + "\n")


def _build_family(distortion, levels, samples, n_train, out_root, force, tile_size=1024):
    ft_train, ft_val = train_test_split(samples, n_train=n_train, seed=7)
    fam_dir = out_root / distortion
    for split, items in [("train", ft_train), ("val", ft_val)]:
        img_dir = fam_dir / split / "images"
        lbl_dir = fam_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for s in items:
            tile_bgr, anns = crop_to_tile(s, tile_size=tile_size)
            tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
            for lvl in levels:
                lvl_str = _format_level(distortion, lvl)
                stem = f"{s.name}__L{lvl_str}"
                img_path = img_dir / f"{stem}.png"
                lbl_path = lbl_dir / f"{stem}.txt"
                if img_path.exists() and lbl_path.exists() and not force:
                    continue
                dist_rgb = apply_one(distortion, lvl, tile_rgb, s.name)
                cv2.imwrite(str(img_path), cv2.cvtColor(dist_rgb, cv2.COLOR_RGB2BGR))
                write_yolo_obb_label(anns, lbl_path, tile_size, tile_size)
    _write_yaml(fam_dir)
    print(f"[{distortion}] {len(ft_train)} train x {len(levels)} levels, "
          f"{len(ft_val)} val x {len(levels)} levels")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distortion", choices=["haze", "jpeg", "noise", "all"], default="all")
    p.add_argument("--raw-root", type=Path, default=DATA_ROOT / "raw" / "DOTA")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--levels", nargs="*", default=None,
                   help="Override family levels (for tests).")
    p.add_argument("--n-train", type=int, default=FINETUNE_N_TRAIN)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    samples = reproduce_source_train(args.raw_root)
    families = ["haze", "jpeg", "noise"] if args.distortion == "all" else [args.distortion]
    for fam in families:
        if args.levels is None:
            levels = FAMILY_LEVELS[fam]
        else:
            levels = [int(float(x)) if fam == "jpeg" else float(x) for x in args.levels]
        _build_family(fam, levels, samples, args.n_train, args.out_root, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
