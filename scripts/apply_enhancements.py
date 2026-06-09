"""Apply the matched classical enhancement to every distorted (d, l) combo
and write restored PNGs + a recovery manifest.

Per W2 spec, the matched enhancements are:
  haze  → Dark Channel Prior         (src.enhancement.dehaze_dcp)
  jpeg  → bilateral filter on Y      (src.enhancement.bilateral_y)
  noise → NL-Means + bilateral pass  (src.enhancement.nlmeans_bilateral)

Layout:
  out_root/{distortion}/{level}/test/images/<name>.png

Manifest columns:
  name, distortion, level, level_numeric,
  snr_db_distorted, snr_db_restored, snr_gain_db,
  clean_path, distorted_path, restored_path.

`snr_gain_db = restored - distorted`; positive means recovery.
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

from src import config
from src.distortions import snr_db
from src.enhancement import bilateral_y, dehaze_dcp, nlmeans_bilateral

ENHANCEMENTS = {
    "haze":  dehaze_dcp,
    "jpeg":  bilateral_y,
    "noise": nlmeans_bilateral,
}


def _format_level(distortion: str, lvl) -> str:
    return f"{float(lvl):.1f}" if distortion == "haze" else str(int(lvl))


def all_combos():
    for d, levels in [
        ("haze",  config.HAZE_LEVELS),
        ("jpeg",  config.JPEG_LEVELS),
        ("noise", config.NOISE_LEVELS),
    ]:
        for lvl in levels:
            yield d, _format_level(d, lvl), float(lvl)


def parse_only(only):
    if not only:
        return None
    out = []
    for part in only.split(","):
        d, l = part.split(":")
        out.append((d.strip(), l.strip()))
    return out


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--distorted-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated combos to run, e.g. 'haze:0.5,jpeg:10'. Default: all 18.")
    parser.add_argument("--force", action="store_true",
                        help="Re-write restored PNGs even if they exist.")
    args = parser.parse_args()

    only = parse_only(args.only)
    combos = [(d, l, n) for d, l, n in all_combos()
              if only is None or (d, l) in only]

    rows = []
    n_written, n_skipped = 0, 0

    for distortion, level_str, level_numeric in combos:
        enhance = ENHANCEMENTS[distortion]
        dist_dir = args.distorted_root / distortion / level_str / "test" / "images"
        clean_dir = args.clean_root / "test" / "images"
        out_dir = args.out_root / distortion / level_str / "test" / "images"

        for dist_path in sorted(dist_dir.glob("*.png")):
            name = dist_path.stem
            clean_path = clean_dir / f"{name}.png"
            out_path = out_dir / f"{name}.png"

            clean_rgb = _imread_rgb(clean_path)
            dist_rgb = _imread_rgb(dist_path)

            if out_path.exists() and not args.force:
                restored_rgb = _imread_rgb(out_path)
                n_skipped += 1
            else:
                restored_rgb = enhance(dist_rgb)
                _imwrite_rgb(out_path, restored_rgb)
                n_written += 1

            snr_d = snr_db(clean_rgb, dist_rgb)
            snr_r = snr_db(clean_rgb, restored_rgb)
            rows.append({
                "name": name,
                "distortion": distortion,
                "level": level_str,
                "level_numeric": level_numeric,
                "snr_db_distorted": snr_d,
                "snr_db_restored": snr_r,
                "snr_gain_db": snr_r - snr_d,
                "clean_path": str(clean_path),
                "distorted_path": str(dist_path),
                "restored_path": str(out_path),
            })

        print(f"[{distortion}/{level_str}] done", file=sys.stderr)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).astype({"level": str})
    df.to_csv(args.manifest, index=False)
    print(f"wrote {n_written} restored tiles, reused {n_skipped}; manifest at {args.manifest}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
