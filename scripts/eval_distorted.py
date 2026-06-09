"""Run YOLOv8s + HED + ORB on every (distortion, level) combo and aggregate sweep CSVs.

Outputs per combo:
  results/distorted/{d}/{l}/{detections,edges,orb,perclass_detections,edge_metrics,perclass_edges}.csv
  outputs/distorted/{d}/{l}/{detections,edges,orb}/<name>.{txt,png,npz}

Sweep aggregates:
  results/distortion_sweep/perclass_detections.csv
  results/distortion_sweep/edge_metrics.csv
  results/distortion_sweep/orb_match.csv

Usage:
  python -m scripts.eval_distorted \\
      --clean-root data/clean \\
      --distorted-root data/distorted \\
      --manifest results/distortion_manifest.csv \\
      --results-root results --outputs-root outputs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.measure import measure_stage
from src.orb_match import lowe_match
from src.pipeline import run_stage


def parse_only(only):
    """Parse `--only haze:0.5,jpeg:10` → [("haze","0.5"),("jpeg","10")] or None."""
    if not only:
        return None
    out = []
    for part in only.split(","):
        d, l = part.split(":")
        out.append((d.strip(), l.strip()))
    return out


def all_combos():
    """Yield (distortion, level_str, level_numeric) for the full sweep."""
    def fmt(d, lvl):
        return f"{float(lvl):.1f}" if d == "haze" else str(int(lvl))
    for d, levels in [
        ("haze",  config.HAZE_LEVELS),
        ("jpeg",  config.JPEG_LEVELS),
        ("noise", config.NOISE_LEVELS),
    ]:
        for lvl in levels:
            yield d, fmt(d, lvl), float(lvl)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--distorted-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated combos to run, e.g. 'haze:0.5,jpeg:10'. Default: all 18.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run models even if per-combo run_stage outputs already exist.")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"level": str})
    only = parse_only(args.only)
    combos = [(d, l, n) for d, l, n in all_combos()
              if only is None or (d, l) in only]

    sweep_perclass = []
    sweep_edges = []
    sweep_orb = []

    clean_lbl_dir = args.clean_root / "test" / "labels"

    for distortion, level_str, level_numeric in combos:
        stage = f"distorted/{distortion}/{level_str}"
        image_dir = args.distorted_root / distortion / level_str / "test" / "images"
        combo_rows = manifest[
            (manifest["distortion"] == distortion) & (manifest["level"] == level_str)
        ]
        snr_db_mean = float(
            combo_rows["snr_db"].replace([np.inf, -np.inf], np.nan).mean()
        )

        # 1. Run models (idempotent: skip if results CSV already there)
        per_combo_results = args.results_root / stage
        if args.force or not (per_combo_results / "detections.csv").exists():
            run_stage(
                stage=stage,
                image_dir=image_dir,
                results_root=args.results_root,
                outputs_root=args.outputs_root,
            )

        # 2. Measure (always re-runs — cheap)
        measure_stage(
            stage=stage,
            image_dir=image_dir,
            gt_label_dir=clean_lbl_dir,
            results_root=args.results_root,
            outputs_root=args.outputs_root,
        )

        # 3. Read per-combo CSVs and prepend metadata for sweep aggregates
        pc = pd.read_csv(per_combo_results / "perclass_detections.csv")
        pc.insert(0, "distortion", distortion)
        pc.insert(1, "level", level_str)
        pc.insert(2, "level_numeric", level_numeric)
        pc.insert(3, "snr_db_mean", snr_db_mean)
        sweep_perclass.append(pc)

        em = pd.read_csv(per_combo_results / "edge_metrics.csv")
        em = em.merge(combo_rows[["name", "snr_db"]], on="name", how="left")
        em.insert(0, "distortion", distortion)
        em.insert(1, "level", level_str)
        em.insert(2, "level_numeric", level_numeric)
        em.insert(3, "snr_db_mean", snr_db_mean)
        sweep_edges.append(em)

        # 4. ORB matches (this stage produces real numbers for the first time)
        orb_rows = []
        for _, row in combo_rows.iterrows():
            name = row["name"]
            clean_npz = args.outputs_root / "clean" / "orb" / f"{name}.npz"
            dist_npz = args.outputs_root / stage / "orb" / f"{name}.npz"
            if not clean_npz.exists() or not dist_npz.exists():
                continue
            c = np.load(clean_npz, allow_pickle=False)
            d = np.load(dist_npz, allow_pickle=False)
            des_c = c["descriptors"]
            des_d = d["descriptors"]
            n_good = lowe_match(des_c, des_d, ratio=0.7)
            denom = max(1, min(len(des_c), len(des_d)))
            orb_rows.append({
                "name": name,
                "snr_db": row["snr_db"],
                "n_clean_kp": int(len(des_c)),
                "n_dist_kp": int(len(des_d)),
                "n_good": n_good,
                "good_ratio": n_good / denom,
            })
        om = pd.DataFrame(orb_rows)
        om.insert(0, "distortion", distortion)
        om.insert(1, "level", level_str)
        om.insert(2, "level_numeric", level_numeric)
        om.insert(3, "snr_db_mean", snr_db_mean)
        sweep_orb.append(om)

        print(f"[{stage}] mean SNR {snr_db_mean:.2f} dB, {len(om)} tiles measured",
              file=sys.stderr)

    sweep_dir = args.results_root / "distortion_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(sweep_perclass, ignore_index=True).to_csv(
        sweep_dir / "perclass_detections.csv", index=False)
    pd.concat(sweep_edges, ignore_index=True).to_csv(
        sweep_dir / "edge_metrics.csv", index=False)
    pd.concat(sweep_orb, ignore_index=True).to_csv(
        sweep_dir / "orb_match.csv", index=False)

    print(f"wrote sweep CSVs to {sweep_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
