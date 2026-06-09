# scripts/eval_sweep.py
"""Generalised sweep evaluator for the distorted and restored stages.

Per W2 + W7 + W8: one stage per (mode, distortion, level) combination.
Calls pipeline.run_stage to produce per-tile model outputs, then
measure_stage to compute the per-combo metrics, then aggregates 3
sweep CSVs (perclass_detections, edge_metrics, orb_match) for the
selected mode.

Usage:
  python -m scripts.eval_sweep --mode distorted \\
      --clean-root data/clean --distorted-root data/distorted \\
      --manifest results/distortion_manifest.csv \\
      --results-root results --outputs-root outputs

  python -m scripts.eval_sweep --mode restored \\
      --clean-root data/clean --restored-root data/restored \\
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

SWEEP_DIR_NAMES = {
    "distorted": "distortion_sweep",
    "restored":  "restoration_sweep",
}


def parse_only(only):
    if not only:
        return None
    out = []
    for part in only.split(","):
        d, l = part.split(":")
        out.append((d.strip(), l.strip()))
    return out


def all_combos():
    def fmt(d, lvl):
        return f"{float(lvl):.1f}" if d == "haze" else str(int(lvl))
    for d, levels in [
        ("haze",  config.HAZE_LEVELS),
        ("jpeg",  config.JPEG_LEVELS),
        ("noise", config.NOISE_LEVELS),
    ]:
        for lvl in levels:
            yield d, fmt(d, lvl), float(lvl)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["distorted", "restored"], required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--distorted-root", type=Path, default=None)
    parser.add_argument("--restored-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated subset of {detections,edges,orb} for run_stage.")
    args = parser.parse_args(argv)

    if args.mode == "distorted" and args.distorted_root is None:
        parser.error("--distorted-root is required when --mode=distorted")
    if args.mode == "restored" and args.restored_root is None:
        parser.error("--restored-root is required when --mode=restored")

    image_root = args.distorted_root if args.mode == "distorted" else args.restored_root

    manifest = pd.read_csv(args.manifest, dtype={"level": str})
    only = parse_only(args.only)
    combos = [(d, l, n) for d, l, n in all_combos()
              if only is None or (d, l) in only]

    sweep_perclass = []
    sweep_edges = []
    sweep_orb = []

    clean_lbl_dir = args.clean_root / "test" / "labels"
    tasks_list = tuple(t.strip() for t in args.tasks.split(",")) if args.tasks else None

    for distortion, level_str, level_numeric in combos:
        stage = f"{args.mode}/{distortion}/{level_str}"
        image_dir = image_root / distortion / level_str / "test" / "images"
        combo_rows = manifest[
            (manifest["distortion"] == distortion) & (manifest["level"] == level_str)
        ]
        snr_db_mean = float(
            combo_rows["snr_db"].replace([np.inf, -np.inf], np.nan).mean()
        )

        per_combo_results = args.results_root / stage
        should_run = args.force or args.tasks is not None or not (per_combo_results / "detections.csv").exists()
        if should_run:
            run_stage(
                stage=stage,
                image_dir=image_dir,
                results_root=args.results_root,
                outputs_root=args.outputs_root,
                tasks=tasks_list,
            )

        measure_stage(
            stage=stage,
            image_dir=image_dir,
            gt_label_dir=clean_lbl_dir,
            results_root=args.results_root,
            outputs_root=args.outputs_root,
        )

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

        orb_rows = []
        for _, row in combo_rows.iterrows():
            name = row["name"]
            clean_npz = args.outputs_root / "clean" / "orb" / f"{name}.npz"
            stage_npz = args.outputs_root / stage / "orb" / f"{name}.npz"
            if not clean_npz.exists() or not stage_npz.exists():
                continue
            c = np.load(clean_npz, allow_pickle=False)
            s = np.load(stage_npz, allow_pickle=False)
            des_c = c["descriptors"]
            des_s = s["descriptors"]
            n_good = lowe_match(des_c, des_s, ratio=0.7)
            denom = max(1, min(len(des_c), len(des_s)))
            orb_rows.append({
                "name": name,
                "snr_db": row["snr_db"],
                "n_clean_kp": int(len(des_c)),
                "n_stage_kp": int(len(des_s)),
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

    sweep_dir = args.results_root / SWEEP_DIR_NAMES[args.mode]
    sweep_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(sweep_perclass, ignore_index=True).to_csv(sweep_dir / "perclass_detections.csv", index=False)
    pd.concat(sweep_edges, ignore_index=True).to_csv(sweep_dir / "edge_metrics.csv", index=False)
    pd.concat(sweep_orb, ignore_index=True).to_csv(sweep_dir / "orb_match.csv", index=False)
    print(f"wrote sweep CSVs to {sweep_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
