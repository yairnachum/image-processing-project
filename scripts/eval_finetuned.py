# scripts/eval_finetuned.py
"""Week-11 evaluator: run each fine-tuned OBB specialist through the SAME
AABB + VOC-AP pipeline as the W8 distorted / W9 restored stages, so the
specialist mAP is finally comparable to the baseline.

Each distortion family is evaluated ONLY on its 6 matched distorted-test combos
(haze specialist on haze/*, etc.), detections only (fine-tuning never touched
HED/ORB). Results for all 18 combos are aggregated into
results/finetuned_sweep/perclass_detections.csv with the same columns as
results/distortion_sweep/perclass_detections.csv.

Usage:
  python -m scripts.eval_finetuned \\
      --clean-root data/clean --distorted-root data/distorted \\
      --manifest results/distortion_manifest.csv \\
      --results-root results --outputs-root outputs \\
      --weights-dir weights
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.measure import measure_stage
from src.pipeline import run_stage
from scripts.eval_sweep import all_combos, parse_only

FAMILY_WEIGHTS = {
    "haze":  "finetuned_haze.pt",
    "jpeg":  "finetuned_jpeg.pt",
    "noise": "finetuned_noise.pt",
}


def family_weights_map(weights_dir: Path) -> dict:
    """Map each distortion family to its specialist checkpoint path."""
    return {fam: weights_dir / fname for fam, fname in FAMILY_WEIGHTS.items()}


def run_finetuned_eval(
    clean_root: Path,
    distorted_root: Path,
    manifest: Path,
    results_root: Path,
    outputs_root: Path,
    weights_dir: Path,
    only=None,
    force: bool = False,
) -> Path:
    """Run the matched specialist on each combo; write the aggregate sweep CSV.

    Returns the sweep directory (results_root/finetuned_sweep).
    """
    manifest_df = pd.read_csv(manifest, dtype={"level": str})
    wmap = family_weights_map(weights_dir)
    clean_lbl_dir = clean_root / "test" / "labels"
    combos = [(d, l, n) for d, l, n in all_combos()
              if only is None or (d, l) in only]

    sweep_perclass = []
    for distortion, level_str, level_numeric in combos:
        weights = str(wmap[distortion])
        stage = f"finetuned/{distortion}/{level_str}"
        image_dir = distorted_root / distortion / level_str / "test" / "images"
        combo_rows = manifest_df[
            (manifest_df["distortion"] == distortion) & (manifest_df["level"] == level_str)
        ]
        snr_db_mean = float(
            combo_rows["snr_db"].replace([np.inf, -np.inf], np.nan).mean()
        )

        per_combo = results_root / stage
        should_run = force or not (per_combo / "detections.csv").exists()
        if should_run:
            run_stage(
                stage=stage,
                image_dir=image_dir,
                results_root=results_root,
                outputs_root=outputs_root,
                tasks=("detections",),
                weights=weights,
            )

        measure_stage(
            stage=stage,
            image_dir=image_dir,
            gt_label_dir=clean_lbl_dir,
            results_root=results_root,
            outputs_root=outputs_root,
            detections_only=True,
        )

        pc = pd.read_csv(per_combo / "perclass_detections.csv")
        pc.insert(0, "distortion", distortion)
        pc.insert(1, "level", level_str)
        pc.insert(2, "level_numeric", level_numeric)
        pc.insert(3, "snr_db_mean", snr_db_mean)
        sweep_perclass.append(pc)

        print(f"[{stage}] mean SNR {snr_db_mean:.2f} dB, weights={Path(weights).name}",
              file=sys.stderr)

    sweep_dir = results_root / "finetuned_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(sweep_perclass, ignore_index=True).to_csv(
        sweep_dir / "perclass_detections.csv", index=False)
    print(f"wrote sweep CSV to {sweep_dir}", file=sys.stderr)
    return sweep_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--distorted-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=config.FINETUNE_WEIGHTS_DIR)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    run_finetuned_eval(
        clean_root=args.clean_root,
        distorted_root=args.distorted_root,
        manifest=args.manifest,
        results_root=args.results_root,
        outputs_root=args.outputs_root,
        weights_dir=args.weights_dir,
        only=parse_only(args.only),
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
