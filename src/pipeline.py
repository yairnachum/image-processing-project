"""Orchestrate per-stage model runs and CSV / artifact output.

A "stage" is one of {clean, distorted, restored, finetuned}. Each stage
has the same shape: read images, run YOLOv8s + HED + ORB, write summary
CSVs + per-image artifacts under a stage-named subdirectory.
"""

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd

from src import config
from src.models.detector import Detector
from src.models.edges import HED
from src.models.orb import ORB

ALL_TASKS = ("detections", "edges", "orb")


def _save_yolo_txt(boxes_xyxy, classes, scores, img_w, img_h, out_path):
    lines = []
    for (x1, y1, x2, y2), c, s in zip(boxes_xyxy, classes, scores):
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(s):.6f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def run_stage(
    stage: str,
    image_dir: Path,
    results_root: Path,
    outputs_root: Path,
    tasks: Optional[Iterable[str]] = None,
    weights: Optional[str] = None,
) -> None:
    """Run the chosen `tasks` on every `*.png` in `image_dir`.

    Writes:
      results_root/{stage}/{task}.csv               (summary, one row per tile)
      outputs_root/{stage}/{task}/<name>.{txt,png,npz}  (per-tile artifacts)

    `tasks` defaults to ("detections", "edges", "orb"). Models are loaded only
    for the tasks requested, which lets distortion sweeps skip the heavy ones
    when prototyping.

    `weights` overrides the detector checkpoint; None uses config.YOLO_WEIGHTS
    (the baseline). Used by the Week-11 fine-tuned sweep to load each specialist.
    """
    tasks = tuple(tasks) if tasks is not None else ALL_TASKS
    unknown = set(tasks) - set(ALL_TASKS)
    if unknown:
        raise ValueError(f"unknown tasks {sorted(unknown)}; allowed: {ALL_TASKS}")

    tiles = sorted(image_dir.glob("*.png"))
    if not tiles:
        raise FileNotFoundError(f"No tiles found in {image_dir}")

    detector = Detector(weights=weights) if "detections" in tasks else None
    hed = HED() if "edges" in tasks else None
    orb = ORB() if "orb" in tasks else None

    det_rows, edge_rows, orb_rows = [], [], []

    for tile_path in tiles:
        name = tile_path.stem
        img_bgr = cv2.imread(str(tile_path))
        if img_bgr is None:
            raise RuntimeError(f"failed to read {tile_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        if detector is not None:
            d = detector.predict(img_rgb)
            det_rows.append({
                "name":      name,
                "n_dets":    int(len(d["boxes_xyxy"])),
                "mean_conf": float(d["scores"].mean()) if len(d["scores"]) else 0.0,
            })
            _save_yolo_txt(
                d["boxes_xyxy"], d["classes"], d["scores"], w, h,
                outputs_root / stage / "detections" / f"{name}.txt",
            )

        if hed is not None:
            e = hed.predict(img_rgb)
            edge_rows.append({
                "name":             name,
                "edge_pixel_frac":  e["edge_pixel_frac"],
                "mean_response":    e["mean_response"],
            })
            edge_out = outputs_root / stage / "edges" / f"{name}.png"
            edge_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(edge_out), e["edge_map"])

        if orb is not None:
            o = orb.predict(img_rgb)
            orb_rows.append({
                "name":          name,
                "n_keypoints":   o["n_keypoints"],
                "mean_response": o["mean_response"],
            })
            orb_out = outputs_root / stage / "orb" / f"{name}.npz"
            orb_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(orb_out, keypoints=o["keypoints"], descriptors=o["descriptors"])

    csv_dir = results_root / stage
    csv_dir.mkdir(parents=True, exist_ok=True)
    if detector is not None:
        pd.DataFrame(det_rows).to_csv(csv_dir / "detections.csv", index=False)
    if hed is not None:
        pd.DataFrame(edge_rows).to_csv(csv_dir / "edges.csv", index=False)
    if orb is not None:
        pd.DataFrame(orb_rows).to_csv(csv_dir / "orb.csv", index=False)


def run_clean_stage(
    clean_root: Optional[Path] = None,
    results_root: Optional[Path] = None,
    outputs_root: Optional[Path] = None,
) -> None:
    """Thin wrapper around `run_stage` for the clean baseline.

    Reads tiles from `clean_root/test/images` and writes under stage="clean".
    Preserved as a stable entry point for the eval CLI.
    """
    clean_root = clean_root or config.CLEAN_ROOT
    results_root = results_root or config.RESULTS_ROOT
    outputs_root = outputs_root or config.OUTPUTS_ROOT
    run_stage(
        stage="clean",
        image_dir=clean_root / "test" / "images",
        results_root=results_root,
        outputs_root=outputs_root,
    )
