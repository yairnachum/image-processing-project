"""Orchestrate per-stage model runs and CSV / artifact output."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from src import config
from src.models.detector import Detector
from src.models.edges import HED
from src.models.orb import ORB


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


def run_clean_stage(
    clean_root: Optional[Path] = None,
    results_root: Optional[Path] = None,
    outputs_root: Optional[Path] = None,
) -> None:
    """Run YOLOv8s, HED, and ORB on every tile in clean_root/test/images.

    Writes:
      results_root/clean/{detections,edges,orb}.csv
      outputs_root/clean/{detections,edges,orb}/<name>.{txt,png,npz}
    """
    clean_root   = clean_root   or config.CLEAN_ROOT
    results_root = results_root or config.RESULTS_ROOT
    outputs_root = outputs_root or config.OUTPUTS_ROOT

    images_dir = clean_root / "test" / "images"
    tiles = sorted(images_dir.glob("*.png"))
    if not tiles:
        raise FileNotFoundError(f"No tiles found in {images_dir}")

    detector = Detector()
    hed      = HED()
    orb      = ORB()

    det_rows, edge_rows, orb_rows = [], [], []

    for tile_path in tiles:
        name = tile_path.stem
        img_bgr = cv2.imread(str(tile_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # --- Detection ---
        d = detector.predict(img_rgb)
        det_rows.append({
            "name":      name,
            "n_dets":    int(len(d["boxes_xyxy"])),
            "mean_conf": float(d["scores"].mean()) if len(d["scores"]) else 0.0,
        })
        _save_yolo_txt(
            d["boxes_xyxy"], d["classes"], d["scores"], w, h,
            outputs_root / "clean" / "detections" / f"{name}.txt",
        )

        # --- HED edges ---
        e = hed.predict(img_rgb)
        edge_rows.append({
            "name":             name,
            "edge_pixel_frac":  e["edge_pixel_frac"],
            "mean_response":    e["mean_response"],
        })
        edge_out = outputs_root / "clean" / "edges" / f"{name}.png"
        edge_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(edge_out), e["edge_map"])

        # --- ORB keypoints ---
        o = orb.predict(img_rgb)
        orb_rows.append({
            "name":          name,
            "n_keypoints":   o["n_keypoints"],
            "mean_response": o["mean_response"],
        })
        orb_out = outputs_root / "clean" / "orb" / f"{name}.npz"
        orb_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(orb_out, keypoints=o["keypoints"], descriptors=o["descriptors"])

    csv_dir = results_root / "clean"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(det_rows ).to_csv(csv_dir / "detections.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(csv_dir / "edges.csv",      index=False)
    pd.DataFrame(orb_rows ).to_csv(csv_dir / "orb.csv",        index=False)
