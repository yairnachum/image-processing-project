"""Read clean-stage predictions + GT, compute per-class detection + edge metrics."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from src import config
from src.metrics import (
    ap_per_class,
    f_score_with_tolerance,
    ods_per_image,
    render_class_edge_map,
)


def _read_yolo_labels(path: Path, with_conf: bool) -> tuple:
    """Parse a YOLO label file → (boxes_norm (N,4), classes (N,), confs (N,) or None)."""
    if not path.exists() or path.stat().st_size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32) if with_conf else None,
        )
    rows = [r.split() for r in path.read_text().splitlines() if r.strip()]
    classes = np.array([int(r[0]) for r in rows], dtype=np.int32)
    boxes = np.array([[float(x) for x in r[1:5]] for r in rows], dtype=np.float32)
    confs = (
        np.array([float(r[5]) for r in rows], dtype=np.float32) if with_conf else None
    )
    return boxes, classes, confs


def _norm_to_xyxy(box_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = box_norm
    return np.array(
        [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
        dtype=np.float32,
    )


def measure_clean_stage(
    clean_root: Optional[Path] = None,
    results_root: Optional[Path] = None,
    outputs_root: Optional[Path] = None,
) -> None:
    """Compute Week 6 metrics on the existing Week 5 clean-stage outputs.

    Reads:
      clean_root/test/{images,labels}/<name>.{png,txt}
      outputs_root/clean/detections/<name>.txt   (cls cx cy w h conf, YOLO-normalised)
      outputs_root/clean/edges/<name>.png        (uint8 HED saliency)

    Writes:
      results_root/clean/perclass_detections.csv
      results_root/clean/edge_metrics.csv
      results_root/clean/perclass_edges.csv
    """
    clean_root = clean_root or config.CLEAN_ROOT
    results_root = results_root or config.RESULTS_ROOT
    outputs_root = outputs_root or config.OUTPUTS_ROOT

    img_dir = clean_root / "test" / "images"
    lbl_dir = clean_root / "test" / "labels"
    det_dir = outputs_root / "clean" / "detections"
    edge_dir = outputs_root / "clean" / "edges"
    out_dir = results_root / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = sorted(img_dir.glob("*.png"))
    if not tiles:
        raise FileNotFoundError(f"No tiles found in {img_dir}")

    # --- Detection: per-class mAP@0.5 ---
    preds, gt = [], []
    for tile in tiles:
        name = tile.stem
        img = cv2.imread(str(tile))
        if img is None:
            raise RuntimeError(f"failed to read {tile}")
        h, w = img.shape[:2]
        gt_boxes, gt_classes, _ = _read_yolo_labels(lbl_dir / f"{name}.txt", with_conf=False)
        for box, c in zip(gt_boxes, gt_classes):
            gt.append({"image_id": name, "class_id": int(c), "xyxy": _norm_to_xyxy(box, w, h)})
        pred_boxes, pred_classes, pred_confs = _read_yolo_labels(
            det_dir / f"{name}.txt", with_conf=True
        )
        for box, c, s in zip(pred_boxes, pred_classes, pred_confs):
            preds.append(
                {
                    "image_id": name,
                    "class_id": int(c),
                    "score": float(s),
                    "xyxy": _norm_to_xyxy(box, w, h),
                }
            )

    n_classes = len(config.DOTA_CLASSES)
    det_ap = ap_per_class(preds, gt, num_classes=n_classes, iou_thresh=0.5)
    rows = []
    for c in range(n_classes):
        rows.append(
            {
                "class_id": c,
                "class_name": config.DOTA_CLASSES[c],
                "n_gt": det_ap[c]["n_gt"],
                "n_pred": det_ap[c]["n_pred"],
                "ap_iou50": det_ap[c]["ap"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "perclass_detections.csv", index=False)

    # --- Edges: per-image ODS + per-class F-score at dataset-wide best threshold ---
    edge_rows = []
    per_image_pred = {}
    per_image_gt_classes = {}
    per_image_size = {}
    for tile in tiles:
        name = tile.stem
        img = cv2.imread(str(tile))
        if img is None:
            raise RuntimeError(f"failed to read {tile}")
        h, w = img.shape[:2]
        edge_pred = cv2.imread(str(edge_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if edge_pred is None:
            raise FileNotFoundError(f"missing edge map for {name}")
        gt_boxes, gt_classes, _ = _read_yolo_labels(lbl_dir / f"{name}.txt", with_conf=False)
        # Aggregate GT edges across all classes for the per-image ODS.
        gt_all = np.zeros((h, w), dtype=np.uint8)
        for c in np.unique(gt_classes):
            gt_all = np.maximum(
                gt_all, render_class_edge_map(gt_boxes, gt_classes, (h, w), int(c), dilate_px=2)
            )
        gt_all_bin = gt_all > 0
        best_t, best_f = ods_per_image(edge_pred, gt_all_bin)
        edge_rows.append({"name": name, "ods_threshold": best_t, "ods_f_score": best_f})
        per_image_pred[name] = edge_pred
        per_image_gt_classes[name] = (gt_boxes, gt_classes)
        per_image_size[name] = (h, w)
    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(out_dir / "edge_metrics.csv", index=False)

    # Dataset-wide threshold proxy: mean of per-image ODS thresholds.
    # NOTE: True BSDS500 ODS-T is a single threshold that maximizes mean F across
    # the dataset; "mean of per-image bests" is a correlated but not identical
    # proxy. We use it because it's cheap and adequate for the project's
    # relative-comparison purpose (clean vs distorted vs restored).
    global_t = int(np.round(edge_df["ods_threshold"].mean())) if len(edge_df) else 128

    # --- Per-class edge F-score at global threshold ---
    perclass_rows = []
    for c in range(n_classes):
        f_scores = []
        n_gt_px_total = 0
        for name, (boxes, classes) in per_image_gt_classes.items():
            if not (classes == c).any():
                continue
            h, w = per_image_size[name]
            gt_c = render_class_edge_map(boxes, classes, (h, w), c, dilate_px=2)
            gt_c_bin = gt_c > 0
            if not gt_c_bin.any():
                continue
            n_gt_px_total += int(gt_c_bin.sum())
            pred_bin = per_image_pred[name] > global_t
            f_scores.append(f_score_with_tolerance(pred_bin, gt_c_bin, tolerance_px=2))
        perclass_rows.append(
            {
                "class_id": c,
                "class_name": config.DOTA_CLASSES[c],
                "n_gt_px": n_gt_px_total,
                "f_score": float(np.mean(f_scores)) if f_scores else float("nan"),
            }
        )
    pd.DataFrame(perclass_rows).to_csv(out_dir / "perclass_edges.csv", index=False)
