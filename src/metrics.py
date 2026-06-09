"""Detection + edge metrics primitives.

All boxes use xyxy in pixel coordinates unless otherwise noted.
"""

from typing import Dict, List

import cv2
import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two axis-aligned boxes given as (x1, y1, x2, y2)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _voc2010_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC-2010 all-points AP: prepend (0, 1), append (1, 0), max-envelope
    precision, then integrate as a sum of rectangles over recall.

    This is the same definition COCO uses (modulo the IoU sweep). It gives
    the intuitive 0.5 when half the GT is recalled at full precision, and
    correctly drops AP when a high-scoring false positive precedes a TP.
    """
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Right-to-left max envelope of precision.
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_per_class(
    preds: List[dict],
    gt: List[dict],
    num_classes: int,
    iou_thresh: float = 0.5,
) -> Dict[int, dict]:
    """Per-class AP@iou_thresh using VOC-2010 all-points interpolation.

    `preds`: each dict has keys image_id, class_id, score, xyxy.
    `gt`:    each dict has keys image_id, class_id, xyxy.

    Returns: dict mapping class_id -> {ap (float|nan), n_gt (int), n_pred (int)}.
    For classes with no GT and no predictions, ap is NaN.
    """
    out: Dict[int, dict] = {}
    for c in range(num_classes):
        gt_c = [g for g in gt if g["class_id"] == c]
        preds_c = sorted(
            [p for p in preds if p["class_id"] == c],
            key=lambda p: -p["score"],
        )
        n_gt = len(gt_c)
        n_pred = len(preds_c)

        if n_gt == 0 and n_pred == 0:
            out[c] = {"ap": float("nan"), "n_gt": 0, "n_pred": 0}
            continue
        if n_gt == 0:
            # All predictions are false positives; AP = 0.
            out[c] = {"ap": 0.0, "n_gt": 0, "n_pred": n_pred}
            continue

        # Index GTs by image_id for fast matching.
        gt_by_image: Dict[str, List[dict]] = {}
        for g in gt_c:
            gt_by_image.setdefault(g["image_id"], []).append(dict(g, matched=False))

        tp = np.zeros(max(n_pred, 1), dtype=np.float64)
        fp = np.zeros(max(n_pred, 1), dtype=np.float64)
        for i, p in enumerate(preds_c):
            candidates = gt_by_image.get(p["image_id"], [])
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(candidates):
                if g["matched"]:
                    continue
                iou = iou_xyxy(p["xyxy"], g["xyxy"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                candidates[best_j]["matched"] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        if n_pred == 0:
            ap = 0.0
        else:
            tp_cum = np.cumsum(tp[:n_pred])
            fp_cum = np.cumsum(fp[:n_pred])
            recall = tp_cum / n_gt
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap = _voc2010_ap(recall, precision)

        out[c] = {"ap": float(ap), "n_gt": int(n_gt), "n_pred": int(n_pred)}
    return out


def _yolo_to_xyxy(box_norm: np.ndarray, img_w: int, img_h: int) -> tuple:
    """YOLO `(cx, cy, w, h)` normalized → pixel `(x1, y1, x2, y2)` ints."""
    cx, cy, bw, bh = box_norm
    x1 = int(round((cx - bw / 2) * img_w))
    y1 = int(round((cy - bh / 2) * img_h))
    x2 = int(round((cx + bw / 2) * img_w))
    y2 = int(round((cy + bh / 2) * img_h))
    return x1, y1, x2, y2


def render_class_aabb_edges(
    boxes_yolo_norm: np.ndarray,
    classes: np.ndarray,
    img_size: tuple,
    target_class: int,
    dilate_px: int = 2,
) -> np.ndarray:
    """Render a binary edge map (uint8, 0/255) of dilated AABB *outlines* for
    boxes of `target_class` only.

    Operates on axis-aligned bounding boxes — when OBB ground truth lands, a
    separate `render_class_obb_edges` helper should be added rather than
    overloading this one.

    `boxes_yolo_norm` is shape (N, 4) in YOLO (cx, cy, w, h) normalised coords.
    `img_size` is (H, W).
    Returns (H, W) uint8 with values in {0, 255}.
    """
    h, w = img_size
    canvas = np.zeros((h, w), dtype=np.uint8)
    for box, c in zip(boxes_yolo_norm, classes):
        if int(c) != int(target_class):
            continue
        x1, y1, x2, y2 = _yolo_to_xyxy(box, w, h)
        x1, x2 = sorted((max(0, x1), min(w - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(h - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, thickness=1)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        canvas = cv2.dilate(canvas, kernel)
    return canvas


def f_score_with_tolerance(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    tolerance_px: int = 2,
) -> float:
    """Pixel-wise F1 with a `tolerance_px` morphological slack on both sides.

    A predicted edge counts as TP if there is a GT edge within `tolerance_px`,
    and a GT edge counts as recalled if there is a prediction within `tolerance_px`.
    """
    if pred_bin.dtype != bool:
        pred_bin = pred_bin.astype(bool)
    if gt_bin.dtype != bool:
        gt_bin = gt_bin.astype(bool)

    if not pred_bin.any() and not gt_bin.any():
        return 1.0
    if not pred_bin.any() or not gt_bin.any():
        return 0.0

    k = 2 * tolerance_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    pred_dil = cv2.dilate(pred_bin.astype(np.uint8), kernel).astype(bool)
    gt_dil = cv2.dilate(gt_bin.astype(np.uint8), kernel).astype(bool)

    tp_pred = int((pred_bin & gt_dil).sum())
    tp_gt = int((gt_bin & pred_dil).sum())
    fp = int((pred_bin & ~gt_dil).sum())
    fn = int((gt_bin & ~pred_dil).sum())

    precision = tp_pred / max(tp_pred + fp, 1)
    recall = tp_gt / max(tp_gt + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ods_per_image(
    pred_u8: np.ndarray,
    gt_bin: np.ndarray,
    thresholds: np.ndarray = None,
    tolerance_px: int = 2,
) -> tuple:
    """Single-image ODS: sweep `thresholds`, return (best_threshold, best_f).

    `pred_u8`: HxW uint8 saliency map.
    `gt_bin`:  HxW bool edge mask.
    """
    if thresholds is None:
        thresholds = np.linspace(20, 235, 30, dtype=np.uint8)
    best_t, best_f = int(thresholds[0]), 0.0
    for t in thresholds:
        pb = pred_u8 > int(t)
        f = f_score_with_tolerance(pb, gt_bin, tolerance_px=tolerance_px)
        if f > best_f:
            best_f, best_t = f, int(t)
    return best_t, best_f
