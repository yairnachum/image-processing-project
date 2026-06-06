"""Detection + edge metrics primitives.

All boxes use xyxy in pixel coordinates unless otherwise noted.
"""

from typing import Dict, List

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
