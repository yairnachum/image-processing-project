import numpy as np

from src.metrics import iou_xyxy, ap_per_class


def test_iou_xyxy_identical_boxes():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    assert iou_xyxy(a, a) == 1.0


def test_iou_xyxy_disjoint_boxes():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([20.0, 20.0, 30.0, 30.0])
    assert iou_xyxy(a, b) == 0.0


def test_iou_xyxy_half_overlap():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([5.0, 0.0, 15.0, 10.0])   # overlap = 50 / union = 150 = 1/3
    assert abs(iou_xyxy(a, b) - (1 / 3)) < 1e-6


def test_ap_per_class_perfect_predictions():
    preds = [
        {"image_id": "a", "class_id": 0, "score": 0.9, "xyxy": np.array([0., 0., 10., 10.])},
        {"image_id": "b", "class_id": 0, "score": 0.8, "xyxy": np.array([0., 0., 10., 10.])},
    ]
    gt = [
        {"image_id": "a", "class_id": 0, "xyxy": np.array([0., 0., 10., 10.])},
        {"image_id": "b", "class_id": 0, "xyxy": np.array([0., 0., 10., 10.])},
    ]
    result = ap_per_class(preds, gt, num_classes=2, iou_thresh=0.5)
    assert abs(result[0]["ap"] - 1.0) < 1e-6
    assert result[0]["n_gt"] == 2
    assert result[0]["n_pred"] == 2
    # Class 1 has no GT and no preds — AP should be NaN, not 0.
    assert np.isnan(result[1]["ap"])
    assert result[1]["n_gt"] == 0


def test_ap_per_class_missed_detection_drops_recall():
    # One image, one class, GT with two boxes but only one detection (the easy one).
    preds = [
        {"image_id": "a", "class_id": 0, "score": 0.9, "xyxy": np.array([0., 0., 10., 10.])},
    ]
    gt = [
        {"image_id": "a", "class_id": 0, "xyxy": np.array([0., 0., 10., 10.])},
        {"image_id": "a", "class_id": 0, "xyxy": np.array([100., 100., 110., 110.])},
    ]
    result = ap_per_class(preds, gt, num_classes=1, iou_thresh=0.5)
    # Recall is 0.5 (1 of 2 matched), precision is 1.0. All-points AP = 0.5.
    assert abs(result[0]["ap"] - 0.5) < 1e-6


def test_ap_per_class_false_positive_drops_precision():
    # One image, one class, two preds, only one true.
    # FP is ranked ABOVE the TP, so the PR curve hits precision=0 before climbing to 0.5.
    preds = [
        {"image_id": "a", "class_id": 0, "score": 0.9, "xyxy": np.array([200., 200., 210., 210.])},  # FP
        {"image_id": "a", "class_id": 0, "score": 0.8, "xyxy": np.array([0., 0., 10., 10.])},        # TP
    ]
    gt = [
        {"image_id": "a", "class_id": 0, "xyxy": np.array([0., 0., 10., 10.])},
    ]
    result = ap_per_class(preds, gt, num_classes=1, iou_thresh=0.5)
    # Walk through PR: FP first → (recall=0, precision=0). Then TP → (recall=1, precision=0.5).
    # All-points AP: precision is 0.5 across all reachable recall, so AP = 0.5.
    assert abs(result[0]["ap"] - 0.5) < 1e-6
