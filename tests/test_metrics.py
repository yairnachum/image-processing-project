import numpy as np

from src.metrics import iou_xyxy, ap_per_class
from src.metrics import render_class_edge_map, f_score_with_tolerance, ods_per_image


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


def test_render_class_edge_map_draws_rectangles():
    """A single AABB centered at (0.5, 0.5) with size (0.4, 0.4) on a 100x100 image
    should produce a hollow rectangle outline (0/255), not a filled rectangle."""
    boxes = np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float32)
    classes = np.array([0], dtype=np.int32)
    edge = render_class_edge_map(boxes, classes, img_size=(100, 100), target_class=0, dilate_px=1)
    assert edge.shape == (100, 100)
    assert edge.dtype == np.uint8
    assert set(np.unique(edge).tolist()) <= {0, 255}
    # Center of the box should be 0 (interior is hollow); edge border should have 255s.
    assert edge[50, 50] == 0
    assert edge[30, 30] == 255 or edge[31, 30] == 255 or edge[30, 31] == 255   # top-left corner area


def test_render_class_edge_map_filters_by_class():
    boxes = np.array([
        [0.25, 0.5, 0.2, 0.2],   # class 0
        [0.75, 0.5, 0.2, 0.2],   # class 1
    ], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int32)
    edge = render_class_edge_map(boxes, classes, img_size=(100, 100), target_class=0, dilate_px=1)
    # Left half (x ~ 25) should have edges, right half (x ~ 75) should be black.
    assert edge[:, 0:50].sum() > 0
    assert edge[:, 50:].sum() == 0


def test_f_score_with_tolerance_identical_edges_is_one():
    a = np.zeros((50, 50), dtype=bool)
    a[20:30, 20:30] = True
    assert abs(f_score_with_tolerance(a, a, tolerance_px=1) - 1.0) < 1e-6


def test_f_score_with_tolerance_disjoint_is_zero():
    a = np.zeros((50, 50), dtype=bool)
    b = np.zeros((50, 50), dtype=bool)
    a[5, 5] = True
    b[40, 40] = True
    assert f_score_with_tolerance(a, b, tolerance_px=1) == 0.0


def test_ods_per_image_picks_best_threshold():
    """A saliency map with a single hot region; ODS should pick a threshold that recovers it."""
    pred = np.zeros((50, 50), dtype=np.uint8)
    pred[20:30, 20:30] = 200    # strong "edges" inside the box
    gt = np.zeros((50, 50), dtype=bool)
    gt[20:30, 20:30] = True
    best_t, best_f = ods_per_image(pred, gt, thresholds=np.array([50, 100, 150, 220], dtype=np.uint8))
    assert best_t in {50, 100, 150}, f"expected a threshold below 200, got {best_t}"
    assert best_f > 0.5
