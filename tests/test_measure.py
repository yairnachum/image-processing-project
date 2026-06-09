from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.measure import measure_clean_stage


def _make_fake_run(tmp_path: Path) -> tuple:
    """Create a fake clean-stage layout the orchestrator can consume."""
    clean_root = tmp_path / "data" / "clean"
    outputs_root = tmp_path / "outputs"
    results_root = tmp_path / "results"

    img_dir = clean_root / "test" / "images"
    lbl_dir = clean_root / "test" / "labels"
    det_dir = outputs_root / "clean" / "detections"
    edge_dir = outputs_root / "clean" / "edges"
    for d in (img_dir, lbl_dir, det_dir, edge_dir):
        d.mkdir(parents=True)

    for name in ("tile_0", "tile_1"):
        # 256x256 image, class-0 GT at center
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{name}.png"), img)
        (lbl_dir / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        # YOLO prediction in the same place (perfect detection at conf 0.9)
        (det_dir / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.4 0.9\n")
        # HED edges: white outline matching GT roughly
        edge = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(edge, (76, 76), (180, 180), 200, thickness=2)
        cv2.imwrite(str(edge_dir / f"{name}.png"), edge)
    return clean_root, results_root, outputs_root


def test_measure_clean_writes_perclass_detections_csv(tmp_path: Path):
    clean_root, results_root, outputs_root = _make_fake_run(tmp_path)
    measure_clean_stage(
        clean_root=clean_root,
        results_root=results_root,
        outputs_root=outputs_root,
    )
    csv = results_root / "clean" / "perclass_detections.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert {"class_id", "class_name", "n_gt", "n_pred", "ap_iou50"} <= set(df.columns)
    row0 = df[df["class_id"] == 0].iloc[0]
    assert row0["n_gt"] == 2
    assert row0["n_pred"] == 2
    assert abs(row0["ap_iou50"] - 1.0) < 1e-3


def test_measure_clean_writes_edge_metrics_csv(tmp_path: Path):
    clean_root, results_root, outputs_root = _make_fake_run(tmp_path)
    measure_clean_stage(
        clean_root=clean_root,
        results_root=results_root,
        outputs_root=outputs_root,
    )
    csv = results_root / "clean" / "edge_metrics.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert {"name", "ods_threshold", "ods_f_score"} <= set(df.columns)
    assert len(df) == 2
    assert (df["ods_f_score"] > 0).all()


def test_measure_clean_writes_perclass_edges_csv(tmp_path: Path):
    clean_root, results_root, outputs_root = _make_fake_run(tmp_path)
    measure_clean_stage(
        clean_root=clean_root,
        results_root=results_root,
        outputs_root=outputs_root,
    )
    csv = results_root / "clean" / "perclass_edges.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert {"class_id", "class_name", "n_gt_px", "f_score"} <= set(df.columns)
    row0 = df[df["class_id"] == 0].iloc[0]
    assert row0["n_gt_px"] > 0
    assert row0["f_score"] > 0


def _make_fake_distorted_run(tmp_path: Path) -> tuple:
    """Mimic a distorted/haze/0.5-style stage with GT pointing back at clean."""
    clean_root = tmp_path / "data" / "clean"
    distorted_root = tmp_path / "data" / "distorted"
    outputs_root = tmp_path / "outputs"
    results_root = tmp_path / "results"

    clean_lbl_dir = clean_root / "test" / "labels"
    dist_img_dir  = distorted_root / "haze" / "0.5" / "test" / "images"
    det_dir       = outputs_root / "distorted" / "haze" / "0.5" / "detections"
    edge_dir      = outputs_root / "distorted" / "haze" / "0.5" / "edges"
    for d in (clean_lbl_dir, dist_img_dir, det_dir, edge_dir):
        d.mkdir(parents=True)

    for name in ("tile_0", "tile_1"):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(dist_img_dir / f"{name}.png"), img)
        (clean_lbl_dir / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        (det_dir / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.4 0.9\n")
        edge = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(edge, (76, 76), (180, 180), 200, thickness=2)
        cv2.imwrite(str(edge_dir / f"{name}.png"), edge)
    return clean_root, distorted_root, results_root, outputs_root


def test_measure_stage_with_explicit_gt_label_dir(tmp_path: Path):
    from src.measure import measure_stage

    clean_root, distorted_root, results_root, outputs_root = _make_fake_distorted_run(tmp_path)
    measure_stage(
        stage="distorted/haze/0.5",
        image_dir=distorted_root / "haze" / "0.5" / "test" / "images",
        gt_label_dir=clean_root / "test" / "labels",
        results_root=results_root,
        outputs_root=outputs_root,
    )
    out_dir = results_root / "distorted" / "haze" / "0.5"
    for csv in ("perclass_detections.csv", "edge_metrics.csv", "perclass_edges.csv"):
        assert (out_dir / csv).exists(), csv
    df = pd.read_csv(out_dir / "perclass_detections.csv")
    row0 = df[df["class_id"] == 0].iloc[0]
    assert row0["n_gt"] == 2
    assert row0["n_pred"] == 2
    assert abs(row0["ap_iou50"] - 1.0) < 1e-3
