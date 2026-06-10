from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.figures import plot_perclass_bar, plot_predictions_grid, plot_distortion_grid, plot_metric_vs_snr, plot_distorted_vs_restored


def test_plot_perclass_bar_writes_png(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"class_id": 0, "class_name": "plane", "ap_iou50": 0.3},
            {"class_id": 1, "class_name": "ship", "ap_iou50": 0.1},
            {"class_id": 2, "class_name": "storage-tank", "ap_iou50": float("nan")},
        ]
    )
    csv = tmp_path / "perclass.csv"
    df.to_csv(csv, index=False)

    out = tmp_path / "fig.png"
    plot_perclass_bar(
        csv,
        value_col="ap_iou50",
        title="Test title",
        ylabel="AP@0.5",
        out_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_predictions_grid_writes_png(tmp_path: Path):
    clean_root = tmp_path / "data" / "clean"
    outputs_root = tmp_path / "outputs"
    img_dir = clean_root / "test" / "images"
    lbl_dir = clean_root / "test" / "labels"
    det_dir = outputs_root / "clean" / "detections"
    edge_dir = outputs_root / "clean" / "edges"
    for d in (img_dir, lbl_dir, det_dir, edge_dir):
        d.mkdir(parents=True)
    for name in ("tile_0", "tile_1"):
        cv2.imwrite(str(img_dir / f"{name}.png"), np.full((256, 256, 3), 80, dtype=np.uint8))
        (lbl_dir / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        (det_dir / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.4 0.85\n")
        cv2.imwrite(str(edge_dir / f"{name}.png"), np.full((256, 256), 120, dtype=np.uint8))

    out = tmp_path / "grid.png"
    plot_predictions_grid(
        clean_root,
        outputs_root,
        sample_names=["tile_0", "tile_1"],
        out_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_distortion_grid_writes_png(tmp_path: Path):
    clean_dir = tmp_path / "clean" / "test" / "images"
    dist_root = tmp_path / "distorted"
    clean_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        cv2.imwrite(
            str(clean_dir / f"tile_{i}.png"),
            rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8),
        )
    levels = ["0.5", "1.5", "3.0"]
    for lvl in levels:
        out_dir = dist_root / "haze" / lvl / "test" / "images"
        out_dir.mkdir(parents=True)
        for i in range(4):
            cv2.imwrite(
                str(out_dir / f"tile_{i}.png"),
                rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8),
            )

    out = tmp_path / "grid.png"
    plot_distortion_grid(
        clean_root=tmp_path / "clean",
        distorted_root=dist_root,
        distortion="haze",
        levels=levels,
        sample_names=[f"tile_{i}" for i in range(4)],
        out_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_metric_vs_snr_writes_png(tmp_path: Path):
    df = pd.DataFrame([
        {"distortion": "haze",  "snr_db_mean": 1.0, "value": 0.10},
        {"distortion": "haze",  "snr_db_mean": 7.0, "value": 0.30},
        {"distortion": "jpeg",  "snr_db_mean": 13.0, "value": 0.20},
        {"distortion": "jpeg",  "snr_db_mean": 22.0, "value": 0.45},
        {"distortion": "noise", "snr_db_mean": 8.0, "value": 0.18},
        {"distortion": "noise", "snr_db_mean": 23.0, "value": 0.40},
    ])
    out = tmp_path / "curve.png"
    plot_metric_vs_snr(
        df,
        value_col="value",
        title="Test",
        ylabel="value",
        out_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_distorted_vs_restored_writes_png(tmp_path: Path):
    rows_d = []
    rows_r = []
    for d in ["haze", "jpeg", "noise"]:
        for snr, v_d, v_r in [(0.0, 0.1, 0.2), (5.0, 0.3, 0.5), (10.0, 0.4, 0.6)]:
            rows_d.append({"distortion": d, "snr_db_mean": snr, "value": v_d})
            rows_r.append({"distortion": d, "snr_db_mean": snr, "value": v_r})
    df_d = pd.DataFrame(rows_d)
    df_r = pd.DataFrame(rows_r)
    out = tmp_path / "recovery.png"
    plot_distorted_vs_restored(
        df_distorted=df_d, df_restored=df_r,
        value_col="value", title="Test recovery", ylabel="value",
        out_path=out, clean_baseline=0.7,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_distorted_vs_restored_with_third_curve(tmp_path: Path):
    rows_d, rows_r, rows_f = [], [], []
    for d in ["haze", "jpeg", "noise"]:
        for snr, v_d, v_r, v_f in [(0.0, 0.1, 0.2, 0.3), (5.0, 0.3, 0.5, 0.6), (10.0, 0.4, 0.6, 0.7)]:
            rows_d.append({"distortion": d, "snr_db_mean": snr, "value": v_d})
            rows_r.append({"distortion": d, "snr_db_mean": snr, "value": v_r})
            rows_f.append({"distortion": d, "snr_db_mean": snr, "value": v_f})
    out = tmp_path / "threeway.png"
    plot_distorted_vs_restored(
        df_distorted=pd.DataFrame(rows_d), df_restored=pd.DataFrame(rows_r),
        value_col="value", title="Three-way", ylabel="value",
        out_path=out, clean_baseline=0.7,
        df_third=pd.DataFrame(rows_f), third_label="fine-tuned",
    )
    assert out.exists()
    assert out.stat().st_size > 1000
