from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.figures import plot_perclass_bar, plot_predictions_grid, plot_distortion_grid


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
