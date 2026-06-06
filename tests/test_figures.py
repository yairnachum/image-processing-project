from pathlib import Path

import pandas as pd

from src.figures import plot_perclass_bar


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
